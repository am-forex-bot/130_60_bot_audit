#!/usr/bin/env python3
"""
Simulate stepped profit lock + momentum fade exit on real S5 (5-second) candle data.

USAGE:
  1. Place your S5 parquet files in a folder, named like:
       GBP_USD_S5_20191101_20260303.parquet
       AUD_USD_S5_20191101_20260303.parquet
       EUR_USD_S5_20191101_20260303.parquet

  2. Run:  python3 simulate_exits.py ./data_folder/

  Requirements: pip install pandas pyarrow numpy

  The script will:
  - Load only the relevant date range from each parquet (fast, skips years of data)
  - Replay every 5-second bar for each trade (near-tick precision)
  - Aggregate S5→M5 for RSI/MACD momentum fade indicators
  - Tell you exactly what each exit strategy would have done
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional


# ==================== CONFIGURATION ====================

# Today's actual trades from OANDA transactions (Mar 3, 2026)
TRADES = [
    # Trades that CLOSED today (we simulate what WOULD have happened with new exits)
    {
        "id": "GBP_USD_1",
        "pair": "GBP_USD",
        "direction": "sell",
        "entry_price": 1.32653,
        "entry_time": "2026-03-03T11:17:32",
        "actual_exit_price": 1.33230,
        "actual_exit_time": "2026-03-03T12:20:36",
        "actual_exit_reason": "SL",
        "actual_pnl_gbp": -661.03,
        "sl_price": 1.33228,
        "tp_price": 1.31328,
        "was_post_tp_reentry": True,
        "units": 151863,
    },
    {
        "id": "AUD_USD_1",
        "pair": "AUD_USD",
        "direction": "sell",
        "entry_price": 0.69653,
        "entry_time": "2026-03-03T15:05:30",
        "actual_exit_price": 0.70227,
        "actual_exit_time": "2026-03-03T16:45:01",
        "actual_exit_reason": "SL",
        "actual_pnl_gbp": -666.29,
        "sl_price": 0.70227,
        "tp_price": 0.68327,
        "was_post_tp_reentry": True,
        "units": 153866,
    },
    {
        "id": "GBP_USD_2",
        "pair": "GBP_USD",
        "direction": "sell",
        "entry_price": 1.33021,
        "entry_time": "2026-03-03T13:45:54",
        "actual_exit_price": 1.33603,
        "actual_exit_time": "2026-03-03T18:41:08",
        "actual_exit_reason": "SL",
        "actual_pnl_gbp": -647.06,
        "sl_price": 1.33602,
        "tp_price": 1.31702,
        "was_post_tp_reentry": False,  # 85 min after SL, not TP
        "units": 147791,
    },
    # Winning trades (to verify profit lock doesn't interfere)
    {
        "id": "EUR_USD_TP",
        "pair": "EUR_USD",
        "direction": "sell",
        "entry_price": 1.17226,  # Approximate — pre-existing position
        "entry_time": "2026-03-02T00:00:00",  # Approximate
        "actual_exit_price": 1.15926,
        "actual_exit_time": "2026-03-03T10:38:27",
        "actual_exit_reason": "TP",
        "actual_pnl_gbp": 1263.55,
        "sl_price": 1.17802,
        "tp_price": 1.15926,
        "was_post_tp_reentry": False,
        "units": 127010,
    },
    {
        "id": "GBP_USD_TP",
        "pair": "GBP_USD",
        "direction": "sell",
        "entry_price": 1.33998,  # Approximate
        "entry_time": "2026-03-02T00:00:00",
        "actual_exit_price": 1.32698,
        "actual_exit_time": "2026-03-03T11:16:24",
        "actual_exit_reason": "TP",
        "actual_pnl_gbp": 1226.69,
        "sl_price": 1.34574,
        "tp_price": 1.32698,
        "was_post_tp_reentry": False,
        "units": 123277,
    },
    {
        "id": "AUD_USD_TP",
        "pair": "AUD_USD",
        "direction": "sell",
        "entry_price": 0.71018,  # Approximate
        "entry_time": "2026-03-02T00:00:00",
        "actual_exit_price": 0.69688,
        "actual_exit_time": "2026-03-03T15:04:21",
        "actual_exit_reason": "TP",
        "actual_pnl_gbp": 1622.21,
        "sl_price": 0.71594,
        "tp_price": 0.69688,
        "was_post_tp_reentry": False,
        "units": 163450,
    },
]

# Profit lock levels: (trigger_pips, lock_pips)
PROFIT_LOCK_LEVELS = [
    (60, 0),     # At +60p: move SL to breakeven
    (80, 30),    # At +80p: lock +30 pips
    (100, 50),   # At +100p: lock +50 pips
    (115, 70),   # At +115p: lock +70 pips
]

# Momentum fade parameters (calculated on M5 bars)
FADE_MIN_PROFIT = 50    # pips — must be this far in profit
FADE_RSI_PERIOD = 14
FADE_RSI_REVERSAL = 15  # RSI must reverse by this many points
FADE_MACD_FAST = 12
FADE_MACD_SLOW = 26
FADE_MACD_SIGNAL = 9

# How far back to load for indicator warmup
DATA_LOOKBACK_DAYS = 3  # Load from Mar 1 for trades on Mar 2-3


# ==================== HELPERS ====================

def pip_mult(pair):
    return 100 if 'JPY' in pair else 10000


def pip_val(pair):
    return 0.01 if 'JPY' in pair else 0.0001


def calc_profit_pips(pair, direction, entry, current):
    mult = pip_mult(pair)
    if direction == 'buy':
        return (current - entry) * mult
    else:
        return (entry - current) * mult


def calc_rsi(closes, period=14):
    """Calculate RSI from close prices array."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)

    if len(gains) < period:
        return np.full_like(closes, 50.0)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd_hist(closes, fast=12, slow=26, signal=9):
    """Calculate MACD histogram from close prices."""
    if len(closes) < slow + signal:
        return np.zeros_like(closes)

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _ema(data, period):
    """Exponential moving average."""
    ema = np.zeros_like(data, dtype=float)
    multiplier = 2.0 / (period + 1)
    ema[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


# ==================== LOAD S5 DATA ====================

def find_s5_file(data_dir: str, pair: str) -> Optional[str]:
    """Find S5 parquet/feather file for a pair using glob patterns."""
    # Try parquet first, then feather
    for ext in ['parquet', 'feather']:
        # Pattern: GBP_USD_S5_*.parquet
        pattern = os.path.join(data_dir, f"{pair}_S5_*.{ext}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

        # Also try without underscore: GBPUSD_S5_*.parquet
        pair_no_sep = pair.replace('_', '')
        pattern = os.path.join(data_dir, f"{pair_no_sep}_S5_*.{ext}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

        # Exact name: GBP_USD_S5.parquet
        fp = os.path.join(data_dir, f"{pair}_S5.{ext}")
        if os.path.exists(fp):
            return fp

    return None


def load_s5_data(data_dir: str, pair: str) -> Optional[pd.DataFrame]:
    """Load S5 candle data from parquet, filtering to only the relevant date range.

    Uses pyarrow predicate pushdown to skip reading years of irrelevant data.
    Returns a DataFrame with columns: time, open, high, low, close
    Time column is naive datetime (UTC stripped).
    """
    filepath = find_s5_file(data_dir, pair)
    if not filepath:
        print(f"  WARNING: No S5 data found for {pair} in {data_dir}")
        print(f"  Expected: {pair}_S5_YYYYMMDD_YYYYMMDD.parquet")
        return None

    ext = os.path.splitext(filepath)[1].lower()
    basename = os.path.basename(filepath)
    print(f"  Loading {basename}...")

    # Calculate date range: we need indicator warmup before earliest trade
    earliest_trade = min(datetime.fromisoformat(t['entry_time']) for t in TRADES)
    load_from = earliest_trade - timedelta(days=DATA_LOOKBACK_DAYS)
    load_from_ts = pd.Timestamp(load_from, tz='UTC')

    try:
        if ext == '.parquet':
            # Use pyarrow predicate pushdown — only reads relevant row groups
            df = pd.read_parquet(
                filepath,
                filters=[('time', '>=', load_from_ts)],
                columns=['time', 'open', 'high', 'low', 'close', 'volume'],
            )
        else:
            # Feather doesn't support predicate pushdown, load all and filter
            df = pd.read_feather(filepath)
            df = df[df['time'] >= load_from_ts]
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None

    if df.empty:
        print(f"  WARNING: No data found for {pair} after {load_from}")
        return None

    # Strip timezone to naive datetime for simpler comparison
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)

    df = df.sort_values('time').reset_index(drop=True)

    t0 = df['time'].iloc[0]
    t1 = df['time'].iloc[-1]
    print(f"  Loaded {len(df):,} S5 bars for {pair} ({t0} to {t1})")

    return df


def aggregate_s5_to_m5(s5_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate S5 (5-second) bars into M5 (5-minute) bars."""
    df = s5_df.set_index('time')
    m5 = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
    m5 = m5.reset_index()
    return m5


def build_m5_indicators(m5_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute RSI and MACD histogram on M5 data."""
    closes = m5_df['close'].values

    rsi = calc_rsi(closes, FADE_RSI_PERIOD)
    macd_hist = calc_macd_hist(closes, FADE_MACD_FAST, FADE_MACD_SLOW, FADE_MACD_SIGNAL)

    m5_df = m5_df.copy()
    m5_df['rsi'] = rsi
    m5_df['macd_hist'] = macd_hist
    return m5_df


# ==================== SIMULATE ====================

def simulate_trade(trade: Dict, s5_df: pd.DataFrame, m5_ind: pd.DataFrame) -> Dict:
    """Simulate a single trade against S5 candle data with all three exit strategies.

    Uses S5 bars for price-level checks (SL/TP/profit lock) — 5-second precision.
    Uses pre-computed M5 indicators for momentum fade detection.
    """

    pair = trade['pair']
    direction = trade['direction']
    entry = trade['entry_price']
    entry_time = datetime.fromisoformat(trade['entry_time'])
    sl = trade['sl_price']
    tp = trade['tp_price']
    mult = pip_mult(pair)
    pv = pip_val(pair)

    # Filter S5 bars to trade's lifetime
    trade_bars = s5_df[s5_df['time'] >= entry_time].copy()

    if trade_bars.empty:
        return {"error": f"No S5 data found for {pair} after {entry_time}"}

    # Convert M5 indicators to a lookup: for any time, find nearest completed M5 bar
    m5_times = m5_ind['time'].values  # numpy datetime64 array
    m5_rsi = m5_ind['rsi'].values
    m5_macd = m5_ind['macd_hist'].values

    # Track state
    max_profit_pips = 0.0
    min_profit_pips = 0.0
    current_sl = sl  # Will be ratcheted by profit lock
    max_profit_time = entry_time

    # Results
    result = {
        "trade_id": trade['id'],
        "pair": pair,
        "direction": direction,
        "entry": entry,
        "actual_exit": trade['actual_exit_reason'],
        "actual_pnl_pips": calc_profit_pips(pair, direction, entry, trade['actual_exit_price']),
        "actual_pnl_gbp": trade['actual_pnl_gbp'],
        "was_post_tp_reentry": trade['was_post_tp_reentry'],

        # Profit lock tracking
        "lock_levels_hit": [],
        "lock_exit_price": None,
        "lock_exit_time": None,
        "lock_exit_pips": None,
        "lock_exit_reason": None,

        # Momentum fade tracking
        "fade_triggered": False,
        "fade_exit_price": None,
        "fade_exit_time": None,
        "fade_exit_pips": None,
        "fade_rsi_at_trigger": None,

        # Combined (whichever fires first)
        "combined_exit_price": None,
        "combined_exit_time": None,
        "combined_exit_pips": None,
        "combined_exit_reason": None,

        # Excursion data
        "max_favorable_pips": 0.0,
        "max_adverse_pips": 0.0,
        "max_profit_time": None,
        "seconds_in_green": 0,
        "seconds_in_red": 0,
        "total_bars": 0,

        # Per-minute trace (sampled from S5 for reasonable file size)
        "bar_trace": [],
    }

    combined_exited = False
    lock_exited = False
    fade_exited = False
    trace_counter = 0

    # Iterate S5 bars
    times = trade_bars['time'].values
    opens = trade_bars['open'].values
    highs = trade_bars['high'].values
    lows = trade_bars['low'].values
    closes = trade_bars['close'].values

    for i in range(len(trade_bars)):
        bar_time = pd.Timestamp(times[i]).to_pydatetime()
        bar_open = opens[i]
        bar_high = highs[i]
        bar_low = lows[i]
        bar_close = closes[i]

        # Calculate profit at extremes
        if direction == 'sell':
            bar_best = bar_low
            bar_worst = bar_high
        else:
            bar_best = bar_high
            bar_worst = bar_low

        profit_at_best = calc_profit_pips(pair, direction, entry, bar_best)
        profit_at_worst = calc_profit_pips(pair, direction, entry, bar_worst)
        profit_at_close = calc_profit_pips(pair, direction, entry, bar_close)

        # Track max excursion
        if profit_at_best > max_profit_pips:
            max_profit_pips = profit_at_best
            max_profit_time = bar_time
        if profit_at_worst < min_profit_pips:
            min_profit_pips = profit_at_worst

        if profit_at_close > 0:
            result['seconds_in_green'] += 5
        else:
            result['seconds_in_red'] += 5
        result['total_bars'] += 1

        # Sample trace every 12 bars (= 1 minute)
        trace_counter += 1
        if trace_counter >= 12:
            # Look up M5 RSI for this time
            m5_idx = np.searchsorted(m5_times, times[i], side='right') - 1
            current_m5_rsi = float(m5_rsi[m5_idx]) if 0 <= m5_idx < len(m5_rsi) else None

            result['bar_trace'].append({
                'time': bar_time.strftime('%H:%M:%S'),
                'price': round(bar_close, 5),
                'profit_pips': round(profit_at_close, 1),
                'mfe': round(max_profit_pips, 1),
                'rsi_m5': round(current_m5_rsi, 1) if current_m5_rsi else None,
            })
            trace_counter = 0

        # --- CHECK ORIGINAL SL/TP ---
        hit_original_sl = (
            (direction == 'sell' and bar_high >= sl) or
            (direction == 'buy' and bar_low <= sl)
        )
        hit_original_tp = (
            (direction == 'sell' and bar_low <= tp) or
            (direction == 'buy' and bar_high >= tp)
        )

        # --- PROFIT LOCK: Check if we should ratchet SL ---
        if not lock_exited:
            for trigger_pips, lock_pips in sorted(PROFIT_LOCK_LEVELS, reverse=True):
                if profit_at_best >= trigger_pips:
                    # Calculate new SL
                    if direction == 'buy':
                        new_sl = entry + (lock_pips * pv)
                        sl_improved = new_sl > current_sl
                    else:
                        new_sl = entry - (lock_pips * pv)
                        sl_improved = new_sl < current_sl

                    if sl_improved:
                        level_name = f"+{trigger_pips}p→lock+{lock_pips}p"
                        if level_name not in [l['level'] for l in result['lock_levels_hit']]:
                            result['lock_levels_hit'].append({
                                'level': level_name,
                                'time': bar_time.strftime('%Y-%m-%dT%H:%M:%S'),
                                'profit_at_trigger': round(profit_at_best, 1),
                            })
                        current_sl = new_sl
                    break

            # Check if ratcheted SL is now hit (only if it's been moved from original)
            if current_sl != sl:
                hit_ratcheted_sl = (
                    (direction == 'sell' and bar_high >= current_sl) or
                    (direction == 'buy' and bar_low <= current_sl)
                )
            else:
                hit_ratcheted_sl = False

            if hit_ratcheted_sl:
                lock_pips_captured = calc_profit_pips(pair, direction, entry, current_sl)
                result['lock_exit_price'] = current_sl
                result['lock_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                result['lock_exit_pips'] = round(lock_pips_captured, 1)
                result['lock_exit_reason'] = 'profit_lock_sl'
                lock_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = current_sl
                    result['combined_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                    result['combined_exit_pips'] = round(lock_pips_captured, 1)
                    result['combined_exit_reason'] = 'profit_lock'
                    combined_exited = True

            elif hit_original_tp and not lock_exited:
                tp_pips = calc_profit_pips(pair, direction, entry, tp)
                result['lock_exit_price'] = tp
                result['lock_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                result['lock_exit_pips'] = round(tp_pips, 1)
                result['lock_exit_reason'] = 'take_profit'
                lock_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = tp
                    result['combined_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                    result['combined_exit_pips'] = round(tp_pips, 1)
                    result['combined_exit_reason'] = 'take_profit'
                    combined_exited = True

            elif hit_original_sl and not lock_exited:
                sl_pips = calc_profit_pips(pair, direction, entry, sl)
                result['lock_exit_price'] = sl
                result['lock_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                result['lock_exit_pips'] = round(sl_pips, 1)
                result['lock_exit_reason'] = 'stop_loss'
                lock_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = sl
                    result['combined_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                    result['combined_exit_pips'] = round(sl_pips, 1)
                    result['combined_exit_reason'] = 'stop_loss'
                    combined_exited = True

        # --- MOMENTUM FADE: Check if momentum is dying while in profit ---
        if not fade_exited and profit_at_close >= FADE_MIN_PROFIT:
            # Look up M5 indicators at current time
            m5_idx = np.searchsorted(m5_times, times[i], side='right') - 1
            if m5_idx >= 5:  # Need at least 5 M5 bars of history
                current_rsi = m5_rsi[m5_idx]
                rsi_5_bars_ago = m5_rsi[m5_idx - 5]
                current_macd = m5_macd[m5_idx]
                prev_macd = m5_macd[m5_idx - 3]

                fade_triggered = False
                if direction == 'sell':
                    # Short: RSI was low (oversold) and now rising = momentum fading
                    rsi_reversal = current_rsi - rsi_5_bars_ago
                    macd_fading = current_macd > prev_macd
                    if rsi_reversal > FADE_RSI_REVERSAL and macd_fading:
                        fade_triggered = True
                else:
                    # Long: RSI was high (overbought) and now dropping
                    rsi_reversal = rsi_5_bars_ago - current_rsi
                    macd_fading = current_macd < prev_macd
                    if rsi_reversal > FADE_RSI_REVERSAL and macd_fading:
                        fade_triggered = True

                if fade_triggered:
                    result['fade_triggered'] = True
                    result['fade_exit_price'] = bar_close
                    result['fade_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                    result['fade_exit_pips'] = round(profit_at_close, 1)
                    result['fade_rsi_at_trigger'] = round(float(current_rsi), 1)
                    fade_exited = True

                    if not combined_exited:
                        result['combined_exit_price'] = bar_close
                        result['combined_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                        result['combined_exit_pips'] = round(profit_at_close, 1)
                        result['combined_exit_reason'] = 'momentum_fade'
                        combined_exited = True

        # If both lock and fade have resolved, stop
        if lock_exited and fade_exited:
            break
        # If original SL/TP hit, ensure everything is recorded and stop
        if hit_original_sl or hit_original_tp:
            if not lock_exited:
                exit_pips = calc_profit_pips(pair, direction, entry, trade['actual_exit_price'])
                result['lock_exit_price'] = trade['actual_exit_price']
                result['lock_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                result['lock_exit_pips'] = round(exit_pips, 1)
                result['lock_exit_reason'] = trade['actual_exit_reason']
                lock_exited = True
            if not combined_exited:
                exit_pips = calc_profit_pips(pair, direction, entry, trade['actual_exit_price'])
                result['combined_exit_price'] = trade['actual_exit_price']
                result['combined_exit_time'] = bar_time.strftime('%Y-%m-%dT%H:%M:%S')
                result['combined_exit_pips'] = round(exit_pips, 1)
                result['combined_exit_reason'] = trade['actual_exit_reason']
                combined_exited = True
            break

    result['max_favorable_pips'] = round(max_profit_pips, 1)
    result['max_adverse_pips'] = round(abs(min_profit_pips), 1)
    result['max_profit_time'] = max_profit_time.strftime('%Y-%m-%dT%H:%M:%S')

    return result


# ==================== MAIN ====================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:  python3 simulate_exits.py ./your_data_folder/")
        print("\nExpected files like: GBP_USD_S5_20191101_20260303.parquet")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found")
        sys.exit(1)

    print("\n" + "=" * 90)
    print("LOADING S5 CANDLE DATA (5-second bars)")
    print("=" * 90)

    needed_pairs = sorted(set(t['pair'] for t in TRADES))
    pair_s5 = {}
    pair_m5_ind = {}

    for pair in needed_pairs:
        print(f"\n  --- {pair} ---")
        s5_df = load_s5_data(data_dir, pair)
        if s5_df is None:
            continue

        pair_s5[pair] = s5_df

        # Aggregate to M5 and compute indicators
        m5_df = aggregate_s5_to_m5(s5_df)
        m5_ind = build_m5_indicators(m5_df)
        pair_m5_ind[pair] = m5_ind
        print(f"  Aggregated to {len(m5_df):,} M5 bars for indicator calculation")

    if not pair_s5:
        print(f"\nERROR: No candle data loaded.")
        print(f"Expected files in {data_dir}/: " + ", ".join(f"{p}_S5_*.parquet" for p in needed_pairs))
        sys.exit(1)

    # Simulate each trade
    print("\n" + "=" * 90)
    print("SIMULATING EXIT STRATEGIES — 5-SECOND PRECISION")
    print("=" * 90)

    results = []
    for trade in TRADES:
        pair = trade['pair']
        if pair not in pair_s5:
            print(f"\n  Skipping {trade['id']} — no data for {pair}")
            continue

        print(f"\n{'─' * 80}")
        print(f"Trade: {trade['id']} | {pair} {trade['direction'].upper()} @ {trade['entry_price']}")
        print(f"  Entry: {trade['entry_time']}  |  Actual exit: {trade['actual_exit_reason']} at {trade['actual_exit_time']}")
        print(f"  Post-TP re-entry: {'YES — WOULD BE BLOCKED BY COOLDOWN' if trade['was_post_tp_reentry'] else 'No'}")

        result = simulate_trade(trade, pair_s5[pair], pair_m5_ind[pair])

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue

        results.append(result)

        # Print excursion analysis
        print(f"\n  PRICE EXCURSION (5-second precision):")
        print(f"    Max favorable (MFE): +{result['max_favorable_pips']} pips at {result['max_profit_time']}")
        print(f"    Max adverse (MAE):   -{result['max_adverse_pips']} pips")
        green_min = result['seconds_in_green'] // 60
        red_min = result['seconds_in_red'] // 60
        print(f"    Time in profit: {green_min}m | Time in loss: {red_min}m | S5 bars processed: {result['total_bars']:,}")

        # Print profit lock results
        print(f"\n  PROFIT LOCK:")
        if result['lock_levels_hit']:
            for lvl in result['lock_levels_hit']:
                print(f"    >> HIT {lvl['level']} at {lvl['time']} (profit was +{lvl['profit_at_trigger']}p)")
        else:
            print(f"    No lock levels reached (MFE was +{result['max_favorable_pips']}p, first trigger at +60p)")
        if result['lock_exit_reason']:
            print(f"    Lock exit: {result['lock_exit_reason']} at {result['lock_exit_pips']:+.1f}p ({result['lock_exit_time']})")

        # Print momentum fade results
        print(f"\n  MOMENTUM FADE:")
        if result['fade_triggered']:
            print(f"    >> TRIGGERED at {result['fade_exit_time']} — closed at +{result['fade_exit_pips']}p")
            print(f"    RSI at trigger: {result['fade_rsi_at_trigger']}")
        else:
            print(f"    Not triggered (needed +{FADE_MIN_PROFIT}p profit with RSI reversal of {FADE_RSI_REVERSAL}+)")

        # Print comparison table
        print(f"\n  COMPARISON:")
        print(f"    {'Strategy':<25} {'Exit Reason':>15} {'Pips':>8} {'vs Actual':>10}")
        print(f"    {'─' * 60}")
        actual_pips = result['actual_pnl_pips']
        print(f"    {'Actual (no changes)':<25} {result['actual_exit']:>15} {actual_pips:>+8.1f} {'—':>10}")

        if result['lock_exit_pips'] is not None:
            diff = result['lock_exit_pips'] - actual_pips
            marker = ' <<<' if diff > 5 else ''
            print(f"    {'With profit lock':<25} {result['lock_exit_reason']:>15} {result['lock_exit_pips']:>+8.1f} {diff:>+10.1f}{marker}")

        if result['fade_triggered']:
            diff = result['fade_exit_pips'] - actual_pips
            marker = ' <<<' if diff > 5 else ''
            print(f"    {'With momentum fade':<25} {'momentum_fade':>15} {result['fade_exit_pips']:>+8.1f} {diff:>+10.1f}{marker}")

        if result['combined_exit_pips'] is not None:
            diff = result['combined_exit_pips'] - actual_pips
            marker = ' <<<' if diff > 5 else ''
            print(f"    {'Combined (first exit)':<25} {result['combined_exit_reason']:>15} {result['combined_exit_pips']:>+8.1f} {diff:>+10.1f}{marker}")

    # ==================== FINAL SUMMARY ====================
    if results:
        print("\n" + "=" * 90)
        print("OVERALL SUMMARY")
        print("=" * 90)

        actual_total = sum(r['actual_pnl_pips'] for r in results)
        lock_total = sum(r['lock_exit_pips'] for r in results if r['lock_exit_pips'] is not None)
        combined_total = sum(r['combined_exit_pips'] for r in results if r['combined_exit_pips'] is not None)

        lock_improved = sum(1 for r in results if r['lock_exit_pips'] is not None and r['lock_exit_pips'] > r['actual_pnl_pips'])
        lock_worse = sum(1 for r in results if r['lock_exit_pips'] is not None and r['lock_exit_pips'] < r['actual_pnl_pips'])
        lock_same = sum(1 for r in results if r['lock_exit_pips'] is not None and abs(r['lock_exit_pips'] - r['actual_pnl_pips']) < 0.5)
        fade_count = sum(1 for r in results if r['fade_triggered'])

        print(f"\n  {'Metric':<35} {'Actual':>10} {'Profit Lock':>12} {'Combined':>10}")
        print(f"  {'─' * 70}")
        print(f"  {'Total pips':<35} {actual_total:>+10.1f} {lock_total:>+12.1f} {combined_total:>+10.1f}")
        print(f"  {'Improvement over actual':<35} {'—':>10} {lock_total - actual_total:>+12.1f} {combined_total - actual_total:>+10.1f}")
        print(f"  {'Trades improved':<35} {'':>10} {lock_improved:>12} {'':>10}")
        print(f"  {'Trades same outcome':<35} {'':>10} {lock_same:>12} {'':>10}")
        print(f"  {'Trades worse':<35} {'':>10} {lock_worse:>12} {'':>10}")
        print(f"  {'Momentum fade triggers':<35} {'':>10} {'':>12} {fade_count:>10}")

        # Post-TP cooldown impact
        print(f"\n  POST-TP COOLDOWN (60 min, independent of above):")
        cooldown_blocked = [r for r in results if r['was_post_tp_reentry']]
        cooldown_saved_pips = sum(abs(r['actual_pnl_pips']) for r in cooldown_blocked if r['actual_pnl_pips'] < 0)
        for r in cooldown_blocked:
            print(f"    BLOCKED: {r['trade_id']} — would have saved {abs(r['actual_pnl_pips']):.1f} pips (£{abs(r['actual_pnl_gbp']):.0f})")
        print(f"    Total trades blocked: {len(cooldown_blocked)}")
        print(f"    Total pips saved: +{cooldown_saved_pips:.1f}")

        # Grand total
        all_improvement = (combined_total - actual_total) + cooldown_saved_pips
        print(f"\n  {'=' * 60}")
        print(f"  GRAND TOTAL (all strategies combined): {all_improvement:+.1f} pips improvement")
        print(f"  {'=' * 60}")

    # Save results
    output_file = os.path.join(data_dir, 'simulation_results.json')
    clean_results = [{k: v for k, v in r.items() if k != 'bar_trace'} for r in results]
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  Detailed results saved to: {output_file}")

    trace_file = os.path.join(data_dir, 'bar_traces.json')
    traces = {r['trade_id']: r['bar_trace'] for r in results}
    with open(trace_file, 'w') as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"  Per-minute profit traces saved to: {trace_file}")


if __name__ == '__main__':
    main()
