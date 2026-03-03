#!/usr/bin/env python3
"""
Simulate stepped profit lock + momentum fade exit on real tick/candle data.

USAGE:
  1. Download M5 (5-minute) candle data from OANDA for the pairs you traded.
     Export as CSV with columns: time, open, high, low, close, volume
     (or use the OANDA API script below)

  2. Place the CSV files in a folder, named like:
       GBP_USD_M5.csv
       AUD_USD_M5.csv
       EUR_USD_M5.csv
       EUR_GBP_M5.csv

  3. Run:  python3 simulate_exits.py ./data_folder/

  The script will replay every bar for each trade and tell you exactly:
  - Did the trade reach each profit lock level?
  - When did momentum fade trigger (if ever)?
  - What would the P&L have been under each exit strategy?

ALTERNATIVE — OANDA API DOWNLOAD:
  If you have the OANDA API token, the script can download directly.
  Run:  python3 simulate_exits.py --download
"""

import csv
import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


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
    (60, 0),     # At +60p: breakeven
    (80, 30),    # At +80p: lock +30
    (100, 50),   # At +100p: lock +50
    (115, 70),   # At +115p: lock +70
]

# Momentum fade parameters
FADE_MIN_PROFIT = 50    # pips
FADE_RSI_PERIOD = 14
FADE_RSI_REVERSAL = 15  # RSI must reverse by this many points
FADE_MACD_FAST = 12
FADE_MACD_SLOW = 26
FADE_MACD_SIGNAL = 9


# ==================== HELPERS ====================

def pip_mult(pair):
    return 100 if 'JPY' in pair else 10000

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


# ==================== LOAD M5 DATA ====================

def load_m5_data(data_dir: str, pair: str) -> List[Dict]:
    """Load M5 candle data from CSV.

    Expected CSV format (OANDA export or custom):
      time,open,high,low,close,volume
      2026-03-02 00:00:00,1.33450,1.33480,1.33420,1.33460,150

    Also supports:
      datetime,o,h,l,c,volume  (OANDA API format)
      Date,Open,High,Low,Close,Volume  (MT4/MT5 export)
    """
    # Try different filename patterns
    filenames = [
        f"{pair}_M5.csv",
        f"{pair.replace('_', '')}_M5.csv",
        f"{pair}_5min.csv",
        f"{pair}.csv",
    ]

    filepath = None
    for fn in filenames:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            filepath = fp
            break

    if not filepath:
        print(f"  WARNING: No M5 data found for {pair} in {data_dir}")
        print(f"  Tried: {', '.join(filenames)}")
        return []

    candles = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Detect column names
        time_col = next((h for h in headers if h.lower() in ('time', 'datetime', 'date', 'timestamp')), headers[0])
        open_col = next((h for h in headers if h.lower() in ('open', 'o')), None)
        high_col = next((h for h in headers if h.lower() in ('high', 'h')), None)
        low_col = next((h for h in headers if h.lower() in ('low', 'l')), None)
        close_col = next((h for h in headers if h.lower() in ('close', 'c')), None)

        if not all([open_col, high_col, low_col, close_col]):
            print(f"  ERROR: Can't identify OHLC columns in {filepath}")
            print(f"  Headers found: {headers}")
            return []

        for row in reader:
            try:
                ts = row[time_col].strip()
                # Try multiple datetime formats
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y.%m.%d %H:%M',
                            '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M'):
                    try:
                        dt = datetime.strptime(ts, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue  # Skip unparseable rows

                candles.append({
                    'time': dt,
                    'open': float(row[open_col]),
                    'high': float(row[high_col]),
                    'low': float(row[low_col]),
                    'close': float(row[close_col]),
                })
            except (ValueError, KeyError):
                continue

    candles.sort(key=lambda c: c['time'])
    print(f"  Loaded {len(candles)} M5 candles for {pair} ({candles[0]['time']} to {candles[-1]['time']})" if candles else f"  No candles loaded for {pair}")
    return candles


# ==================== SIMULATE ====================

def simulate_trade(trade: Dict, candles: List[Dict]) -> Dict:
    """Simulate a single trade against M5 candle data with all three exit strategies."""

    pair = trade['pair']
    direction = trade['direction']
    entry = trade['entry_price']
    entry_time = datetime.fromisoformat(trade['entry_time'])
    sl = trade['sl_price']
    tp = trade['tp_price']
    mult = pip_mult(pair)
    pip_val = 0.01 if 'JPY' in pair else 0.0001

    # Filter candles to trade's lifetime (and some before for indicators)
    lookback = [c for c in candles if c['time'] < entry_time][-50:]  # 50 bars before for RSI/MACD
    trade_candles = [c for c in candles if c['time'] >= entry_time]

    if not trade_candles:
        return {"error": f"No candle data found for {pair} after {entry_time}"}

    all_candles = lookback + trade_candles
    closes = np.array([c['close'] for c in all_candles])

    # Pre-calculate indicators on full series
    rsi_series = calc_rsi(closes, FADE_RSI_PERIOD)
    macd_hist_series = calc_macd_hist(closes, FADE_MACD_FAST, FADE_MACD_SLOW, FADE_MACD_SIGNAL)

    # Track state
    max_profit_pips = 0.0
    min_profit_pips = 0.0
    current_sl = sl  # Will be ratcheted by profit lock

    # Results for each strategy
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
        "time_in_green_bars": 0,
        "time_in_red_bars": 0,
        "bars_to_max_profit": 0,

        # Per-bar trace (for plotting)
        "bar_trace": [],
    }

    offset = len(lookback)  # Index offset into indicator arrays
    combined_exited = False
    lock_exited = False
    fade_exited = False

    for i, candle in enumerate(trade_candles):
        idx = offset + i  # Index into full indicator arrays

        # Use high/low to check SL/TP hits within the bar
        if direction == 'sell':
            # For shorts: high checks SL, low checks TP
            bar_worst = candle['high']   # Worst price for short
            bar_best = candle['low']     # Best price for short
            bar_exit = candle['close']   # Where we'd actually exit
        else:
            bar_worst = candle['low']
            bar_best = candle['high']
            bar_exit = candle['close']

        profit_at_best = calc_profit_pips(pair, direction, entry, bar_best)
        profit_at_worst = calc_profit_pips(pair, direction, entry, bar_worst)
        profit_at_close = calc_profit_pips(pair, direction, entry, candle['close'])

        # Track max excursion
        if profit_at_best > max_profit_pips:
            max_profit_pips = profit_at_best
            result['bars_to_max_profit'] = i
        if profit_at_worst < min_profit_pips:
            min_profit_pips = profit_at_worst

        if profit_at_close > 0:
            result['time_in_green_bars'] += 1
        else:
            result['time_in_red_bars'] += 1

        # Record bar trace
        result['bar_trace'].append({
            'time': candle['time'].isoformat(),
            'profit_pips': round(profit_at_close, 1),
            'max_profit': round(max_profit_pips, 1),
            'rsi': round(rsi_series[idx], 1) if idx < len(rsi_series) else None,
        })

        # --- CHECK ORIGINAL SL/TP ---
        hit_original_sl = (
            (direction == 'sell' and candle['high'] >= sl) or
            (direction == 'buy' and candle['low'] <= sl)
        )
        hit_original_tp = (
            (direction == 'sell' and candle['low'] <= tp) or
            (direction == 'buy' and candle['high'] >= tp)
        )

        # --- PROFIT LOCK: Check if we should ratchet SL ---
        if not lock_exited:
            for trigger_pips, lock_pips in sorted(PROFIT_LOCK_LEVELS, reverse=True):
                if profit_at_best >= trigger_pips:
                    # Calculate new SL
                    if direction == 'buy':
                        new_sl = entry + (lock_pips * pip_val)
                        sl_improved = new_sl > current_sl
                    else:
                        new_sl = entry - (lock_pips * pip_val)
                        sl_improved = new_sl < current_sl

                    if sl_improved:
                        level_name = f"+{trigger_pips}p→lock+{lock_pips}p"
                        if level_name not in [l['level'] for l in result['lock_levels_hit']]:
                            result['lock_levels_hit'].append({
                                'level': level_name,
                                'time': candle['time'].isoformat(),
                                'profit_at_trigger': round(profit_at_best, 1),
                            })
                        current_sl = new_sl
                    break

            # Check if ratcheted SL is now hit
            hit_ratcheted_sl = (
                (direction == 'sell' and candle['high'] >= current_sl and current_sl != sl) or
                (direction == 'buy' and candle['low'] <= current_sl and current_sl != sl)
            )

            if hit_ratcheted_sl:
                lock_pips_captured = calc_profit_pips(pair, direction, entry, current_sl)
                result['lock_exit_price'] = current_sl
                result['lock_exit_time'] = candle['time'].isoformat()
                result['lock_exit_pips'] = round(lock_pips_captured, 1)
                result['lock_exit_reason'] = 'profit_lock_sl'
                lock_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = current_sl
                    result['combined_exit_time'] = candle['time'].isoformat()
                    result['combined_exit_pips'] = round(lock_pips_captured, 1)
                    result['combined_exit_reason'] = 'profit_lock'
                    combined_exited = True

            elif hit_original_tp and not lock_exited:
                tp_pips = calc_profit_pips(pair, direction, entry, tp)
                result['lock_exit_price'] = tp
                result['lock_exit_time'] = candle['time'].isoformat()
                result['lock_exit_pips'] = round(tp_pips, 1)
                result['lock_exit_reason'] = 'take_profit'
                lock_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = tp
                    result['combined_exit_time'] = candle['time'].isoformat()
                    result['combined_exit_pips'] = round(tp_pips, 1)
                    result['combined_exit_reason'] = 'take_profit'
                    combined_exited = True

            elif hit_original_sl and not lock_exited:
                sl_pips = calc_profit_pips(pair, direction, entry, sl)
                result['lock_exit_price'] = sl
                result['lock_exit_time'] = candle['time'].isoformat()
                result['lock_exit_pips'] = round(sl_pips, 1)
                result['lock_exit_reason'] = 'stop_loss'
                lock_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = sl
                    result['combined_exit_time'] = candle['time'].isoformat()
                    result['combined_exit_pips'] = round(sl_pips, 1)
                    result['combined_exit_reason'] = 'stop_loss'
                    combined_exited = True

        # --- MOMENTUM FADE: Check if momentum is dying while in profit ---
        if not fade_exited and profit_at_close >= FADE_MIN_PROFIT and idx >= FADE_RSI_PERIOD + 5:
            current_rsi = rsi_series[idx]
            rsi_5_bars_ago = rsi_series[idx - 5]
            current_macd_hist = macd_hist_series[idx]
            prev_macd_hist = macd_hist_series[idx - 3]

            fade_triggered = False
            if direction == 'sell':
                # Short: RSI was low and rising (momentum fading), MACD hist rising
                rsi_reversal = current_rsi - rsi_5_bars_ago
                macd_fading = current_macd_hist > prev_macd_hist
                if rsi_reversal > FADE_RSI_REVERSAL and macd_fading:
                    fade_triggered = True
            else:
                # Long: RSI was high and dropping, MACD hist declining
                rsi_reversal = rsi_5_bars_ago - current_rsi
                macd_fading = current_macd_hist < prev_macd_hist
                if rsi_reversal > FADE_RSI_REVERSAL and macd_fading:
                    fade_triggered = True

            if fade_triggered:
                result['fade_triggered'] = True
                result['fade_exit_price'] = candle['close']
                result['fade_exit_time'] = candle['time'].isoformat()
                result['fade_exit_pips'] = round(profit_at_close, 1)
                result['fade_rsi_at_trigger'] = round(current_rsi, 1)
                fade_exited = True

                if not combined_exited:
                    result['combined_exit_price'] = candle['close']
                    result['combined_exit_time'] = candle['time'].isoformat()
                    result['combined_exit_pips'] = round(profit_at_close, 1)
                    result['combined_exit_reason'] = 'momentum_fade'
                    combined_exited = True

        # If both lock and fade have exited (or original SL/TP hit), stop
        if lock_exited and fade_exited:
            break
        if hit_original_sl or hit_original_tp:
            if not lock_exited:
                pips = calc_profit_pips(pair, direction, entry, trade['actual_exit_price'])
                result['lock_exit_price'] = trade['actual_exit_price']
                result['lock_exit_time'] = candle['time'].isoformat()
                result['lock_exit_pips'] = round(pips, 1)
                result['lock_exit_reason'] = trade['actual_exit_reason']
                lock_exited = True
            if not combined_exited:
                pips = calc_profit_pips(pair, direction, entry, trade['actual_exit_price'])
                result['combined_exit_price'] = trade['actual_exit_price']
                result['combined_exit_time'] = candle['time'].isoformat()
                result['combined_exit_pips'] = round(pips, 1)
                result['combined_exit_reason'] = trade['actual_exit_reason']
                combined_exited = True
            break

    result['max_favorable_pips'] = round(max_profit_pips, 1)
    result['max_adverse_pips'] = round(abs(min_profit_pips), 1)

    return result


# ==================== DOWNLOAD FROM OANDA ====================

def download_oanda_data(output_dir: str):
    """Download M5 data from OANDA API for the last 3 days."""
    import requests

    token = os.getenv('OANDA_ACCESS_TOKEN', '')
    account_id = os.getenv('OANDA_ACCOUNT_ID', '')
    env = os.getenv('OANDA_ENV', 'practice')

    if not token:
        print("ERROR: Set OANDA_ACCESS_TOKEN environment variable")
        print("  export OANDA_ACCESS_TOKEN='your-token-here'")
        sys.exit(1)

    base_url = 'https://api-fxpractice.oanda.com' if env == 'practice' else 'https://api-fxtrade.oanda.com'
    headers = {'Authorization': f'Bearer {token}'}

    os.makedirs(output_dir, exist_ok=True)

    pairs = ['GBP_USD', 'AUD_USD', 'EUR_USD', 'EUR_GBP']
    from_time = (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%dT00:00:00Z')

    for pair in pairs:
        print(f"Downloading {pair} M5 data...")
        url = f"{base_url}/v3/instruments/{pair}/candles"
        params = {
            'granularity': 'M5',
            'from': from_time,
            'count': 5000,
            'price': 'M',  # Mid prices
        }
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"  ERROR: {resp.status_code} — {resp.text[:200]}")
            continue

        candles = resp.json().get('candles', [])
        filepath = os.path.join(output_dir, f"{pair}_M5.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume'])
            for c in candles:
                if c.get('complete', True):
                    mid = c['mid']
                    writer.writerow([
                        c['time'][:19].replace('T', ' '),
                        mid['o'], mid['h'], mid['l'], mid['c'],
                        c.get('volume', 0)
                    ])
        print(f"  Saved {len(candles)} candles to {filepath}")

    print(f"\nData saved to {output_dir}/")


# ==================== MAIN ====================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nQuick start:")
        print("  python3 simulate_exits.py --download        # Download from OANDA API")
        print("  python3 simulate_exits.py ./my_data_folder/  # Use local CSV files")
        sys.exit(1)

    if sys.argv[1] == '--download':
        data_dir = './m5_data'
        download_oanda_data(data_dir)
    else:
        data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found")
        sys.exit(1)

    print("\n" + "=" * 90)
    print("LOADING M5 CANDLE DATA")
    print("=" * 90)

    # Load data for all pairs
    pair_data = {}
    needed_pairs = set(t['pair'] for t in TRADES)
    for pair in needed_pairs:
        candles = load_m5_data(data_dir, pair)
        if candles:
            pair_data[pair] = candles

    if not pair_data:
        print("\nERROR: No candle data loaded. Check your CSV files.")
        print(f"Expected files in {data_dir}/: " + ", ".join(f"{p}_M5.csv" for p in needed_pairs))
        sys.exit(1)

    # Simulate each trade
    print("\n" + "=" * 90)
    print("SIMULATING EXIT STRATEGIES ON EACH TRADE")
    print("=" * 90)

    results = []
    for trade in TRADES:
        pair = trade['pair']
        if pair not in pair_data:
            print(f"\n  Skipping {trade['id']} — no data for {pair}")
            continue

        print(f"\n{'─' * 80}")
        print(f"Trade: {trade['id']} | {pair} {trade['direction'].upper()} @ {trade['entry_price']}")
        print(f"  Entry: {trade['entry_time']}  |  Actual exit: {trade['actual_exit_reason']} at {trade['actual_exit_time']}")
        print(f"  Post-TP re-entry: {'YES' if trade['was_post_tp_reentry'] else 'No'}")

        result = simulate_trade(trade, pair_data[pair])

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue

        results.append(result)

        # Print excursion analysis
        print(f"\n  PRICE EXCURSION:")
        print(f"    Max favorable (MFE): +{result['max_favorable_pips']} pips (reached at bar {result['bars_to_max_profit']})")
        print(f"    Max adverse (MAE):   -{result['max_adverse_pips']} pips")
        print(f"    Time in green: {result['time_in_green_bars']} bars | Time in red: {result['time_in_red_bars']} bars")

        # Print profit lock results
        print(f"\n  PROFIT LOCK:")
        if result['lock_levels_hit']:
            for lvl in result['lock_levels_hit']:
                print(f"    Hit {lvl['level']} at {lvl['time']} (profit was +{lvl['profit_at_trigger']}p)")
        else:
            print(f"    No lock levels reached (max profit was +{result['max_favorable_pips']}p, first trigger at +60p)")
        print(f"    Lock exit: {result['lock_exit_reason']} at {result['lock_exit_pips']}p")

        # Print momentum fade results
        print(f"\n  MOMENTUM FADE:")
        if result['fade_triggered']:
            print(f"    TRIGGERED at {result['fade_exit_time']} — closed at +{result['fade_exit_pips']}p")
            print(f"    RSI at trigger: {result['fade_rsi_at_trigger']}")
        else:
            print(f"    Not triggered (needed +{FADE_MIN_PROFIT}p profit with RSI reversal of {FADE_RSI_REVERSAL}+)")

        # Print comparison
        print(f"\n  COMPARISON:")
        print(f"    {'Strategy':<25} {'Exit':>8} {'Pips':>8} {'vs Actual':>10}")
        print(f"    {'─' * 55}")
        actual_pips = result['actual_pnl_pips']
        print(f"    {'Actual (no changes)':<25} {result['actual_exit']:>8} {actual_pips:>+8.1f} {'':>10}")
        if result['lock_exit_pips'] is not None:
            diff = result['lock_exit_pips'] - actual_pips
            print(f"    {'With profit lock':<25} {result['lock_exit_reason']:>8} {result['lock_exit_pips']:>+8.1f} {diff:>+10.1f}")
        if result['fade_triggered']:
            diff = result['fade_exit_pips'] - actual_pips
            print(f"    {'With momentum fade':<25} {'fade':>8} {result['fade_exit_pips']:>+8.1f} {diff:>+10.1f}")
        if result['combined_exit_pips'] is not None:
            diff = result['combined_exit_pips'] - actual_pips
            print(f"    {'Combined (first exit)':<25} {result['combined_exit_reason']:>8} {result['combined_exit_pips']:>+8.1f} {diff:>+10.1f}")

    # Final summary
    if results:
        print("\n" + "=" * 90)
        print("OVERALL SUMMARY")
        print("=" * 90)

        actual_total = sum(r['actual_pnl_pips'] for r in results)
        lock_total = sum(r['lock_exit_pips'] for r in results if r['lock_exit_pips'] is not None)
        combined_total = sum(r['combined_exit_pips'] for r in results if r['combined_exit_pips'] is not None)

        # Count where each strategy improved things
        lock_improved = sum(1 for r in results if r['lock_exit_pips'] is not None and r['lock_exit_pips'] > r['actual_pnl_pips'])
        lock_worse = sum(1 for r in results if r['lock_exit_pips'] is not None and r['lock_exit_pips'] < r['actual_pnl_pips'])
        fade_count = sum(1 for r in results if r['fade_triggered'])

        print(f"\n  {'Metric':<35} {'Actual':>10} {'Lock':>10} {'Combined':>10}")
        print(f"  {'─' * 70}")
        print(f"  {'Total pips':<35} {actual_total:>+10.1f} {lock_total:>+10.1f} {combined_total:>+10.1f}")
        print(f"  {'Improvement over actual':<35} {'':>10} {lock_total - actual_total:>+10.1f} {combined_total - actual_total:>+10.1f}")
        print(f"  {'Trades improved':<35} {'':>10} {lock_improved:>10} {'':>10}")
        print(f"  {'Trades worse':<35} {'':>10} {lock_worse:>10} {'':>10}")
        print(f"  {'Momentum fade triggers':<35} {'':>10} {'':>10} {fade_count:>10}")

        print(f"\n  With post-TP cooldown ALSO applied:")
        cooldown_blocked = [r for r in results if r['was_post_tp_reentry']]
        cooldown_saved_pips = sum(abs(r['actual_pnl_pips']) for r in cooldown_blocked if r['actual_pnl_pips'] < 0)
        print(f"    Trades blocked by cooldown: {len(cooldown_blocked)}")
        print(f"    Additional pips saved: +{cooldown_saved_pips:.1f}")
        print(f"    Grand total improvement: +{combined_total - actual_total + cooldown_saved_pips:.1f} pips")

    # Save detailed results as JSON
    output_file = os.path.join(data_dir, 'simulation_results.json')
    # Remove bar_trace for cleaner output (it's huge)
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items() if k != 'bar_trace'}
        clean_results.append(cr)

    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n  Detailed results saved to: {output_file}")

    # Also save bar traces for plotting
    trace_file = os.path.join(data_dir, 'bar_traces.json')
    traces = {r['trade_id']: r['bar_trace'] for r in results}
    with open(trace_file, 'w') as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"  Bar-by-bar traces saved to: {trace_file}")
    print(f"\n  Use bar_traces.json to plot the profit curve for each trade")


if __name__ == '__main__':
    main()
