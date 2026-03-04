#!/usr/bin/env python3
"""
Walk-Forward Time Window Analysis for 130/60 Forex Bot
======================================================
Finds the optimal 30-minute trading windows using walk-forward validation.

Tests multiple lookback schemes:
  - Expanding: train Y1→test Y2, train Y1-Y2→test Y3, ...
  - Rolling 1yr: train Y1→test Y2, train Y2→test Y3, ...
  - Rolling 2yr: train Y1-Y2→test Y3, train Y2-Y3→test Y4, ...
  - Rolling 3yr: train Y1-Y3→test Y4, train Y2-Y4→test Y5, ...
  - Previous year only: always train on just the prior year

Compares which lookback gives best out-of-sample results.

SPEED OPTIMISED:
  - Vectorised indicator calculation (numpy/pandas)
  - Vectorised signal detection (boolean masks)
  - Numba JIT for S5 trade simulation inner loop
  - Precompute ALL trades once, walk-forward is pure array filtering
  - Parallel pair processing with multiprocessing

USAGE:
  python walk_forward_windows.py /path/to/s5_parquet_folder/

Requirements: pip install pandas pyarrow numpy numba
"""

import os
import sys
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: numba not installed — trade simulation will be slower")
    print("Install with: pip install numba")

# ==================== CONFIGURATION ====================

# 130/60 strategy parameters
TP_PIPS = 130
SL_PIPS = 60

# Indicator parameters (must match bot exactly)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STDDEV = 2
ATR_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
MOMENTUM_LOOKBACK = 10
HURST_WINDOW = 100
HURST_THRESHOLD = 0.52
CONFIDENCE_THRESHOLD = 0.60

# Pairs to test
SYMBOLS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD',
    'USD_CAD', 'EUR_GBP', 'GBP_JPY', 'EUR_JPY',
    'AUD_JPY', 'NZD_USD', 'EUR_AUD', 'GBP_AUD'
]

# 48 windows per day (30-min each)
NUM_WINDOWS = 48

# Walk-forward year boundaries (calendar years)
# Data starts Nov 2019, so first full year is 2020
YEAR_START = 2020
YEAR_END = 2025  # Last full year


# ==================== HELPERS ====================

def pip_mult(pair: str) -> int:
    return 100 if 'JPY' in pair else 10000

def pip_val(pair: str) -> float:
    return 0.01 if 'JPY' in pair else 0.0001

def window_id(dt) -> int:
    """Convert datetime to window ID (0-47)."""
    return dt.hour * 2 + (1 if dt.minute >= 30 else 0)

def window_label(wid: int) -> str:
    """Convert window ID to human-readable label."""
    h = wid // 2
    m = (wid % 2) * 30
    return f"{h:02d}:{m:02d}"


# ==================== DATA LOADING ====================

def find_s5_file(data_dir: str, pair: str) -> Optional[str]:
    """Find S5 parquet file for a pair."""
    import glob as _glob
    for ext in ['parquet', 'feather']:
        for pattern in [
            os.path.join(data_dir, f"{pair}_S5_*.{ext}"),
            os.path.join(data_dir, f"{pair.replace('_', '')}_S5_*.{ext}"),
        ]:
            matches = _glob.glob(pattern)
            if matches:
                return matches[0]
        fp = os.path.join(data_dir, f"{pair}_S5.{ext}")
        if os.path.exists(fp):
            return fp
    return None


def load_s5_full(data_dir: str, pair: str) -> Optional[pd.DataFrame]:
    """Load FULL S5 dataset for a pair. No date filtering — we need all years."""
    filepath = find_s5_file(data_dir, pair)
    if not filepath:
        return None

    print(f"  Loading {os.path.basename(filepath)}...", end=" ", flush=True)
    t0 = _time.time()

    ext = os.path.splitext(filepath)[1].lower()
    cols = ['time', 'open', 'high', 'low', 'close', 'volume']

    if ext == '.parquet':
        df = pd.read_parquet(filepath, columns=cols)
    else:
        df = pd.read_feather(filepath, columns=cols)

    # Normalise time column
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)

    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    elapsed = _time.time() - t0
    print(f"{len(df):,} bars ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}) [{elapsed:.1f}s]")
    return df


def aggregate_s5_to_m5(s5: pd.DataFrame) -> pd.DataFrame:
    """Aggregate S5 bars to M5 using vectorised pandas resample. FAST."""
    df = s5.set_index('time')
    m5 = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])
    m5.reset_index(inplace=True)
    return m5


def aggregate_to_timeframe(m5: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate M5 to higher timeframes for MTF analysis."""
    df = m5.set_index('time')
    agg = df.resample(tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])
    agg.reset_index(inplace=True)
    return agg


# ==================== VECTORISED INDICATORS ====================

def ema_vec(data: np.ndarray, period: int) -> np.ndarray:
    """Vectorised EMA using pandas (faster than manual loop for large arrays)."""
    return pd.Series(data).ewm(span=period, adjust=False).mean().values


def add_indicators_vectorised(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to M5 dataframe. Fully vectorised."""
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    n = len(close)

    # EMAs
    df['ema_9'] = ema_vec(close, EMA_FAST)
    df['ema_21'] = ema_vec(close, EMA_SLOW)

    # RSI (vectorised Wilder smoothing)
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gains).ewm(alpha=1.0/RSI_PERIOD, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(alpha=1.0/RSI_PERIOD, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema_fast = ema_vec(close, MACD_FAST)
    ema_slow = ema_vec(close, MACD_SLOW)
    macd_line = ema_fast - ema_slow
    macd_signal = ema_vec(macd_line, MACD_SIGNAL)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal

    # Bollinger Bands
    sma20 = pd.Series(close).rolling(BB_PERIOD).mean().values
    std20 = pd.Series(close).rolling(BB_PERIOD).std().values
    df['bb_upper'] = sma20 + BB_STDDEV * std20
    df['bb_lower'] = sma20 - BB_STDDEV * std20

    # ATR
    tr_hl = high - low
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr_hc = np.abs(high - prev_close)
    tr_lc = np.abs(low - prev_close)
    tr = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))
    df['atr'] = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).mean().values

    # Momentum
    df['momentum'] = close - np.roll(close, MOMENTUM_LOOKBACK)
    df['momentum'].iloc[:MOMENTUM_LOOKBACK] = 0

    return df


def compute_trend_strength_vec(df: pd.DataFrame) -> np.ndarray:
    """Vectorised trend strength calculation. Returns array of [-1, 1]."""
    n = len(df)
    score = np.zeros(n)

    ema9 = df['ema_9'].values
    ema21 = df['ema_21'].values
    close = df['close'].values
    macd = df['macd'].values
    macd_sig = df['macd_signal'].values
    mom = df['momentum'].values

    score += np.where(ema9 > ema21, 0.3, -0.3)
    sma20 = pd.Series(close).rolling(20).mean().values
    score += np.where(close > sma20, 0.2, -0.2)
    score += np.where(macd > macd_sig, 0.25, -0.25)
    score += np.where(mom > 0, 0.25, -0.25)

    return np.clip(score, -1.0, 1.0)


def compute_mtf_bias_vec(m5: pd.DataFrame) -> np.ndarray:
    """Compute multi-timeframe bias for each M5 bar.

    Uses M5 (primary), M15, H1 EMAs. Pre-aggregates then maps back.
    Simplified but faithful to bot logic.
    """
    n = len(m5)
    bias = np.zeros(n)

    # For each higher timeframe, calculate trend score and map back to M5
    for tf_str, weight in [('5min', 0.20), ('15min', 0.30), ('1h', 0.25), ('4h', 0.20)]:
        if tf_str == '5min':
            tf_df = m5.copy()
        else:
            tf_df = aggregate_to_timeframe(m5, tf_str)

        if len(tf_df) < 25:
            continue

        tf_df = add_ema_only(tf_df)
        # Trend score per bar in this timeframe
        tf_close = tf_df['close'].values
        tf_ema9 = tf_df['ema_9'].values
        tf_ema21 = tf_df['ema_21'].values
        tf_atr = pd.Series(
            tf_df['high'].values - tf_df['low'].values
        ).ewm(span=14, adjust=False).mean().values

        tf_score = np.zeros(len(tf_df))
        tf_score += np.where(tf_ema9 > tf_ema21, 0.5, -0.5)
        tf_score += np.where(tf_close > tf_ema9, 0.3, -0.3)
        # EMA separation normalised by ATR
        ema_sep = np.where(tf_atr > 0, (tf_ema9 - tf_ema21) / tf_atr, 0)
        tf_score += np.clip(ema_sep * 0.1, -0.2, 0.2)
        tf_score = np.clip(tf_score, -1.0, 1.0)

        # Map back to M5 index using forward fill
        tf_times = tf_df['time'].values
        m5_times = m5['time'].values
        # For each M5 bar, find the most recent higher-TF bar
        idx = np.searchsorted(tf_times, m5_times, side='right') - 1
        idx = np.clip(idx, 0, len(tf_score) - 1)
        bias += tf_score[idx] * weight

    return np.clip(bias, -1.0, 1.0)


def add_ema_only(df: pd.DataFrame) -> pd.DataFrame:
    """Add just EMAs to a dataframe (for MTF calculation)."""
    close = df['close'].values.astype(np.float64)
    df['ema_9'] = ema_vec(close, EMA_FAST)
    df['ema_21'] = ema_vec(close, EMA_SLOW)
    return df


def compute_hurst_rolling(h1: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling Hurst exponent on H1 data.

    Returns H1 dataframe with 'hurst' column.
    Uses R/S analysis matching bot's implementation.
    """
    closes = h1['close'].values.astype(np.float64)
    n = len(closes)
    hurst_vals = np.full(n, 0.5)

    window = HURST_WINDOW
    min_w = 10
    max_w = 50
    num_w = 15

    for i in range(window, n):
        chunk = closes[i - window:i]
        returns = np.diff(np.log(chunk))

        if len(returns) < min_w * 2:
            continue

        window_sizes = np.unique(np.logspace(
            np.log10(min_w),
            np.log10(min(max_w, len(returns) // 2)),
            num_w
        ).astype(np.int32))

        if len(window_sizes) < 3:
            continue

        rs_data = []
        for w in window_sizes:
            n_sub = len(returns) // w
            if n_sub == 0:
                continue
            rs_list = []
            for j in range(n_sub):
                seg = returns[j * w:(j + 1) * w]
                mean_seg = np.mean(seg)
                devs = np.cumsum(seg - mean_seg)
                R = np.max(devs) - np.min(devs)
                S = np.std(seg, ddof=1)
                if S > 1e-10:
                    rs_list.append(R / S)
            if rs_list:
                rs_data.append((w, np.mean(rs_list)))

        if len(rs_data) < 3:
            continue

        log_w = np.log(np.array([x[0] for x in rs_data]))
        log_rs = np.log(np.array([x[1] for x in rs_data]))

        nn = len(log_w)
        sx = np.sum(log_w)
        sy = np.sum(log_rs)
        sxy = np.sum(log_w * log_rs)
        sx2 = np.sum(log_w ** 2)
        denom = nn * sx2 - sx ** 2
        if abs(denom) > 1e-10:
            h = (nn * sxy - sx * sy) / denom
            hurst_vals[i] = np.clip(h, 0.0, 1.0)

    h1['hurst'] = hurst_vals
    return h1


def compute_order_flow_vec(m1: pd.DataFrame) -> pd.DataFrame:
    """Vectorised order flow bias on M1 data.

    Simplified but faithful: VWAP position + buying pressure + close position.
    Returns M1 df with 'order_flow' column.
    """
    close = m1['close'].values
    high = m1['high'].values
    low = m1['low'].values
    opn = m1['open'].values
    n = len(close)

    flow = np.zeros(n)
    lookback = 60  # 60 M1 bars = 1 hour

    for i in range(lookback, n):
        chunk_c = close[i - lookback:i + 1]
        chunk_h = high[i - lookback:i + 1]
        chunk_l = low[i - lookback:i + 1]
        chunk_o = opn[i - lookback:i + 1]

        # VWAP position
        hlc3 = (chunk_h + chunk_l + chunk_c) / 3
        vwap = np.mean(hlc3)
        price_range = chunk_h.max() - chunk_l.min()
        if price_range > 0:
            vwap_score = (chunk_c[-1] - vwap) / price_range
        else:
            vwap_score = 0.0

        # Buying pressure
        buying = np.sum(chunk_c > chunk_o) / len(chunk_c) - 0.5

        # Close position in bar
        bar_range = chunk_h[-20:] - chunk_l[-20:]
        valid = bar_range > 0
        if valid.any():
            close_pos = np.mean(
                ((chunk_c[-20:][valid] - chunk_l[-20:][valid]) / bar_range[valid]) - 0.5
            )
        else:
            close_pos = 0.0

        bias = (vwap_score * 0.35 + buying * 0.35 + close_pos * 0.30)
        flow[i] = np.clip(bias * 2.0, -1.0, 1.0)

    m1['order_flow'] = flow
    return m1


# ==================== VECTORISED SIGNAL GENERATION ====================

def generate_all_signals(m5: pd.DataFrame, pair: str,
                         mtf_bias: np.ndarray,
                         trend_strength: np.ndarray,
                         hurst_at_m5: np.ndarray,
                         order_flow_at_m5: np.ndarray) -> pd.DataFrame:
    """Generate ALL signals across the entire M5 dataset at once.

    Returns DataFrame with columns: time, direction, strategy, confidence, entry_price, window_id
    """
    n = len(m5)
    close = m5['close'].values
    high = m5['high'].values
    low = m5['low'].values
    rsi = m5['rsi'].values
    macd = m5['macd'].values
    macd_sig = m5['macd_signal'].values
    bb_upper = m5['bb_upper'].values
    bb_lower = m5['bb_lower'].values
    atr = m5['atr'].values
    momentum = m5['momentum'].values
    ema9 = m5['ema_9'].values
    ema21 = m5['ema_21'].values
    times = m5['time'].values

    pm = pip_mult(pair)
    pv = pip_val(pair)

    # Pre-compute pip-scaled ATR for filtering
    atr_pips = atr * pm

    # Pre-compute shifted values for MACD crossover
    prev_macd = np.roll(macd, 1)
    prev_macd_sig = np.roll(macd_sig, 1)
    prev_macd[0] = macd[0]
    prev_macd_sig[0] = macd_sig[0]

    # Pre-compute momentum in pips (12-bar for pullback strategy)
    mom_12 = np.zeros(n)
    mom_12[12:] = (close[12:] / close[:-12] - 1) * pm

    # MACD crossover signals
    macd_cross_up = (macd > macd_sig) & (prev_macd <= prev_macd_sig)
    macd_cross_down = (macd < macd_sig) & (prev_macd >= prev_macd_sig)

    # Momentum signals
    momentum_up = (momentum > 0) & (rsi > 45)
    momentum_down = (momentum < 0) & (rsi < 55)

    # Trend direction
    bullish = trend_strength > 0.2
    bearish = trend_strength < -0.2

    # ATR filter
    atr_ok = (atr_pips >= 5) & (atr_pips <= 100)

    # Hurst filter
    hurst_ok = hurst_at_m5 >= HURST_THRESHOLD

    # MTF bias
    mtf_pos = mtf_bias > 0.5
    mtf_neg = mtf_bias < -0.5

    # Collect signals
    sig_times = []
    sig_dirs = []
    sig_strats = []
    sig_confs = []
    sig_entries = []

    # Minimum warmup (need 50 bars for indicators to stabilise)
    start = max(50, MACD_SLOW + MACD_SIGNAL + 1)

    for i in range(start, n):
        if not atr_ok[i] or not hurst_ok[i]:
            continue

        best_signal = None
        best_conf = 0.0
        best_strat = None
        best_dir = None

        # ----- STRATEGY 1: TREND FOLLOWING -----
        if abs(trend_strength[i]) >= 0.3 and abs(mtf_bias[i]) >= 0.5:
            if bullish[i] and mtf_pos[i]:
                if macd_cross_up[i] or momentum_up[i]:
                    conf = 0.30
                    if macd_cross_up[i]:
                        conf += 0.20
                    if momentum_up[i]:
                        conf += 0.10
                    of = order_flow_at_m5[i]
                    if of > 0.3:
                        conf += 0.10
                    elif of > 0.1:
                        conf += 0.05
                    conf += mtf_bias[i] * 0.10
                    conf += abs(trend_strength[i]) * 0.10
                    if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_dir = 1  # buy
                        best_strat = 0  # trend_following

            elif bearish[i] and mtf_neg[i]:
                if macd_cross_down[i] or momentum_down[i]:
                    conf = 0.30
                    if macd_cross_down[i]:
                        conf += 0.20
                    if momentum_down[i]:
                        conf += 0.10
                    of = order_flow_at_m5[i]
                    if of < -0.3:
                        conf += 0.10
                    elif of < -0.1:
                        conf += 0.05
                    conf += abs(mtf_bias[i]) * 0.10
                    conf += abs(trend_strength[i]) * 0.10
                    if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_dir = -1  # sell
                        best_strat = 0

        # ----- STRATEGY 2: MOMENTUM PULLBACK -----
        if abs(mtf_bias[i]) >= 0.5 and abs(mom_12[i]) >= 10 and i >= 15:
            if mom_12[i] > 0 and mtf_bias[i] > 0:
                # Pullback: low of last 3 bars < close 4 bars ago
                pullback = min(low[i-2], low[i-1], low[i]) < close[i-4]
                if pullback and rsi[i] < 70 and order_flow_at_m5[i] > -0.3:
                    conf = 0.30
                    if abs(mom_12[i]) > 30:
                        conf += 0.15
                    elif abs(mom_12[i]) > 20:
                        conf += 0.10
                    else:
                        conf += 0.05
                    if 40 <= rsi[i] <= 60:
                        conf += 0.10
                    elif rsi[i] < 40:
                        conf += 0.05
                    if trend_strength[i] > 0.2:
                        conf += 0.10
                    conf += mtf_bias[i] * 0.10
                    if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_dir = 1
                        best_strat = 1  # momentum_pullback

            elif mom_12[i] < 0 and mtf_bias[i] < 0:
                pullback = max(high[i-2], high[i-1], high[i]) > close[i-4]
                if pullback and rsi[i] > 30 and order_flow_at_m5[i] < 0.3:
                    conf = 0.30
                    if abs(mom_12[i]) > 30:
                        conf += 0.15
                    elif abs(mom_12[i]) > 20:
                        conf += 0.10
                    else:
                        conf += 0.05
                    if 40 <= rsi[i] <= 60:
                        conf += 0.10
                    elif rsi[i] > 60:
                        conf += 0.05
                    if trend_strength[i] < -0.2:
                        conf += 0.10
                    conf += abs(mtf_bias[i]) * 0.10
                    if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_dir = -1
                        best_strat = 1

        # ----- STRATEGY 3: VOLATILITY BREAKOUT -----
        if i >= 20:
            recent_atr = np.mean(atr[i-4:i+1])
            longer_atr = np.mean(atr[i-19:i+1])
            if longer_atr > 0 and recent_atr > longer_atr * 1.15:
                atr_exp = recent_atr / longer_atr
                if close[i] > bb_upper[i] and not np.isnan(bb_upper[i]):
                    conf = 0.30 + 0.15  # base + BB breakout
                    if atr_exp > 1.5:
                        conf += 0.10
                    elif atr_exp > 1.25:
                        conf += 0.05
                    if 60 <= rsi[i] <= 75:
                        conf += 0.10
                    elif 55 <= rsi[i] < 60:
                        conf += 0.05
                    if mtf_bias[i] > 0:
                        conf += 0.10
                    if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_dir = 1
                        best_strat = 2  # volatility_breakout

                elif close[i] < bb_lower[i] and not np.isnan(bb_lower[i]):
                    conf = 0.30 + 0.15
                    if atr_exp > 1.5:
                        conf += 0.10
                    elif atr_exp > 1.25:
                        conf += 0.05
                    if 25 <= rsi[i] <= 40:
                        conf += 0.10
                    elif 40 < rsi[i] <= 45:
                        conf += 0.05
                    if mtf_bias[i] < 0:
                        conf += 0.10
                    if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_dir = -1
                        best_strat = 2

        # Record best signal at this bar
        if best_dir is not None:
            sig_times.append(times[i])
            sig_dirs.append(best_dir)
            sig_strats.append(best_strat)
            sig_confs.append(min(best_conf, 0.95))
            sig_entries.append(close[i])

    if not sig_times:
        return pd.DataFrame()

    signals = pd.DataFrame({
        'time': sig_times,
        'direction': sig_dirs,
        'strategy': sig_strats,
        'confidence': sig_confs,
        'entry_price': sig_entries,
    })
    signals['window_id'] = signals['time'].apply(
        lambda t: pd.Timestamp(t).hour * 2 + (1 if pd.Timestamp(t).minute >= 30 else 0)
    )
    signals['year'] = signals['time'].apply(lambda t: pd.Timestamp(t).year)
    return signals


# ==================== TRADE SIMULATION ====================

if HAS_NUMBA:
    @njit(cache=True)
    def _simulate_trade_numba(s5_high, s5_low, start_idx, direction,
                               entry_price, tp_price, sl_price, max_bars=50000):
        """Simulate a single trade on S5 data. Returns pips outcome.

        direction: 1=buy, -1=sell
        Returns: (outcome_pips, bars_used)
        """
        for i in range(start_idx, min(start_idx + max_bars, len(s5_high))):
            if direction == 1:  # BUY
                if s5_high[i] >= tp_price:
                    return 1, i - start_idx  # TP hit
                if s5_low[i] <= sl_price:
                    return -1, i - start_idx  # SL hit
            else:  # SELL
                if s5_low[i] <= tp_price:
                    return 1, i - start_idx  # TP hit
                if s5_high[i] >= sl_price:
                    return -1, i - start_idx  # SL hit
        return 0, max_bars  # Neither hit (still open)
else:
    def _simulate_trade_numba(s5_high, s5_low, start_idx, direction,
                               entry_price, tp_price, sl_price, max_bars=50000):
        for i in range(start_idx, min(start_idx + max_bars, len(s5_high))):
            if direction == 1:
                if s5_high[i] >= tp_price:
                    return 1, i - start_idx
                if s5_low[i] <= sl_price:
                    return -1, i - start_idx
            else:
                if s5_low[i] <= tp_price:
                    return 1, i - start_idx
                if s5_high[i] >= sl_price:
                    return -1, i - start_idx
        return 0, max_bars


def simulate_all_trades(signals: pd.DataFrame, s5: pd.DataFrame,
                        pair: str) -> pd.DataFrame:
    """Simulate signals against S5 data with ONE-TRADE-AT-A-TIME constraint.

    Like the live bot: no new signal on this pair while a trade is open.
    Signals that fire during an open trade are skipped entirely.
    """
    if signals.empty:
        return pd.DataFrame()

    pv = pip_val(pair)
    pm = pip_mult(pair)
    s5_times = s5['time'].values
    s5_high = s5['high'].values.astype(np.float64)
    s5_low = s5['low'].values.astype(np.float64)

    results = []
    sig_times = signals['time'].values
    sig_dirs = signals['direction'].values
    sig_entries = signals['entry_price'].values
    sig_confs = signals['confidence'].values
    sig_strats = signals['strategy'].values
    sig_windows = signals['window_id'].values
    sig_years = signals['year'].values

    # Pre-compute S5 start indices for each signal using searchsorted (FAST)
    s5_start_indices = np.searchsorted(s5_times, sig_times, side='left')

    total = len(signals)
    report_every = max(1, total // 20)
    skipped = 0
    unresolved = 0

    # Track when this pair becomes free (S5 index after trade resolves)
    pair_free_after_idx = 0

    for j in range(total):
        if j % report_every == 0:
            pct = j / total * 100
            print(f"\r    Simulating trades: {pct:.0f}% ({j:,}/{total:,})", end="", flush=True)

        start_idx = s5_start_indices[j]
        if start_idx >= len(s5_high):
            continue

        # ONE-TRADE-AT-A-TIME: skip if pair still has open trade
        if start_idx < pair_free_after_idx:
            skipped += 1
            continue

        entry = sig_entries[j]
        d = sig_dirs[j]

        if d == 1:  # buy
            tp_price = entry + TP_PIPS * pv
            sl_price = entry - SL_PIPS * pv
        else:  # sell
            tp_price = entry - TP_PIPS * pv
            sl_price = entry + SL_PIPS * pv

        # No artificial cap — scan all remaining S5 bars like the real bot
        remaining = len(s5_high) - start_idx
        outcome, bars = _simulate_trade_numba(
            s5_high, s5_low, start_idx, d, entry, tp_price, sl_price,
            max_bars=remaining
        )

        # Block pair until this trade resolves (or end of data if unresolved)
        pair_free_after_idx = start_idx + bars + 1

        if outcome == 1:
            pips = TP_PIPS
        elif outcome == -1:
            pips = -SL_PIPS
        else:
            unresolved += 1
            continue  # Trade still open at end of data — can't score it

        results.append({
            'time': sig_times[j],
            'direction': d,
            'strategy': sig_strats[j],
            'confidence': sig_confs[j],
            'entry_price': entry,
            'pips': pips,
            'outcome': outcome,
            'window_id': sig_windows[j],
            'year': sig_years[j],
            'duration_bars': bars,
        })

    print(f"\r    Simulating trades: 100% ({total:,}/{total:,}) — "
          f"{len(results):,} taken, {skipped:,} skipped (pair busy), "
          f"{unresolved} unresolved (still open at end of data)  ")

    return pd.DataFrame(results) if results else pd.DataFrame()


# ==================== COOLDOWN FILTER ====================

def apply_cooldown(trades: pd.DataFrame, cooldown_bars: int = 720) -> pd.DataFrame:
    """Apply post-TP cooldown: after a TP hit, skip same-pair signals for N S5 bars.

    720 S5 bars = 60 minutes (matching bot's POST_TP_COOLDOWN_SECONDS = 3600).
    This filters the trades DataFrame to remove trades that would have been blocked.
    """
    if trades.empty:
        return trades

    trades = trades.sort_values('time').reset_index(drop=True)
    keep = np.ones(len(trades), dtype=bool)
    last_tp_time = None

    for i in range(len(trades)):
        t = trades.iloc[i]['time']
        if last_tp_time is not None:
            elapsed = (pd.Timestamp(t) - pd.Timestamp(last_tp_time)).total_seconds()
            if elapsed < 3600:  # 60-min cooldown
                keep[i] = False
                continue

        if trades.iloc[i]['outcome'] == 1:  # TP hit
            last_tp_time = t

    return trades[keep].reset_index(drop=True)


# ==================== WALK-FORWARD ENGINE ====================

def rank_windows(trades: pd.DataFrame, top_n: int = 14) -> List[int]:
    """Rank windows by total pips in the training period. Return top N window IDs."""
    if trades.empty:
        return []

    window_pips = trades.groupby('window_id')['pips'].sum()
    # Only include windows with positive expectancy
    profitable = window_pips[window_pips > 0].sort_values(ascending=False)
    return profitable.head(top_n).index.tolist()


def evaluate_windows(trades: pd.DataFrame, allowed_windows: List[int]) -> Dict:
    """Evaluate performance when only trading in allowed windows vs trading all."""
    if trades.empty:
        return {'filtered_pips': 0, 'all_pips': 0, 'filtered_trades': 0,
                'all_trades': 0, 'filtered_wr': 0, 'all_wr': 0, 'improvement': 0}

    all_pips = trades['pips'].sum()
    all_trades = len(trades)
    all_wins = (trades['pips'] > 0).sum()
    all_wr = all_wins / all_trades if all_trades > 0 else 0

    filtered = trades[trades['window_id'].isin(allowed_windows)]
    f_pips = filtered['pips'].sum() if not filtered.empty else 0
    f_trades = len(filtered)
    f_wins = (filtered['pips'] > 0).sum() if not filtered.empty else 0
    f_wr = f_wins / f_trades if f_trades > 0 else 0

    return {
        'filtered_pips': float(f_pips),
        'all_pips': float(all_pips),
        'filtered_trades': f_trades,
        'all_trades': all_trades,
        'filtered_wr': f_wr,
        'all_wr': all_wr,
        'improvement': float(f_pips - all_pips),
        'pips_per_trade_filtered': float(f_pips / f_trades) if f_trades > 0 else 0,
        'pips_per_trade_all': float(all_pips / all_trades) if all_trades > 0 else 0,
    }


def run_walk_forward(all_trades: pd.DataFrame, scheme_name: str,
                     train_test_splits: List[Tuple[List[int], int]],
                     top_n: int = 14) -> Dict:
    """Run walk-forward for a given scheme.

    train_test_splits: list of (train_years, test_year) tuples
    Returns summary dict with OOS results.
    """
    print(f"\n{'='*80}")
    print(f"  SCHEME: {scheme_name}")
    print(f"{'='*80}")

    oos_results = []
    all_selected_windows = defaultdict(int)  # Track window selection frequency

    for train_years, test_year in train_test_splits:
        train_data = all_trades[all_trades['year'].isin(train_years)]
        test_data = all_trades[all_trades['year'] == test_year]

        if train_data.empty or test_data.empty:
            print(f"  Train {train_years} → Test {test_year}: SKIPPED (no data)")
            continue

        # Select best windows from training period
        best_windows = rank_windows(train_data, top_n)

        for w in best_windows:
            all_selected_windows[w] += 1

        # Evaluate on test period
        result = evaluate_windows(test_data, best_windows)
        result['train_years'] = train_years
        result['test_year'] = test_year
        result['selected_windows'] = best_windows
        oos_results.append(result)

        # Print summary
        win_labels = [window_label(w) for w in best_windows[:8]]
        print(f"  Train {train_years} → Test {test_year}:")
        print(f"    Windows selected: {win_labels}{'...' if len(best_windows) > 8 else ''}")
        print(f"    OOS: {result['filtered_pips']:+.0f}p ({result['filtered_trades']} trades, "
              f"WR {result['filtered_wr']:.1%}) vs ALL: {result['all_pips']:+.0f}p "
              f"({result['all_trades']} trades, WR {result['all_wr']:.1%})")
        improvement_pct = (result['filtered_pips'] / result['all_pips'] - 1) * 100 if result['all_pips'] != 0 else 0
        print(f"    Per-trade: {result['pips_per_trade_filtered']:+.1f}p/trade (filtered) vs "
              f"{result['pips_per_trade_all']:+.1f}p/trade (all)")
        if result['filtered_pips'] > result['all_pips']:
            print(f"    >>> IMPROVEMENT: {result['improvement']:+.0f} pips")
        elif result['filtered_pips'] < result['all_pips']:
            print(f"    <<< WORSE: {result['improvement']:+.0f} pips")

    # Aggregate OOS results
    if not oos_results:
        return {'scheme': scheme_name, 'total_oos_pips': 0}

    total_filtered = sum(r['filtered_pips'] for r in oos_results)
    total_all = sum(r['all_pips'] for r in oos_results)
    total_f_trades = sum(r['filtered_trades'] for r in oos_results)
    total_a_trades = sum(r['all_trades'] for r in oos_results)

    # Most consistently selected windows
    consistent_windows = sorted(all_selected_windows.items(), key=lambda x: -x[1])

    print(f"\n  {'─'*60}")
    print(f"  SCHEME SUMMARY: {scheme_name}")
    print(f"  {'─'*60}")
    print(f"  Total OOS pips (filtered): {total_filtered:+.0f}")
    print(f"  Total OOS pips (all):      {total_all:+.0f}")
    print(f"  Improvement:               {total_filtered - total_all:+.0f} pips")
    print(f"  OOS trades taken:          {total_f_trades} / {total_a_trades}")
    print(f"  Most consistent windows:")
    for wid, count in consistent_windows[:10]:
        print(f"    {window_label(wid)} — selected in {count}/{len(oos_results)} folds")

    return {
        'scheme': scheme_name,
        'total_oos_filtered_pips': total_filtered,
        'total_oos_all_pips': total_all,
        'improvement': total_filtered - total_all,
        'oos_trades_filtered': total_f_trades,
        'oos_trades_all': total_a_trades,
        'per_fold': oos_results,
        'consistent_windows': consistent_windows,
    }


# ==================== PROCESS ONE PAIR ====================

def process_pair(data_dir: str, pair: str) -> Optional[pd.DataFrame]:
    """Full pipeline for one pair: load → indicators → signals → simulate."""
    print(f"\n{'─'*70}")
    print(f"  PROCESSING: {pair}")
    print(f"{'─'*70}")

    # 1. Load S5 data
    s5 = load_s5_full(data_dir, pair)
    if s5 is None or len(s5) < 1000:
        print(f"  SKIPPED: insufficient data for {pair}")
        return None

    t0 = _time.time()

    # 2. Aggregate to M5
    print(f"  Aggregating S5 → M5...", end=" ", flush=True)
    m5 = aggregate_s5_to_m5(s5)
    print(f"{len(m5):,} M5 bars [{_time.time()-t0:.1f}s]")

    # 3. Calculate M5 indicators
    t1 = _time.time()
    print(f"  Computing M5 indicators...", end=" ", flush=True)
    m5 = add_indicators_vectorised(m5)
    print(f"[{_time.time()-t1:.1f}s]")

    # 4. Trend strength (vectorised)
    trend_strength = compute_trend_strength_vec(m5)

    # 5. MTF bias
    t2 = _time.time()
    print(f"  Computing MTF bias...", end=" ", flush=True)
    mtf_bias = compute_mtf_bias_vec(m5)
    print(f"[{_time.time()-t2:.1f}s]")

    # 6. Hurst exponent (on H1 data, mapped back to M5)
    t3 = _time.time()
    print(f"  Computing Hurst exponent...", end=" ", flush=True)
    h1 = aggregate_to_timeframe(m5, '1h')
    if len(h1) > HURST_WINDOW + 10:
        h1 = compute_hurst_rolling(h1)
        h1_times = h1['time'].values
        h1_hurst = h1['hurst'].values
        m5_times = m5['time'].values
        idx = np.searchsorted(h1_times, m5_times, side='right') - 1
        idx = np.clip(idx, 0, len(h1_hurst) - 1)
        hurst_at_m5 = h1_hurst[idx]
    else:
        hurst_at_m5 = np.full(len(m5), 0.5)
    print(f"[{_time.time()-t3:.1f}s]")

    # 7. Order flow (on M1, aggregated from S5 — NOT from M5)
    t4 = _time.time()
    print(f"  Computing order flow (S5→M1)...", end=" ", flush=True)
    m1 = aggregate_to_timeframe(s5, '1min')
    if len(m1) > 100:
        m1 = compute_order_flow_vec(m1)
        m1_times = m1['time'].values
        m1_flow = m1['order_flow'].values
        idx = np.searchsorted(m1_times, m5['time'].values, side='right') - 1
        idx = np.clip(idx, 0, len(m1_flow) - 1)
        order_flow_at_m5 = m1_flow[idx]
    else:
        order_flow_at_m5 = np.zeros(len(m5))
    print(f"[{_time.time()-t4:.1f}s]")

    # 8. Generate signals
    t5 = _time.time()
    print(f"  Generating signals...", end=" ", flush=True)
    signals = generate_all_signals(m5, pair, mtf_bias, trend_strength,
                                    hurst_at_m5, order_flow_at_m5)
    if signals.empty:
        print(f"0 signals generated")
        return None
    print(f"{len(signals):,} signals [{_time.time()-t5:.1f}s]")

    # 9. Simulate trades on S5 data
    t6 = _time.time()
    print(f"  Simulating {len(signals):,} trades on S5 data...")
    trades = simulate_all_trades(signals, s5, pair)
    if trades.empty:
        print(f"  No trades resolved")
        return None
    trades['pair'] = pair
    wins = (trades['pips'] > 0).sum()
    losses = (trades['pips'] < 0).sum()
    print(f"  Result: {len(trades):,} trades, {wins} wins / {losses} losses, "
          f"WR={wins/len(trades):.1%}, Total={trades['pips'].sum():+.0f}p [{_time.time()-t6:.1f}s]")

    return trades


# ==================== MAIN ====================

def main():
    if len(sys.argv) < 2:
        print("Usage: python walk_forward_windows.py /path/to/s5_data_folder/")
        print("\nExpects parquet files like: EUR_USD_S5_20191101_20260303.parquet")
        sys.exit(1)

    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} is not a directory")
        sys.exit(1)

    print("=" * 80)
    print("  WALK-FORWARD TIME WINDOW ANALYSIS — 130/60 BOT")
    print("  48 windows × 30 min | S5-precision trade simulation")
    print("=" * 80)

    # Process all pairs
    all_trades_list = []
    t_start = _time.time()

    for pair in SYMBOLS:
        try:
            trades = process_pair(data_dir, pair)
            if trades is not None and not trades.empty:
                all_trades_list.append(trades)
        except Exception as e:
            print(f"  ERROR processing {pair}: {e}")
            import traceback
            traceback.print_exc()

    if not all_trades_list:
        print("\nERROR: No trades generated across any pair. Check your data.")
        sys.exit(1)

    all_trades = pd.concat(all_trades_list, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"  TOTAL: {len(all_trades):,} trades across {len(all_trades_list)} pairs")
    print(f"  Years: {sorted(all_trades['year'].unique())}")
    print(f"  Processing time: {_time.time()-t_start:.1f}s")
    print(f"{'='*80}")

    # Apply cooldown filter
    print(f"\n  Applying 60-min post-TP cooldown per pair...")
    cooldown_trades_list = []
    for pair in all_trades['pair'].unique():
        pair_trades = all_trades[all_trades['pair'] == pair].copy()
        filtered = apply_cooldown(pair_trades)
        cooldown_trades_list.append(filtered)
    all_trades = pd.concat(cooldown_trades_list, ignore_index=True)
    print(f"  After cooldown: {len(all_trades):,} trades")

    # Year summary
    print(f"\n  Per-year breakdown:")
    for year in sorted(all_trades['year'].unique()):
        yt = all_trades[all_trades['year'] == year]
        wins = (yt['pips'] > 0).sum()
        print(f"    {year}: {len(yt):,} trades, WR={wins/len(yt):.1%}, "
              f"Total={yt['pips'].sum():+.0f}p")

    # Window overview (all data)
    print(f"\n  Window performance (all years combined):")
    window_stats = all_trades.groupby('window_id').agg(
        trades=('pips', 'count'),
        total_pips=('pips', 'sum'),
        win_rate=('pips', lambda x: (x > 0).mean()),
    ).sort_values('total_pips', ascending=False)

    print(f"  {'Window':<10} {'Trades':>8} {'Pips':>10} {'WR':>8}")
    print(f"  {'─'*40}")
    for wid, row in window_stats.head(15).iterrows():
        print(f"  {window_label(wid):<10} {row['trades']:>8.0f} {row['total_pips']:>+10.0f} "
              f"{row['win_rate']:>7.1%}")
    print(f"  ... ({len(window_stats)} total windows)")

    # Define years available
    years = sorted(all_trades['year'].unique())
    min_year = max(years[0], YEAR_START)
    max_year = min(years[-1], YEAR_END)
    usable_years = [y for y in years if min_year <= y <= max_year]

    if len(usable_years) < 2:
        print(f"\nERROR: Need at least 2 years of data. Got: {usable_years}")
        sys.exit(1)

    print(f"\n  Usable years for walk-forward: {usable_years}")

    # ==================== DEFINE WALK-FORWARD SCHEMES ====================

    schemes = []

    # 1. EXPANDING WINDOW: train on all prior years
    splits = []
    for i in range(1, len(usable_years)):
        train = usable_years[:i]
        test = usable_years[i]
        splits.append((train, test))
    schemes.append(("Expanding (all prior years)", splits))

    # 2. ROLLING 1-YEAR
    splits = []
    for i in range(1, len(usable_years)):
        train = [usable_years[i-1]]
        test = usable_years[i]
        splits.append((train, test))
    schemes.append(("Rolling 1-year", splits))

    # 3. ROLLING 2-YEAR
    if len(usable_years) >= 3:
        splits = []
        for i in range(2, len(usable_years)):
            train = usable_years[i-2:i]
            test = usable_years[i]
            splits.append((train, test))
        schemes.append(("Rolling 2-year", splits))

    # 4. ROLLING 3-YEAR
    if len(usable_years) >= 4:
        splits = []
        for i in range(3, len(usable_years)):
            train = usable_years[i-3:i]
            test = usable_years[i]
            splits.append((train, test))
        schemes.append(("Rolling 3-year", splits))

    # 5. PREVIOUS YEAR ONLY (same as rolling 1-year but named for clarity)
    # Already covered by rolling 1-year

    # ==================== RUN ALL SCHEMES ====================

    scheme_results = []
    for scheme_name, splits in schemes:
        result = run_walk_forward(all_trades, scheme_name, splits)
        scheme_results.append(result)

    # ==================== FINAL COMPARISON ====================

    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON — WHICH LOOKBACK IS BEST?")
    print(f"{'='*80}")
    print(f"  {'Scheme':<35} {'OOS Filtered':>14} {'OOS All':>10} {'Improvement':>12} {'Trades':>8}")
    print(f"  {'─'*80}")

    best_scheme = None
    best_improvement = float('-inf')

    for r in scheme_results:
        imp = r.get('improvement', 0)
        print(f"  {r['scheme']:<35} {r.get('total_oos_filtered_pips', 0):>+14.0f} "
              f"{r.get('total_oos_all_pips', 0):>+10.0f} {imp:>+12.0f} "
              f"{r.get('oos_trades_filtered', 0):>8}")
        if imp > best_improvement:
            best_improvement = imp
            best_scheme = r

    print(f"\n  BEST SCHEME: {best_scheme['scheme']}")
    print(f"  Improvement: {best_improvement:+.0f} pips out-of-sample")

    # Show the most robust windows (consistently selected across best scheme)
    if best_scheme and 'consistent_windows' in best_scheme:
        print(f"\n  MOST ROBUST WINDOWS (selected across multiple folds):")
        folds = len(best_scheme.get('per_fold', []))
        print(f"  {'Window':<10} {'Selected':>10} {'Consistency':>14}")
        print(f"  {'─'*40}")
        for wid, count in best_scheme['consistent_windows'][:15]:
            pct = count / folds * 100 if folds > 0 else 0
            print(f"  {window_label(wid):<10} {count:>10} {pct:>13.0f}%")

    # Recommend current windows based on best scheme's most recent fold
    if best_scheme and best_scheme.get('per_fold'):
        latest_fold = best_scheme['per_fold'][-1]
        rec_windows = latest_fold.get('selected_windows', [])
        print(f"\n  RECOMMENDED WINDOWS FOR LIVE TRADING:")
        print(f"  (Based on {best_scheme['scheme']}, trained on {latest_fold.get('train_years', '?')})")
        print(f"  Windows: {[window_label(w) for w in rec_windows]}")
        print(f"\n  Config format for bot:")
        print(f"  TRADE_WINDOWS = [")
        for w in rec_windows:
            h = w // 2
            hh = w % 2
            print(f"      ({h}, {hh}),   # {window_label(w)}")
        print(f"  ]")
        print(f"  USE_TIME_WINDOWS = True")

    # Save results
    output_file = os.path.join(data_dir, 'walk_forward_results.json')
    try:
        import json

        # Convert results to JSON-serialisable format
        save_data = {
            'generated': datetime.now().isoformat(),
            'total_trades': len(all_trades),
            'pairs_processed': list(all_trades['pair'].unique()),
            'years': usable_years,
            'scheme_results': [],
        }

        for r in scheme_results:
            scheme_save = {
                'scheme': r['scheme'],
                'total_oos_filtered_pips': r.get('total_oos_filtered_pips', 0),
                'total_oos_all_pips': r.get('total_oos_all_pips', 0),
                'improvement': r.get('improvement', 0),
                'consistent_windows': [
                    {'window': window_label(w), 'window_id': w, 'count': c}
                    for w, c in r.get('consistent_windows', [])
                ],
            }
            save_data['scheme_results'].append(scheme_save)

        # All-data window stats
        save_data['window_stats_all_years'] = {}
        for wid, row in window_stats.iterrows():
            save_data['window_stats_all_years'][window_label(wid)] = {
                'window_id': int(wid),
                'trades': int(row['trades']),
                'total_pips': float(row['total_pips']),
                'win_rate': float(row['win_rate']),
            }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_file}")
    except Exception as e:
        print(f"\n  Warning: Could not save results: {e}")

    total_time = _time.time() - t_start
    print(f"\n  Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
