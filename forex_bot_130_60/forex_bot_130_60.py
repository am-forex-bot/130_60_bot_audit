#!/usr/bin/env python3
"""
Professional Forex Trading Bot for OANDA - 130/60 TP/SL Configuration
TP/SL: 130/60 — Trend Following with Breathing Room
Expected: 70 trades, £66,362 profit, 28.6% win rate (from backtest analysis)
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import requests
from decimal import Decimal, getcontext
import talib
import warnings
import pytz
import csv
from dataclasses import dataclass, asdict
from enum import Enum


warnings.filterwarnings('ignore')
getcontext().prec = 10

# ==================== CONFIGURATION ====================
class Config:
    """Configuration settings for the trading bot"""
    
    # OANDA API Configuration
    OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', 'your_account_id')
    OANDA_ACCESS_TOKEN = os.getenv('OANDA_ACCESS_TOKEN', 'your_access_token')
    OANDA_ENVIRONMENT = os.getenv('OANDA_ENV', 'practice')
    
    # API Endpoints
    if OANDA_ENVIRONMENT == 'practice':
        OANDA_API_URL = 'https://api-fxpractice.oanda.com'
        OANDA_STREAM_URL = 'https://stream-fxpractice.oanda.com'
    else:
        OANDA_API_URL = 'https://api-fxtrade.oanda.com'
        OANDA_STREAM_URL = 'https://stream-fxtrade.oanda.com'
    
    # Trading Parameters - OPTIMIZED
    DAILY_TARGET = 0.005  # 0.5% daily target
    MAX_DAILY_LOSS = 0.02  # 2% max daily loss
    MAX_POSITIONS = 8  # Increased from 5 — more diversification, better margin utilisation
    MAX_DAILY_TRADES = 30  # Prevent overtrading
    
    # Risk Management
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    MIN_RISK_REWARD = 2.0  # Minimum 2:1 R/R
    MAX_SPREAD_PIPS = 2.0  # Dynamic adjustment
    CONFIDENCE_THRESHOLD = 0.60  # High-probability only
    
    # Position Sizing
    USE_KELLY_CRITERION = False  # Disabled — noisy with small samples, can only reduce size
    MAX_KELLY_FRACTION = 0.25
    MAX_CORRELATION_RISK = 0.02
    USE_PARTIAL_PROFITS = False  # DISABLED until fixed
    
    # Trailing Stop — DISABLED (broken and counterproductive for 130/60 strategy)
    USE_TRAILING_STOP = False
    TRAILING_STOP_ACTIVATION = 1.0
    TRAILING_STOP_DISTANCE = 0.5

    # ===== STEPPED PROFIT LOCK =====
    # NOT a trailing stop — discrete ratcheting levels that lock in profit.
    # The trade keeps its original 130-pip TP, but if it reverses after running
    # deep into the green, the stop catches it at a locked-in profit level.
    USE_PROFIT_LOCK = True
    PROFIT_LOCK_LEVELS = [
        # (trigger_pips, lock_pips) — when profit reaches trigger, move SL to lock
        (60, 0),     # At +1R (60p profit): move SL to breakeven
        (80, 30),    # At +80p: lock in 30 pips profit
        (100, 50),   # At +100p: lock in 50 pips profit
        (115, 70),   # At +115p: lock in 70 pips (close to 1R captured)
    ]

    # ===== MOMENTUM-FADE EXIT =====
    # When a trade is significantly in profit but momentum is dying, close early.
    # This captures the "it's +80 pips but it's never getting to 130" scenario.
    USE_MOMENTUM_FADE_EXIT = True
    MOMENTUM_FADE_MIN_PROFIT_PIPS = 50   # Only consider fade exit if +50 pips
    MOMENTUM_FADE_RSI_REVERSAL = 15      # RSI moved 15+ points back from extreme

    # ===== POST-TP COOLDOWN =====
    # After a TP is hit on a pair, don't re-enter the same pair for this many seconds.
    # Prevents "going to the well" after the move has exhausted.
    POST_TP_COOLDOWN_SECONDS = 3600  # 60 minutes
    
    # ===== TP/SL CONFIGURATION (130/60) =====
    STATIC_TP_PIPS = 130  # 130 pip take profit
    STATIC_SL_PIPS = 60   # 60 pip stop loss — gives entries room to breathe
    
    # ===== BEST 14 30-MINUTE WINDOWS FROM ANALYSIS =====
    # Based on time window analysis: £66,362 profit, 70 trades, 28.6% win rate
    # Format: [(hour, half_hour), ...] where half_hour: 0=:00-:29, 1=:30-:59
    TRADE_WINDOWS = [
        (2, 0),   # 02:00-02:29: £9,792 (8 trades, 37.5% WR) 🔥
        (4, 1),   # 04:30-04:59: £7,158 (2 trades, 100% WR!!) 💎
        (2, 1),   # 02:30-02:59: £6,480 (1 trade, 100% WR)
        (0, 0),   # 00:00-00:29: £5,855 (10 trades, 20% WR)
        (4, 0),   # 04:00-04:29: £5,627 (4 trades, 50% WR) 🎯
        (18, 0),  # 18:00-18:29: £5,437 (1 trade, 100% WR)
        (16, 1),  # 16:30-16:59: £4,723 (5 trades, 20% WR)
        (5, 0),   # 05:00-05:29: £4,148 (2 trades, 50% WR)
          # 14:00-14:29: £4,021 (6 trades, 16.7% WR)
        (1, 1),   # 01:30-01:59: £3,543 (5 trades, 40% WR)
        (6, 1),   # 06:30-06:59: £2,860 (3 trades, 33.3% WR)
        (16, 0),  # 16:00-16:29: £2,829 (5 trades, 20% WR)
          # 15:00-15:29: £2,125 (7 trades, 14.3% WR)
           # 01:00-01:29: £1,764 (11 trades, 9.1% WR)
    ]
    
    # Set to True to enable time window filtering
    USE_TIME_WINDOWS = False  # Disabled — re-evaluate after data accumulates with new TP/SL
    
    # Smart Order Routing               
    USE_SMART_ROUTING = True           
    MAX_SPREAD_FOR_MARKET = 1.5        
    LIMIT_ORDER_TIMEOUT = 300          
    SLIPPAGE_BUFFER = 0.5             
    
    # Trading Pairs
    SYMBOLS = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 
        'USD_CAD', 'EUR_GBP', 'GBP_JPY', 'EUR_JPY',
        'AUD_JPY', 'NZD_USD', 'EUR_AUD', 'GBP_AUD'
    ]
    
    # Correlation Groups
    CORRELATION_GROUPS = [
        ['EUR_USD', 'EUR_GBP', 'EUR_JPY', 'EUR_AUD'],
        ['GBP_USD', 'GBP_JPY', 'GBP_AUD'],
        ['AUD_USD', 'AUD_JPY', 'EUR_AUD', 'GBP_AUD'],
        ['USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY']
    ]
    
    # Timeframes
    TIMEFRAMES = {
        'scalp': 'M1',
        'primary': 'M5',
        'trend': 'M15',
        'major': 'H1',
        'daily': 'H4'
    }
    
    # Technical Indicators
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STDDEV = 2
    ATR_PERIOD = 14
    
    # Volatility Requirements
    MIN_ATR_PIPS = 5
    MAX_ATR_PIPS = 100
    
    # Hurst Regime Filter — only trade when market is trending
    # H > 0.5 = trending (persistent), H < 0.5 = mean-reverting
    # 0.52 is a conservative threshold — filters the worst chop
    HURST_THRESHOLD = 0.52
    HURST_WINDOW = 100         # H1 bars for R/S calculation
    HURST_MIN_WINDOW = 10      # Smallest R/S sub-window
    HURST_MAX_WINDOW = 50      # Largest R/S sub-window
    HURST_NUM_WINDOWS = 15     # Log-spaced window count
    
    # Margin Safety — prevent margin calls with concurrent positions
    MAX_MARGIN_USAGE_PCT = 0.60   # Never use more than 60% of balance as margin
    LEVERAGE = 50                  # OANDA practice account leverage
    
    # Trading Sessions (UK time)
    SESSIONS = {
        'tokyo': {'start': 0, 'end': 9, 'quality': 0.6},
        'london_open': {'start': 7, 'end': 9, 'quality': 1.0},
        'london': {'start': 9, 'end': 13, 'quality': 0.9},
        'london_ny_overlap': {'start': 13, 'end': 16, 'quality': 1.0},
        'ny': {'start': 16, 'end': 22, 'quality': 0.8},
        'dead': {'start': 22, 'end': 24, 'quality': 0.3}
    }
    
    # News Filter
    NEWS_BLACKOUT_MINUTES = 30
    
    # Logging
    LOG_DIR = './logs'
    LOG_LEVEL = logging.DEBUG
    ANALYSIS_DIR = './logs/analysis'

    # State persistence file — survives restarts
    STATE_FILE = './logs/bot_state.json'

    # Hard drawdown circuit breaker (uses persisted high-water mark)
    MAX_DRAWDOWN_PCT = 0.15  # Halt new trades if down >15% from persisted peak

    # Account Currency
    ACCOUNT_CURRENCY = 'GBP'

# ==================== PERSISTENT STATE ====================
class PersistentState:
    """Persists critical bot state to disk so it survives restarts."""

    def __init__(self, filepath: str = None):
        self.filepath = filepath or Config.STATE_FILE
        self._state = self._load()

    def _load(self) -> Dict:
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                logging.getLogger('ForexBot').info(
                    f"Loaded persisted state: {data.get('total_trades', 0)} trades, "
                    f"peak={data.get('all_time_peak', 0):.2f}, "
                    f"W/L={data.get('winning_trades', 0)}/{data.get('losing_trades', 0)}"
                )
                return data
        except Exception as e:
            logging.getLogger('ForexBot').warning(f"Failed to load state file: {e}")
        return self._defaults()

    @staticmethod
    def _defaults() -> Dict:
        return {
            'all_time_peak': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_amounts': [],
            'loss_amounts': [],
            'strategy_performance': {},
            'trade_ids_seen': [],  # last 200 trade IDs to avoid double-counting
            'last_updated': None,
        }

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            self._state['last_updated'] = datetime.now().isoformat()
            # Keep trade_ids_seen bounded
            if len(self._state.get('trade_ids_seen', [])) > 200:
                self._state['trade_ids_seen'] = self._state['trade_ids_seen'][-200:]
            # Keep win/loss amount lists bounded
            for key in ('win_amounts', 'loss_amounts'):
                if len(self._state.get(key, [])) > 200:
                    self._state[key] = self._state[key][-200:]
            with open(self.filepath, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            logging.getLogger('ForexBot').error(f"Failed to save state: {e}")

    def get(self, key: str, default=None):
        return self._state.get(key, default)

    def set(self, key: str, value):
        self._state[key] = value

    def update_peak(self, balance: float) -> float:
        """Update all-time peak and return current drawdown from peak."""
        peak = self._state.get('all_time_peak', 0.0)
        if balance > peak:
            self._state['all_time_peak'] = balance
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0.0
        return dd

    def record_trade_result(self, trade_id: str, pnl: float, strategy: str):
        """Record a closed trade result. Idempotent — skips already-seen trade IDs."""
        seen = self._state.get('trade_ids_seen', [])
        if trade_id in seen:
            return  # Already counted
        seen.append(trade_id)
        self._state['trade_ids_seen'] = seen

        self._state['total_trades'] = self._state.get('total_trades', 0) + 1
        self._state['total_pnl'] = self._state.get('total_pnl', 0.0) + pnl

        if pnl > 0:
            self._state['winning_trades'] = self._state.get('winning_trades', 0) + 1
            self._state.setdefault('win_amounts', []).append(pnl)
        else:
            self._state['losing_trades'] = self._state.get('losing_trades', 0) + 1
            self._state.setdefault('loss_amounts', []).append(abs(pnl))

        # Per-strategy tracking
        sp = self._state.setdefault('strategy_performance', {})
        if strategy not in sp:
            sp[strategy] = {'wins': 0, 'losses': 0, 'pnl': 0.0, 'total_trades': 0,
                            'total_win': 0.0, 'total_loss': 0.0}
        sp[strategy]['total_trades'] += 1
        sp[strategy]['pnl'] += pnl
        if pnl > 0:
            sp[strategy]['wins'] += 1
            sp[strategy]['total_win'] += pnl
        else:
            sp[strategy]['losses'] += 1
            sp[strategy]['total_loss'] += abs(pnl)

        self.save()

    def is_drawdown_breached(self, current_balance: float) -> bool:
        """Check if we've breached the hard drawdown limit from persisted peak."""
        dd = self.update_peak(current_balance)
        return dd > Config.MAX_DRAWDOWN_PCT

    @property
    def win_rate(self) -> float:
        total = self._state.get('winning_trades', 0) + self._state.get('losing_trades', 0)
        if total == 0:
            return 0.0
        return self._state.get('winning_trades', 0) / total

    @property
    def expectancy(self) -> float:
        wins = self._state.get('win_amounts', [])
        losses = self._state.get('loss_amounts', [])
        total = len(wins) + len(losses)
        if total == 0:
            return 0.0
        wr = len(wins) / total
        avg_w = sum(wins) / len(wins) if wins else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        return (wr * avg_w) - ((1 - wr) * avg_l)


# ==================== DATA CLASSES ====================
@dataclass
class TradeRecord:
    timestamp: str  # When the record was created (when signal fired/blocked)
    trade_id: str
    symbol: str
    direction: str
    strategy: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    market_conditions: Dict
    indicators: Dict
    status: str = "EXECUTED"  # EXECUTED, BLOCKED, CLOSED
    blocked_reason: Optional[str] = None
    entry_time: Optional[str] = None  # ✅ ADDED - Actual entry time (for real trades) or virtual entry time (for blocked)
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    exit_reason: Optional[str] = None
    max_favorable: Optional[float] = None
    max_adverse: Optional[float] = None
    duration_minutes: Optional[int] = None
    session: Optional[str] = None
    order_flow_bias: Optional[float] = None
    multi_tf_alignment: Optional[bool] = None

@dataclass
class SignalRecord:
    timestamp: str
    symbol: str
    strategy: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    market_conditions: Dict
    indicators: Dict
    traded: bool
    not_traded_reason: Optional[str] = None
    session: Optional[str] = None

@dataclass
class PerformanceSnapshot:
    timestamp: str
    balance: float
    equity: float
    daily_pnl: float
    daily_return_pct: float
    total_pnl: float
    total_return_pct: float
    open_positions: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    kelly_fraction: float
    expectancy: float
    best_session: Optional[str] = None
    worst_session: Optional[str] = None

@dataclass
class PartialTarget:
    price: float
    percentage: float
    units: float
    hit: bool = False

@dataclass
class ExcursionTracker:
    """Track price movement for analysis even after trade closes"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    hit_1r: bool = False
    hit_2r: bool = False
    hit_2_5r: bool = False
    hit_3r: bool = False
    hit_3_5r: bool = False
    hit_4r: bool = False
    hit_4_5r: bool = False
    hit_5r: bool = False
    last_check: Optional[str] = None

@dataclass
class BlockedTrade:
    """Track a blocked trade to simulate what would have happened"""
    blocked_id: str
    timestamp: str
    symbol: str
    direction: str
    strategy: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    blocked_reason: str
    market_conditions: Dict
    indicators: Dict
    session: Optional[str] = None

    # Position sizing (calculated like real trades would be)
    position_size_units: float = 0.0  # Actual units that would have been traded
    risk_amount: float = 0.0  # Risk in account currency
    margin_required: float = 0.0  # Margin that would have been required

    # Tracking fields
    status: str = "ACTIVE"  # ACTIVE, HIT_TP, HIT_SL
    would_have_filled: bool = False
    virtual_entry_price: Optional[float] = None
    virtual_entry_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    virtual_pnl: Optional[float] = None  # In account currency
    virtual_pnl_pips: Optional[float] = None
    duration_minutes: Optional[int] = None
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

# ==================== BLOCKED TRADE TRACKER ====================
class BlockedTradeTracker:
    """
    Tracks trades that were blocked by insufficient margin to analyze
    what performance would have been like if they had executed.
    """

    def __init__(self, oanda_client, risk_manager):
        self.client = oanda_client
        self.risk_manager = risk_manager
        self.active_blocked_trades: Dict[str, BlockedTrade] = {}
        self.completed_blocked_trades: List[BlockedTrade] = []
        self.logger = logging.getLogger('BlockedTradeTracker')

        # Create separate CSV files for blocked trades
        self.blocked_initial_csv = f"{Config.ANALYSIS_DIR}/blocked_trades_initial.csv"
        self.blocked_outcomes_csv = f"{Config.ANALYSIS_DIR}/blocked_trades_outcomes.csv"

        self.logger.info("🚫 BlockedTradeTracker initialized - Will track virtual outcomes of blocked trades")

    def add_blocked_trade(self, signal: Dict, symbol: str, blocked_reason: str, account_balance: float):
        """Add a new blocked trade to track with position sizing"""
        blocked_id = f"BLOCKED_{symbol}_{int(time.time() * 1000)}"

        # Calculate position size using same logic as real trades
        position_size_units = self.risk_manager.calculate_position_size(
            signal, account_balance, symbol, self.client
        )

        # Calculate risk amount in account currency
        entry = signal['entry']
        stop_loss = signal['stop_loss']
        risk_pips = abs(entry - stop_loss) * (100 if 'JPY' in symbol else 10000)

        # Calculate pip value for position size
        base_currency = symbol.split('_')[0]
        quote_currency = symbol.split('_')[1]

        if 'JPY' in symbol:
            pip_value = abs(position_size_units) * 0.01  # For JPY pairs
        else:
            pip_value = abs(position_size_units) * 0.0001

        # Convert to account currency if needed
        if quote_currency != Config.ACCOUNT_CURRENCY:
            conversion_rate = self.client.get_conversion_rate(quote_currency, Config.ACCOUNT_CURRENCY)
            pip_value *= conversion_rate

        risk_amount = risk_pips * pip_value

        # Estimate margin required (typically 3.33% for major pairs, 50:1 leverage)
        # This is an approximation - actual margin depends on leverage and account type
        notional_value = abs(position_size_units) * entry
        margin_required = notional_value * 0.0333  # Assuming 30:1 leverage

        # Convert margin to account currency if needed
        if base_currency != Config.ACCOUNT_CURRENCY:
            margin_conversion_rate = self.client.get_conversion_rate(base_currency, Config.ACCOUNT_CURRENCY)
            margin_required *= margin_conversion_rate

        blocked_trade = BlockedTrade(
            blocked_id=blocked_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            direction=signal['direction'],
            strategy=signal['strategy'],
            confidence=signal['confidence'],
            entry_price=signal['entry'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            blocked_reason=blocked_reason,
            market_conditions=signal.get('market_conditions', {}),
            indicators=signal.get('indicators', {}),
            session=signal.get('session'),
            position_size_units=position_size_units,
            risk_amount=risk_amount,
            margin_required=margin_required
        )

        self.active_blocked_trades[blocked_id] = blocked_trade

        # Log initial blocked trade attempt
        self._log_initial_blocked_trade(blocked_trade)

        self.logger.info(f"🚫 Added blocked trade to tracker: {blocked_id} - {symbol} {signal['direction'].upper()} @ {signal['entry']:.5f}")
        self.logger.info(f"   Units: {position_size_units:.0f} | Risk: {Config.ACCOUNT_CURRENCY}{risk_amount:.2f} | Margin: {Config.ACCOUNT_CURRENCY}{margin_required:.2f}")
        self.logger.info(f"   Will track until TP={signal['take_profit']:.5f} or SL={signal['stop_loss']:.5f}")

        return blocked_id

    def update_blocked_trades(self):
        """Check all active blocked trades and update their status based on current prices"""
        if not self.active_blocked_trades:
            return

        # Get symbols we need to check
        symbols = list(set(bt.symbol for bt in self.active_blocked_trades.values()))

        try:
            prices = self.client.get_prices(symbols)
        except Exception as e:
            self.logger.error(f"Failed to get prices for blocked trade tracking: {e}")
            return

        completed_ids = []

        for blocked_id, blocked_trade in self.active_blocked_trades.items():
            symbol = blocked_trade.symbol

            if symbol not in prices:
                continue

            current_bid = prices[symbol]['bid']
            current_ask = prices[symbol]['ask']
            spread = prices[symbol]['spread']

            # Determine if the virtual trade would have filled
            if not blocked_trade.would_have_filled:
                would_fill = False
                fill_price = None

                if blocked_trade.direction == 'buy':
                    # For buy, we need ask price to reach or go below our entry
                    if current_ask <= blocked_trade.entry_price:
                        would_fill = True
                        fill_price = blocked_trade.entry_price  # Would fill at our limit or better
                else:  # sell
                    # For sell, we need bid price to reach or go above our entry
                    if current_bid >= blocked_trade.entry_price:
                        would_fill = True
                        fill_price = blocked_trade.entry_price

                if would_fill:
                    blocked_trade.would_have_filled = True
                    blocked_trade.virtual_entry_price = fill_price
                    blocked_trade.virtual_entry_time = datetime.now().isoformat()
                    self.logger.info(f"✅ Virtual fill: {blocked_id} would have entered at {fill_price:.5f}")

            # If virtual trade has filled, check for TP/SL
            if blocked_trade.would_have_filled:
                current_price = current_bid if blocked_trade.direction == 'buy' else current_ask

                # Calculate current P&L in pips
                if blocked_trade.direction == 'buy':
                    pips = (current_price - blocked_trade.virtual_entry_price) * (100 if 'JPY' in symbol else 10000)
                else:
                    pips = (blocked_trade.virtual_entry_price - current_price) * (100 if 'JPY' in symbol else 10000)

                # Update max favorable/adverse
                if pips > blocked_trade.max_favorable_pips:
                    blocked_trade.max_favorable_pips = pips
                if pips < 0 and abs(pips) > abs(blocked_trade.max_adverse_pips):
                    blocked_trade.max_adverse_pips = pips

                # Check if TP or SL would be hit
                hit_tp = False
                hit_sl = False

                if blocked_trade.direction == 'buy':
                    # For long: check if bid reached TP or SL
                    if current_bid >= blocked_trade.take_profit:
                        hit_tp = True
                    elif current_bid <= blocked_trade.stop_loss:
                        hit_sl = True
                else:  # sell
                    # For short: check if ask reached TP or SL
                    if current_ask <= blocked_trade.take_profit:
                        hit_tp = True
                    elif current_ask >= blocked_trade.stop_loss:
                        hit_sl = True

                if hit_tp:
                    blocked_trade.status = "HIT_TP"
                    blocked_trade.exit_price = blocked_trade.take_profit
                    blocked_trade.exit_time = datetime.now().isoformat()
                    blocked_trade.exit_reason = "Take Profit"

                    # Calculate P&L in pips
                    if blocked_trade.direction == 'buy':
                        blocked_trade.virtual_pnl_pips = (blocked_trade.take_profit - blocked_trade.virtual_entry_price) * (100 if 'JPY' in symbol else 10000)
                    else:
                        blocked_trade.virtual_pnl_pips = (blocked_trade.virtual_entry_price - blocked_trade.take_profit) * (100 if 'JPY' in symbol else 10000)

                    # Calculate P&L in account currency
                    price_diff = blocked_trade.exit_price - blocked_trade.virtual_entry_price
                    if blocked_trade.direction == 'sell':
                        price_diff = -price_diff

                    pnl_in_quote = abs(blocked_trade.position_size_units) * price_diff

                    # Convert to account currency
                    quote_currency = symbol.split('_')[1]
                    if quote_currency != Config.ACCOUNT_CURRENCY:
                        conversion_rate = self.client.get_conversion_rate(quote_currency, Config.ACCOUNT_CURRENCY)
                        blocked_trade.virtual_pnl = pnl_in_quote * conversion_rate
                    else:
                        blocked_trade.virtual_pnl = pnl_in_quote

                    # Calculate duration
                    entry_time = datetime.fromisoformat(blocked_trade.virtual_entry_time)
                    exit_time = datetime.fromisoformat(blocked_trade.exit_time)
                    blocked_trade.duration_minutes = int((exit_time - entry_time).total_seconds() / 60)

                    self.logger.info(f"🎯 Virtual TP HIT: {blocked_id} - Would have made {blocked_trade.virtual_pnl_pips:.1f} pips ({Config.ACCOUNT_CURRENCY}{blocked_trade.virtual_pnl:.2f}) in {blocked_trade.duration_minutes} mins")
                    completed_ids.append(blocked_id)

                elif hit_sl:
                    blocked_trade.status = "HIT_SL"
                    blocked_trade.exit_price = blocked_trade.stop_loss
                    blocked_trade.exit_time = datetime.now().isoformat()
                    blocked_trade.exit_reason = "Stop Loss"

                    # Calculate P&L in pips (will be negative)
                    if blocked_trade.direction == 'buy':
                        blocked_trade.virtual_pnl_pips = (blocked_trade.stop_loss - blocked_trade.virtual_entry_price) * (100 if 'JPY' in symbol else 10000)
                    else:
                        blocked_trade.virtual_pnl_pips = (blocked_trade.virtual_entry_price - blocked_trade.stop_loss) * (100 if 'JPY' in symbol else 10000)

                    # Calculate P&L in account currency (will be negative)
                    price_diff = blocked_trade.exit_price - blocked_trade.virtual_entry_price
                    if blocked_trade.direction == 'sell':
                        price_diff = -price_diff

                    pnl_in_quote = abs(blocked_trade.position_size_units) * price_diff

                    # Convert to account currency
                    quote_currency = symbol.split('_')[1]
                    if quote_currency != Config.ACCOUNT_CURRENCY:
                        conversion_rate = self.client.get_conversion_rate(quote_currency, Config.ACCOUNT_CURRENCY)
                        blocked_trade.virtual_pnl = pnl_in_quote * conversion_rate
                    else:
                        blocked_trade.virtual_pnl = pnl_in_quote

                    # Calculate duration
                    entry_time = datetime.fromisoformat(blocked_trade.virtual_entry_time)
                    exit_time = datetime.fromisoformat(blocked_trade.exit_time)
                    blocked_trade.duration_minutes = int((exit_time - entry_time).total_seconds() / 60)

                    self.logger.info(f"❌ Virtual SL HIT: {blocked_id} - Would have lost {abs(blocked_trade.virtual_pnl_pips):.1f} pips ({Config.ACCOUNT_CURRENCY}{abs(blocked_trade.virtual_pnl):.2f}) in {blocked_trade.duration_minutes} mins")
                    completed_ids.append(blocked_id)

            # No expiration - track indefinitely until TP/SL hit

        # Move completed trades to completed list and log outcomes
        for blocked_id in completed_ids:
            blocked_trade = self.active_blocked_trades.pop(blocked_id)
            self.completed_blocked_trades.append(blocked_trade)
            self._log_blocked_trade_outcome(blocked_trade)

    def _log_initial_blocked_trade(self, blocked_trade: BlockedTrade):
        """Log the initial blocked trade attempt to CSV"""
        try:
            data = {
                'blocked_id': blocked_trade.blocked_id,
                'timestamp': blocked_trade.timestamp,
                'symbol': blocked_trade.symbol,
                'direction': blocked_trade.direction,
                'strategy': blocked_trade.strategy,
                'confidence': blocked_trade.confidence,
                'entry_price': blocked_trade.entry_price,
                'stop_loss': blocked_trade.stop_loss,
                'take_profit': blocked_trade.take_profit,
                'blocked_reason': blocked_trade.blocked_reason,
                'session': blocked_trade.session,
                'position_size_units': blocked_trade.position_size_units,
                'risk_amount': blocked_trade.risk_amount,
                'margin_required': blocked_trade.margin_required,
                'market_conditions': json.dumps(blocked_trade.market_conditions),
                'indicators': json.dumps(blocked_trade.indicators)
            }

            self._append_to_csv(self.blocked_initial_csv, data)
        except Exception as e:
            self.logger.error(f"Failed to log initial blocked trade: {e}", exc_info=True)

    def _log_blocked_trade_outcome(self, blocked_trade: BlockedTrade):
        """Log the final outcome of a blocked trade to CSV"""
        try:
            data = {
                'blocked_id': blocked_trade.blocked_id,
                'timestamp': blocked_trade.timestamp,
                'symbol': blocked_trade.symbol,
                'direction': blocked_trade.direction,
                'strategy': blocked_trade.strategy,
                'confidence': blocked_trade.confidence,
                'entry_price': blocked_trade.entry_price,
                'stop_loss': blocked_trade.stop_loss,
                'take_profit': blocked_trade.take_profit,
                'blocked_reason': blocked_trade.blocked_reason,
                'session': blocked_trade.session,
                'position_size_units': blocked_trade.position_size_units,
                'risk_amount': blocked_trade.risk_amount,
                'margin_required': blocked_trade.margin_required,
                'would_have_filled': blocked_trade.would_have_filled,
                'virtual_entry_price': blocked_trade.virtual_entry_price or '',
                'virtual_entry_time': blocked_trade.virtual_entry_time or '',
                'status': blocked_trade.status,
                'exit_price': blocked_trade.exit_price or '',
                'exit_time': blocked_trade.exit_time or '',
                'exit_reason': blocked_trade.exit_reason or '',
                'virtual_pnl': blocked_trade.virtual_pnl or '',
                'virtual_pnl_pips': blocked_trade.virtual_pnl_pips or '',
                'duration_minutes': blocked_trade.duration_minutes or '',
                'max_favorable_pips': blocked_trade.max_favorable_pips,
                'max_adverse_pips': blocked_trade.max_adverse_pips,
                'market_conditions': json.dumps(blocked_trade.market_conditions),
                'indicators': json.dumps(blocked_trade.indicators)
            }

            self._append_to_csv(self.blocked_outcomes_csv, data)

            # ✅ ALSO LOG TO UNIFIED CSV - This is the main analysis file
            # Extract order_flow_bias and multi_tf_alignment from market_conditions (same as real trades)
            order_flow = blocked_trade.market_conditions.get('order_flow', 0) if blocked_trade.market_conditions else 0
            multi_tf_bias = blocked_trade.market_conditions.get('multi_tf_bias', 0) if blocked_trade.market_conditions else 0

            trade_record = TradeRecord(
                timestamp=blocked_trade.timestamp,  # When it was blocked
                trade_id=blocked_trade.blocked_id,
                symbol=blocked_trade.symbol,
                direction=blocked_trade.direction,
                strategy=blocked_trade.strategy,
                confidence=blocked_trade.confidence,
                entry_price=blocked_trade.virtual_entry_price,  # Use virtual entry
                stop_loss=blocked_trade.stop_loss,
                take_profit=blocked_trade.take_profit,
                position_size=blocked_trade.position_size_units,
                risk_amount=blocked_trade.risk_amount,
                market_conditions=blocked_trade.market_conditions,
                indicators=blocked_trade.indicators,
                status="BLOCKED_COMPLETE",
                blocked_reason=blocked_trade.blocked_reason,
                entry_time=blocked_trade.virtual_entry_time,  # ✅ FIXED - Use virtual entry time
                exit_time=blocked_trade.exit_time,
                exit_price=blocked_trade.exit_price,
                pnl=blocked_trade.virtual_pnl,
                pnl_pips=blocked_trade.virtual_pnl_pips,
                exit_reason=blocked_trade.exit_reason,
                max_favorable=blocked_trade.max_favorable_pips,
                max_adverse=blocked_trade.max_adverse_pips,
                duration_minutes=blocked_trade.duration_minutes,
                session=blocked_trade.session,
                order_flow_bias=order_flow,  # ✅ FIXED - Extract from market_conditions
                multi_tf_alignment=abs(multi_tf_bias) > 0.5  # ✅ FIXED - Calculate same as real trades
            )

            # Log to UNIFIED CSV using is_blocked=True flag
            advanced_logger.log_complete_trade(trade_record, is_blocked=True)

            # Log summary
            if blocked_trade.status == "HIT_TP":
                self.logger.info(f"📊 BLOCKED TRADE OUTCOME: {blocked_trade.symbol} - WOULD HAVE WON {blocked_trade.virtual_pnl_pips:.1f} pips ({Config.ACCOUNT_CURRENCY}{blocked_trade.virtual_pnl:.2f})")
            elif blocked_trade.status == "HIT_SL":
                self.logger.info(f"📊 BLOCKED TRADE OUTCOME: {blocked_trade.symbol} - WOULD HAVE LOST {abs(blocked_trade.virtual_pnl_pips):.1f} pips ({Config.ACCOUNT_CURRENCY}{abs(blocked_trade.virtual_pnl):.2f})")
            else:
                self.logger.info(f"📊 BLOCKED TRADE OUTCOME: {blocked_trade.symbol} - {blocked_trade.status}")

        except Exception as e:
            self.logger.error(f"Failed to log blocked trade outcome: {e}", exc_info=True)

    def _append_to_csv(self, filename: str, data: Dict):
        """Append data to CSV file"""
        try:
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
                f.flush()
        except Exception as e:
            self.logger.error(f"Failed to write to CSV {filename}: {e}", exc_info=True)

    def get_statistics(self) -> Dict:
        """Get statistics about blocked trades"""
        if not self.completed_blocked_trades:
            return {
                'total_blocked': len(self.active_blocked_trades),
                'completed': 0,
                'active': len(self.active_blocked_trades)
            }

        filled_trades = [bt for bt in self.completed_blocked_trades if bt.would_have_filled]
        tp_hits = [bt for bt in filled_trades if bt.status == "HIT_TP"]
        sl_hits = [bt for bt in filled_trades if bt.status == "HIT_SL"]

        total_pips = sum(bt.virtual_pnl_pips for bt in filled_trades if bt.virtual_pnl_pips)
        win_rate = len(tp_hits) / len(filled_trades) if filled_trades else 0

        return {
            'total_blocked': len(self.completed_blocked_trades) + len(self.active_blocked_trades),
            'completed': len(self.completed_blocked_trades),
            'active': len(self.active_blocked_trades),
            'would_have_filled': len(filled_trades),
            'would_have_won': len(tp_hits),
            'would_have_lost': len(sl_hits),
            'virtual_win_rate': win_rate,
            'total_virtual_pips': total_pips,
            'avg_pips_per_trade': total_pips / len(filled_trades) if filled_trades else 0
        }

# ==================== MARGIN-AWARE PORTFOLIO SIMULATOR ====================
class MarginAwarePortfolioSimulator:
    """
    Advanced simulator that replays all trades (executed and blocked) with full margin awareness.
    Can test different exit strategies and accurately model margin constraints.
    """

    def __init__(self, starting_balance: float, max_leverage: float = 30.0):
        self.starting_balance = starting_balance
        self.max_leverage = max_leverage
        self.margin_rate = 1.0 / max_leverage  # e.g., 30:1 leverage = 0.0333 margin rate
        self.logger = logging.getLogger('MarginSimulator')

    def load_trades_from_csv(self, trades_csv: str, blocked_initial_csv: str, blocked_outcomes_csv: str) -> Dict:
        """Load all trade data from CSVs"""
        executed_trades = []
        blocked_trades = {}

        # Load executed trades
        if os.path.exists(trades_csv):
            try:
                df_executed = pd.read_csv(trades_csv)
                executed_trades = df_executed.to_dict('records')
                self.logger.info(f"Loaded {len(executed_trades)} executed trades")
            except Exception as e:
                self.logger.error(f"Failed to load executed trades: {e}")

        # Load blocked trades initial
        if os.path.exists(blocked_initial_csv):
            try:
                df_blocked = pd.read_csv(blocked_initial_csv)
                for _, row in df_blocked.iterrows():
                    blocked_trades[row['blocked_id']] = row.to_dict()
                self.logger.info(f"Loaded {len(blocked_trades)} blocked trades")
            except Exception as e:
                self.logger.error(f"Failed to load blocked initial: {e}")

        # Merge with blocked outcomes
        if os.path.exists(blocked_outcomes_csv):
            try:
                df_outcomes = pd.read_csv(blocked_outcomes_csv)
                for _, row in df_outcomes.iterrows():
                    bid = row['blocked_id']
                    if bid in blocked_trades:
                        blocked_trades[bid].update(row.to_dict())
                self.logger.info(f"Merged outcomes for {len(df_outcomes)} blocked trades")
            except Exception as e:
                self.logger.error(f"Failed to load blocked outcomes: {e}")

        return {
            'executed': executed_trades,
            'blocked': list(blocked_trades.values())
        }

    def simulate_with_exit_strategy(self, trades_data: Dict, exit_strategy: str = 'tp_sl',
                                    time_limit_hours: Optional[int] = None) -> Dict:
        """
        Simulate portfolio with margin tracking and specified exit strategy.

        Args:
            trades_data: Dict with 'executed' and 'blocked' trades
            exit_strategy: 'tp_sl' (use TP/SL) or 'time_based' (exit after time_limit_hours)
            time_limit_hours: Hours after which to exit (for time_based strategy)

        Returns:
            Comprehensive simulation results
        """
        self.logger.info(f"Starting simulation: strategy={exit_strategy}, time_limit={time_limit_hours}h")

        # Combine all trades and sort chronologically
        all_trades = []

        for trade in trades_data['executed']:
            all_trades.append({
                'type': 'EXECUTED',
                'data': trade,
                'timestamp': trade['timestamp']
            })

        for trade in trades_data['blocked']:
            all_trades.append({
                'type': 'BLOCKED',
                'data': trade,
                'timestamp': trade['timestamp']
            })

        # Sort by timestamp
        all_trades.sort(key=lambda x: x['timestamp'])

        # Simulation state
        balance = self.starting_balance
        available_margin = balance * self.margin_rate * self.max_leverage
        open_positions = {}  # trade_id -> position info
        completed_trades = []
        margin_blocked_count = 0
        would_execute_count = 0

        for trade_event in all_trades:
            trade_type = trade_event['type']
            trade = trade_event['data']

            # Calculate margin required for this trade
            try:
                position_size = float(trade.get('position_size_units', trade.get('position_size', 0)))
                entry_price = float(trade['entry_price'])
                margin_required = float(trade.get('margin_required', 0))

                if margin_required == 0:
                    # Estimate if not provided
                    notional = abs(position_size) * entry_price
                    margin_required = notional * self.margin_rate
            except (ValueError, KeyError, TypeError) as e:
                self.logger.warning(f"Skipping trade due to missing data: {e}")
                continue

            # Check if we have enough margin to open this trade
            if trade_type == 'BLOCKED':
                # This trade was originally blocked, but could it have executed in our simulation?
                if margin_required <= available_margin:
                    would_execute_count += 1
                    # Simulate it as if it executed
                    trade_id = trade['blocked_id']
                    position_size = float(trade['position_size_units'])
                else:
                    # Still would be blocked
                    margin_blocked_count += 1
                    continue
            else:  # EXECUTED
                trade_id = trade.get('trade_id', f"EXEC_{len(completed_trades)}")
                if trade.get('status') == 'BLOCKED':
                    # Skip trades that show as blocked in executed log
                    margin_blocked_count += 1
                    continue

            # Open the position
            open_positions[trade_id] = {
                'entry_time': trade['timestamp'],
                'symbol': trade['symbol'],
                'direction': trade['direction'],
                'entry_price': entry_price,
                'position_size': position_size,
                'margin_required': margin_required,
                'stop_loss': float(trade['stop_loss']),
                'take_profit': float(trade['take_profit']),
                'strategy': trade.get('strategy', 'unknown')
            }

            available_margin -= margin_required

            # Process exits based on strategy
            if trade_type == 'EXECUTED' and trade.get('exit_time'):
                # Trade already has exit data
                exit_price = float(trade.get('exit_price', entry_price))
                pnl = float(trade.get('pnl', 0))

            elif trade_type == 'BLOCKED' and trade.get('would_have_filled'):
                # Use virtual outcome data
                if exit_strategy == 'tp_sl':
                    # Use the TP/SL outcome
                    if trade.get('status') == 'HIT_TP':
                        exit_price = float(trade['take_profit'])
                        pnl = float(trade.get('virtual_pnl', 0))
                    elif trade.get('status') == 'HIT_SL':
                        exit_price = float(trade['stop_loss'])
                        pnl = float(trade.get('virtual_pnl', 0))
                    else:
                        # Still open
                        continue

                elif exit_strategy == 'time_based' and time_limit_hours:
                    # Calculate P&L at time_limit
                    entry_time = pd.to_datetime(trade['timestamp'])
                    # In real implementation, we'd need tick data to know price at time_limit
                    # For now, use the outcome data as approximation
                    pnl = float(trade.get('virtual_pnl', 0))
                    exit_price = float(trade.get('exit_price', entry_price))
            else:
                # Position still open or no exit data
                continue

            # Close position
            if trade_id in open_positions:
                pos = open_positions.pop(trade_id)
                available_margin += pos['margin_required']
                balance += pnl

                completed_trades.append({
                    'trade_id': trade_id,
                    'type': trade_type,
                    'symbol': pos['symbol'],
                    'strategy': pos['strategy'],
                    'entry_time': pos['entry_time'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / self.starting_balance) * 100
                })

        # Calculate final statistics
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]

        total_pnl = sum(t['pnl'] for t in completed_trades)
        final_balance = balance
        total_return_pct = ((final_balance - self.starting_balance) / self.starting_balance) * 100

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 0

        return {
            'simulation_params': {
                'starting_balance': self.starting_balance,
                'exit_strategy': exit_strategy,
                'time_limit_hours': time_limit_hours,
                'max_leverage': self.max_leverage
            },
            'performance': {
                'final_balance': final_balance,
                'total_pnl': total_pnl,
                'total_return_pct': total_return_pct,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            },
            'margin_analysis': {
                'margin_blocked_count': margin_blocked_count,
                'would_execute_count': would_execute_count,
                'still_open_positions': len(open_positions)
            },
            'completed_trades': completed_trades
        }

    def compare_strategies(self, trades_data: Dict) -> Dict:
        """Compare multiple exit strategies"""
        self.logger.info("Running strategy comparison analysis...")

        strategies = [
            ('TP/SL Only', 'tp_sl', None),
            ('24h Exit', 'time_based', 24),
            ('48h Exit', 'time_based', 48),
            ('72h Exit', 'time_based', 72)
        ]

        results = {}
        for name, strategy, time_limit in strategies:
            self.logger.info(f"Simulating: {name}")
            result = self.simulate_with_exit_strategy(trades_data, strategy, time_limit)
            results[name] = result

        # Generate comparison report
        comparison = {
            'strategies': results,
            'best_strategy': None,
            'analysis': {}
        }

        # Find best performing strategy
        best_return = -float('inf')
        best_name = None
        for name, result in results.items():
            if result['performance']['total_return_pct'] > best_return:
                best_return = result['performance']['total_return_pct']
                best_name = name

        comparison['best_strategy'] = best_name

        # Comparative analysis
        tp_sl_result = results['TP/SL Only']['performance']
        for name in ['24h Exit', '48h Exit', '72h Exit']:
            if name in results:
                time_result = results[name]['performance']
                comparison['analysis'][name] = {
                    'pnl_difference': time_result['total_pnl'] - tp_sl_result['total_pnl'],
                    'return_difference_pct': time_result['total_return_pct'] - tp_sl_result['total_return_pct'],
                    'trades_difference': time_result['total_trades'] - tp_sl_result['total_trades'],
                    'win_rate_difference': time_result['win_rate'] - tp_sl_result['win_rate']
                }

        return comparison

    def generate_report(self, comparison_results: Dict, output_file: str):
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 100)
        report.append("MARGIN-AWARE PORTFOLIO SIMULATION - COMPREHENSIVE ANALYSIS")
        report.append("=" * 100)
        report.append("")

        # Strategy comparison
        report.append("STRATEGY COMPARISON:")
        report.append("-" * 100)

        for name, result in comparison_results['strategies'].items():
            perf = result['performance']
            margin = result['margin_analysis']

            report.append(f"\n{name}:")
            report.append(f"  Starting Balance: {Config.ACCOUNT_CURRENCY}{result['simulation_params']['starting_balance']:.2f}")
            report.append(f"  Final Balance:    {Config.ACCOUNT_CURRENCY}{perf['final_balance']:.2f}")
            report.append(f"  Total P&L:        {Config.ACCOUNT_CURRENCY}{perf['total_pnl']:.2f} ({perf['total_return_pct']:.2f}%)")
            report.append(f"  Total Trades:     {perf['total_trades']} ({perf['winning_trades']}W / {perf['losing_trades']}L)")
            report.append(f"  Win Rate:         {perf['win_rate']:.1%}")
            report.append(f"  Avg Win:          {Config.ACCOUNT_CURRENCY}{perf['avg_win']:.2f}")
            report.append(f"  Avg Loss:         {Config.ACCOUNT_CURRENCY}{perf['avg_loss']:.2f}")
            report.append(f"  Profit Factor:    {perf['profit_factor']:.2f}")
            report.append(f"  Margin Blocks:    {margin['margin_blocked_count']}")
            report.append(f"  Would Execute:    {margin['would_execute_count']}")

        # Best strategy
        report.append(f"\n{'=' * 100}")
        report.append(f"BEST STRATEGY: {comparison_results['best_strategy']}")
        report.append(f"{'=' * 100}")

        # Comparative analysis
        if comparison_results['analysis']:
            report.append("\n\nCOMPARATIVE ANALYSIS (vs TP/SL Only):")
            report.append("-" * 100)

            for name, analysis in comparison_results['analysis'].items():
                report.append(f"\n{name}:")
                report.append(f"  P&L Difference:        {Config.ACCOUNT_CURRENCY}{analysis['pnl_difference']:+.2f}")
                report.append(f"  Return Difference:     {analysis['return_difference_pct']:+.2f}%")
                report.append(f"  Trades Difference:     {analysis['trades_difference']:+d}")
                report.append(f"  Win Rate Difference:   {analysis['win_rate_difference']:+.1%}")

        # Write to file
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)

        self.logger.info(f"Report generated: {output_file}")
        return report_text

# ==================== LOGGING SYSTEM ====================
class AdvancedLogger:
    def __init__(self):
        self.setup_directories()
        self.setup_logging()
        self.trades_log = []
        self.signals_log = []
        self.performance_log = []
        self.session_performance = {}
        self.strategy_performance = {}
        
    def setup_directories(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.ANALYSIS_DIR, exist_ok=True)
    
    def setup_logging(self):
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            f"{Config.LOG_DIR}/forex_bot.log",
            maxBytes=10*1024*1024,
            backupCount=30
        )
        console_handler = logging.StreamHandler()
        
        file_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setFormatter(logging.Formatter(log_format))
        
        self.logger = logging.getLogger('ForexBot')
        self.logger.setLevel(Config.LOG_LEVEL)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_trade(self, trade: TradeRecord):
        self.trades_log.append(trade)
        
        try:
            self._append_to_csv(f"{Config.ANALYSIS_DIR}/trades_log.csv", asdict(trade))
            self.logger.debug(f"Trade {trade.trade_id} written to CSV - Status: {trade.status}")
        except Exception as e:
            self.logger.error(f"FAILED to write trade {trade.trade_id} to CSV: {e}", exc_info=True)
        
        if trade.session:
            if trade.session not in self.session_performance:
                self.session_performance[trade.session] = {'wins': 0, 'losses': 0, 'pnl': 0}
            
            if trade.pnl:
                if trade.pnl > 0:
                    self.session_performance[trade.session]['wins'] += 1
                else:
                    self.session_performance[trade.session]['losses'] += 1
                self.session_performance[trade.session]['pnl'] += trade.pnl
        
        if trade.strategy:
            if trade.strategy not in self.strategy_performance:
                self.strategy_performance[trade.strategy] = {
                    'wins': 0, 'losses': 0, 'pnl': 0, 'total_trades': 0,
                    'total_win': 0, 'total_loss': 0
                }
            self.strategy_performance[trade.strategy]['total_trades'] += 1
            if trade.pnl:
                if trade.pnl > 0:
                    self.strategy_performance[trade.strategy]['wins'] += 1
                    self.strategy_performance[trade.strategy]['total_win'] += trade.pnl
                else:
                    self.strategy_performance[trade.strategy]['losses'] += 1
                    self.strategy_performance[trade.strategy]['total_loss'] += abs(trade.pnl)
                self.strategy_performance[trade.strategy]['pnl'] += trade.pnl
    
    def log_complete_trade(self, trade: TradeRecord, is_blocked: bool = False):
        """
        Log ONLY COMPLETE trades to the unified CSV.
        Called when:
        - Real trade closes (from update_closed_trades)
        - Blocked trade hits virtual TP/SL (from BlockedTradeTracker)

        This is the SINGLE SOURCE OF TRUTH for trade analysis.
        """
        # Only log if trade is truly complete
        if not trade.exit_time or not trade.exit_price:
            self.logger.warning(f"Attempted to log incomplete trade {trade.trade_id}")
            return

        try:
            # Calculate additional analysis fields
            entry_dt = datetime.fromisoformat(trade.entry_time or trade.timestamp)
            day_of_week = entry_dt.strftime('%A')
            hour_of_day = entry_dt.hour

            # Prepare complete trade data
            data = {
                'status': 'BLOCKED_IM' if is_blocked else 'TRADED',
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'strategy': trade.strategy,
                'confidence': trade.confidence,
                'entry_time': trade.entry_time or trade.timestamp,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'tp_set': trade.take_profit,
                'sl_set': trade.stop_loss,
                'pnl_gbp': trade.pnl,
                'pnl_pips': trade.pnl_pips,
                'duration_minutes': trade.duration_minutes,
                'position_size': trade.position_size,
                'session': trade.session,
                'day_of_week': day_of_week,
                'hour_of_day': hour_of_day,
                'exit_reason': trade.exit_reason,
                # Market conditions at entry
                'market_regime': trade.market_conditions.get('regime') if trade.market_conditions else None,
                'volatility_at_entry': trade.market_conditions.get('volatility') if trade.market_conditions else None,
                'spread_pips': trade.market_conditions.get('spread_pips') if trade.market_conditions else None,
                'atr_pips': trade.market_conditions.get('atr_pips') if trade.market_conditions else None,
                'order_flow_bias': trade.order_flow_bias,
                'multi_tf_alignment': trade.multi_tf_alignment,
                # MAE/MFE
                'max_favorable_excursion_pips': trade.max_favorable,
                'max_adverse_excursion_pips': trade.max_adverse,
                # Full data as JSON for deep analysis
                'market_conditions_json': json.dumps(trade.market_conditions) if trade.market_conditions else '{}',
                'indicators_json': json.dumps(trade.indicators) if trade.indicators else '{}'
            }

            # Write to UNIFIED CSV
            self._append_to_csv(f"{Config.ANALYSIS_DIR}/trades_complete.csv", data)
            self.logger.info(f"✅ Complete trade logged to trades_complete.csv: {trade.trade_id} - Status: {data['status']}")

        except Exception as e:
            self.logger.error(f"FAILED to log complete trade {trade.trade_id}: {e}", exc_info=True)

    def log_signal(self, signal: SignalRecord):
        self.signals_log.append(signal)
        self._append_to_csv(f"{Config.ANALYSIS_DIR}/signals_log.csv", asdict(signal))

    def update_signal_traded(self, signal_record: 'SignalRecord'):
        """Update the last matching signal in the CSV to mark it as traded.

        The signals CSV previously always showed traded=False because signals were
        logged before execution and never updated afterwards. This fixes that by
        rewriting the last line of the CSV with the updated traded status.
        """
        try:
            csv_path = f"{Config.ANALYSIS_DIR}/signals_log.csv"
            if not os.path.exists(csv_path):
                return

            # Read all lines, update the last one matching this signal
            with open(csv_path, 'r', newline='') as f:
                lines = f.readlines()

            if len(lines) < 2:
                return

            # The signal we just executed should be the last (or near last) row
            # Search from the end for the matching timestamp+symbol+strategy
            key = f"{signal_record.symbol},{signal_record.strategy},{signal_record.direction}"
            for i in range(len(lines) - 1, 0, -1):
                if key in lines[i] and 'False' in lines[i]:
                    lines[i] = lines[i].replace(',False,', ',True,', 1)
                    break

            with open(csv_path, 'w', newline='') as f:
                f.writelines(lines)

        except Exception as e:
            self.logger.warning(f"Failed to update signal traded status: {e}")
    
    def log_performance(self, snapshot: PerformanceSnapshot):
        self.performance_log.append(snapshot)
        self._append_to_csv(f"{Config.ANALYSIS_DIR}/performance_log.csv", asdict(snapshot))
    
    def save_daily_analysis(self):
        analysis = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': self._generate_summary(),
            'trades': [asdict(t) for t in self.trades_log[-100:]],
            'signals': [asdict(s) for s in self.signals_log[-200:]],
            'performance': [asdict(p) for p in self.performance_log[-1440:]],
            'session_performance': self.session_performance,
            'strategy_performance': self.strategy_performance,
            'recommendations': self._generate_recommendations()
        }
        
        filename = f"{Config.ANALYSIS_DIR}/daily_analysis_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Daily analysis saved to {filename}")
        return filename
    
    def _append_to_csv(self, filename: str, data: Dict):
        """Append data to CSV, handling nested dicts properly"""
        try:
            # Convert nested dicts to JSON strings for CSV storage
            csv_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    csv_data[key] = json.dumps(value)
                elif value is None:
                    csv_data[key] = ''
                else:
                    csv_data[key] = value
            
            file_exists = os.path.exists(filename)
            with open(filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_data)
                f.flush()  # Force write to disk
                
        except Exception as e:
            self.logger.error(f"Failed to write to CSV {filename}: {e}", exc_info=True)
    
    def _generate_summary(self) -> Dict:
        if not self.performance_log:
            return {}
        
        latest = self.performance_log[-1]
        daily_trades = [t for t in self.trades_log if t.timestamp > (datetime.now() - timedelta(days=1)).isoformat()]
        
        best_session = None
        if self.session_performance:
            best_session = max(self.session_performance.items(), 
                             key=lambda x: x[1]['pnl'], 
                             default=(None, {}))[0]
        
        best_strategy = None
        if self.strategy_performance:
            best_strategy = max(self.strategy_performance.items(),
                              key=lambda x: x[1]['pnl'],
                              default=(None, {}))[0]
        
        return {
            'current_balance': latest.balance,
            'daily_return': latest.daily_return_pct,
            'total_return': latest.total_return_pct,
            'trades_today': len(daily_trades),
            'win_rate': latest.win_rate,
            'sharpe_ratio': latest.sharpe_ratio,
            'max_drawdown': latest.max_drawdown,
            'profit_factor': latest.profit_factor,
            'expectancy': latest.expectancy,
            'best_session': best_session,
            'best_strategy': best_strategy,
            'session_stats': self.session_performance,
            'strategy_stats': self.strategy_performance
        }
    
    def _generate_recommendations(self) -> List[str]:
        recommendations = []
        
        if self.performance_log:
            latest = self.performance_log[-1]
            
            if latest.win_rate < 0.4:
                recommendations.append("Win rate below 40% - Review entry criteria")
            
            if latest.profit_factor < 1.5:
                recommendations.append("Profit factor below 1.5 - Improve risk/reward ratio")
            
            if latest.max_drawdown > 0.1:
                recommendations.append("Drawdown exceeds 10% - Reduce position sizes")
            
            if latest.expectancy < 0:
                recommendations.append("Negative expectancy - System needs review")
            
            if self.session_performance:
                for session, stats in self.session_performance.items():
                    if stats['losses'] > stats['wins'] * 2:
                        recommendations.append(f"Poor performance in {session} session - Consider avoiding")
            
            if self.strategy_performance:
                for strategy, stats in self.strategy_performance.items():
                    total_trades = stats.get('total_trades', 0)
                    if total_trades >= 20:
                        win_rate = stats['wins'] / total_trades
                        if win_rate < 0.35:
                            recommendations.append(f"Strategy {strategy} underperforming - Consider disabling")
        
        return recommendations

# Initialize logger
advanced_logger = AdvancedLogger()
logger = advanced_logger.logger

# ==================== NEWS FILTER ====================
class NewsFilter:
    def __init__(self):
        self.high_impact_events = []
        self.last_update = None
        self.currency_specific_events = []
        self.high_impact_events = self._fetch_high_impact_events()
        self.last_update = datetime.now()
    
    def _fetch_high_impact_events(self) -> List[datetime]:
        """Fetch high impact news from multiple sources"""
        events = []
        
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"ForexFactory returned {len(data)} events")
                for event in data:
                    if event.get('impact', '').lower() in ['high', 'red']:
                        date_str = event.get('date', '')
                        time_str = event.get('time', '')
                        if date_str and time_str:
                            datetime_str = f"{date_str} {time_str}"
                            try:
                                event_time = datetime.strptime(datetime_str, '%Y-%m-%d %I:%M%p')
                                event_time = event_time.replace(tzinfo=pytz.timezone('US/Eastern'))
                                events.append(event_time)
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Failed to fetch ForexFactory calendar: {e}")
        
        return events

    def is_news_blackout(self, symbol: str = None) -> Tuple[bool, str]:
        """Enhanced news blackout check with reason"""
        current_time = datetime.now()

        logger.debug(f"Checking news blackout at {current_time}")
        
        if self.last_update and (current_time - self.last_update).seconds > 3600:
            self.high_impact_events = self._fetch_high_impact_events()
            self.last_update = current_time
        
        for event_time in self.high_impact_events:
            time_until = (event_time - current_time).total_seconds() / 60
            
            if -15 <= time_until <= 30:
                reason = f"News blackout: High impact event in {time_until:.0f} mins"
                logger.info(reason)
                return True, reason
        
        if symbol and self.currency_specific_events: 
            currencies = symbol.split('_')
            for event_time, event_currency in self.currency_specific_events:
                if any(curr in currencies for curr in event_currency):
                    time_until = (event_time - current_time).total_seconds() / 60
                    if -15 <= time_until <= 30:
                        return True, f"News for {event_currency}"
        
        hour = current_time.hour
        minute = current_time.minute
        weekday = current_time.weekday()
        
        if weekday == 6 and hour == 22 and minute < 30:
            return True, "Sunday market open"
        
        if weekday == 4 and hour >= 21 and minute > 30:
            return True, "Friday close"

        logger.debug(f"About to return False - weekday={weekday}, hour={hour}, minute={minute}")
        
        return False, "No news"

# ==================== TIME WINDOW FILTER ====================
def is_in_trading_window(timestamp: datetime = None) -> Tuple[bool, str]:
    """
    Check if current time is in one of the optimal 30-minute trading windows
    Returns: (allowed, reason)
    """
    if not Config.USE_TIME_WINDOWS:
        return True, "Time filtering disabled"
    
    if timestamp is None:
        timestamp = datetime.now()
    
    hour = timestamp.hour
    half_hour = timestamp.minute // 30  # 0 for :00-:29, 1 for :30-:59
    
    if (hour, half_hour) in Config.TRADE_WINDOWS:
        window_str = f"{hour:02d}:{half_hour*30:02d}"
        return True, f"✅ In trading window: {window_str}"
    else:
        return False, f"⏸️  Outside trading windows (current: {hour:02d}:{half_hour*30:02d})"

# ==================== OANDA CLIENT ====================
class OandaClient:
    def __init__(self):
        self.account_id = Config.OANDA_ACCOUNT_ID
        self.headers = {
            'Authorization': f'Bearer {Config.OANDA_ACCESS_TOKEN}',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_account_info(self) -> Dict:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}"
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()['account']
        else:
            logger.error(f"Failed to get account info: {response.text}")
            return {}
    
    def get_prices(self, instruments: List[str]) -> Dict:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/pricing"
        params = {'instruments': ','.join(instruments)}
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            prices = {}
            data = response.json()
            
            for price in data.get('prices', []):
                if price.get('bids') and price.get('asks'):
                    bid = float(price['bids'][0]['price'])
                    ask = float(price['asks'][0]['price'])
                    spread = ask - bid
                    
                    prices[price['instrument']] = {
                        'bid': bid,
                        'ask': ask,
                        'spread': spread
                    }
                    
                    if 'JPY' in price['instrument']:
                        spread_pips = spread * 100
                    else:
                        spread_pips = spread * 10000
                    logger.debug(f"{price['instrument']}: Bid={bid:.5f}, Ask={ask:.5f}, Spread={spread_pips:.2f} pips")
            
            return prices
        else:
            logger.error(f"Failed to get prices: {response.text}")
            return {}
    
    def get_candles(self, instrument: str, granularity: str, count: int = 200) -> pd.DataFrame:
        url = f"{Config.OANDA_API_URL}/v3/instruments/{instrument}/candles"
        params = {
            'granularity': granularity,
            'count': count,
            'price': 'MBA'
        }
        
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'candles' not in data:
                logger.error(f"No candles in response for {instrument}")
                return pd.DataFrame()
            
            candles = []
            for candle in data['candles']:
                if candle.get('complete', False):
                    high = float(candle['mid']['h'])
                    low = float(candle['mid']['l'])
                    
                    if 'ask' in candle:
                        high = max(high, float(candle['ask']['h']))
                    if 'bid' in candle:
                        low = min(low, float(candle['bid']['l']))
                        
                    candles.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': high,
                        'low': low,
                        'close': float(candle['mid']['c']),
                        'volume': int(candle.get('volume', 0))
                    })
            if not candles:
                logger.warning(f"No complete candles for {instrument}")
                return pd.DataFrame()
            
            df = pd.DataFrame(candles)
            df.set_index('time', inplace=True)
            return df
        else:
            logger.error(f"Failed to get candles for {instrument}: {response.status_code} - {response.text}")
            return pd.DataFrame()
    
    def place_order(self, order_request: Dict) -> Dict:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/orders"

        logger.info(f"Placing order: {order_request}")
        response = self.session.post(url, json={'order': order_request})

        if response.status_code == 201:
            result = response.json()
            logger.info(f"Order placed successfully: {result}")
            return result
        else:
            error_text = response.text
            logger.error(f"Failed to place order: {error_text}")

            # Return error info so caller can handle it
            try:
                error_json = response.json()
                return {'error': error_json, 'error_text': error_text, 'status_code': response.status_code}
            except:
                return {'error': error_text, 'status_code': response.status_code}
    
    def modify_trade(self, trade_id: str, stop_loss: float = None, take_profit: float = None) -> bool:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/trades/{trade_id}/orders"
        
        data = {}
        if stop_loss is not None:
            data['stopLoss'] = {'price': str(round(stop_loss, 5))}
        if take_profit is not None:
            data['takeProfit'] = {'price': str(round(take_profit, 5))}
        
        response = self.session.put(url, json=data)
        
        if response.status_code == 200:
            logger.info(f"Trade {trade_id} modified successfully")
            return True
        else:
            logger.error(f"Failed to modify trade {trade_id}: {response.text}")
            return False
    
    def close_trade(self, trade_id: str, units: str = None) -> bool:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        
        data = {}
        if units:
            data['units'] = units
        
        response = self.session.put(url, json=data)
        
        if response.status_code == 200:
            logger.info(f"Trade {trade_id} closed successfully")
            return True
        else:
            logger.error(f"Failed to close trade {trade_id}: {response.text}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/openPositions"
        response = self.session.get(url)
        
        if response.status_code == 200:
            positions = response.json().get('positions', [])
            return positions
        else:
            url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/positions"
            response = self.session.get(url)
            
            if response.status_code == 200:
                all_positions = response.json()['positions']
                open_positions = []
                
                for pos in all_positions:
                    long_units = float(pos.get('long', {}).get('units', 0))
                    short_units = float(pos.get('short', {}).get('units', 0))
                    
                    if long_units != 0 or short_units != 0:
                        open_positions.append(pos)
                
                return open_positions
            else:
                logger.error(f"Failed to get positions: {response.text}")
                return []
    
    def get_open_trades(self) -> List[Dict]:
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/trades"
        response = self.session.get(url)
        
        if response.status_code == 200:
            trades = response.json()['trades']
            return trades
        else:
            logger.error(f"Failed to get trades: {response.text}")
            return []
    
    def get_conversion_rate(self, from_currency: str, to_currency: str) -> float:
        """Get live conversion rate between two currencies"""
        if from_currency == to_currency:
            return 1.0
        
        pair = f"{from_currency}_{to_currency}"
        reverse_pair = f"{to_currency}_{from_currency}"
        
        prices = self.get_prices([pair, reverse_pair])
        
        if pair in prices:
            return (prices[pair]['bid'] + prices[pair]['ask']) / 2
        elif reverse_pair in prices:
            return 1 / ((prices[reverse_pair]['bid'] + prices[reverse_pair]['ask']) / 2)
        else:
            if from_currency != 'USD' and to_currency != 'USD':
                from_usd = self.get_conversion_rate(from_currency, 'USD')
                usd_to = self.get_conversion_rate('USD', to_currency)
                return from_usd * usd_to
            
            logger.warning(f"Could not get conversion rate for {from_currency} to {to_currency}")
            fallback_rates = {
                ('GBP', 'USD'): 1.36,
                ('USD', 'GBP'): 0.735,
                ('EUR', 'USD'): 1.18,
                ('USD', 'EUR'): 0.847,
                ('USD', 'JPY'): 147.0,
                ('JPY', 'USD'): 0.0068
            }
            return fallback_rates.get((from_currency, to_currency), 1.0)

    def get_current_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Fetch live exchange rate from OANDA"""
        url = f"{Config.OANDA_API_URL}/v3/accounts/{self.account_id}/pricing"
        
        pair = f"{from_currency}_{to_currency}"
        params = {'instruments': pair}
        
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data and len(data['prices']) > 0:
                price_data = data['prices'][0]
                if price_data.get('bids') and price_data.get('asks'):
                    bid = float(price_data['bids'][0]['price'])
                    ask = float(price_data['asks'][0]['price'])
                    return (bid + ask) / 2
        
        reverse_pair = f"{to_currency}_{from_currency}"
        params = {'instruments': reverse_pair}
        response = self.session.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data and len(data['prices']) > 0:
                price_data = data['prices'][0]
                if price_data.get('bids') and price_data.get('asks'):
                    bid = float(price_data['bids'][0]['price'])
                    ask = float(price_data['asks'][0]['price'])
                    mid_price = (bid + ask) / 2
                    return 1 / mid_price
        
        return self.get_conversion_rate(from_currency, to_currency)

# ==================== MARKET ANALYZER ====================
class EnhancedMarketAnalyzer:
    def __init__(self, oanda_client: OandaClient):
        self.client = oanda_client
        self.cache = {}
        self.cache_timeout = 30
        self.news_filter = NewsFilter()
    
    def get_current_session(self) -> Tuple[str, float]:
        """Get current trading session with dynamic timezone"""
        from pytz import timezone
        
        utc_now = datetime.now(timezone('UTC'))
        tokyo_tz = timezone('Asia/Tokyo')
        london_tz = timezone('Europe/London')
        ny_tz = timezone('America/New_York')
        
        tokyo_time = utc_now.astimezone(tokyo_tz)
        london_time = utc_now.astimezone(london_tz)
        ny_time = utc_now.astimezone(ny_tz)
        
        if utc_now.weekday() >= 5:
            return "weekend", 0.0
        
        sessions = []
        
        if 9 <= tokyo_time.hour < 18:
            sessions.append(("tokyo", 0.7))
        
        if 8 <= london_time.hour < 17:
            sessions.append(("london", 0.9))
        
        if 8 <= ny_time.hour < 17:
            sessions.append(("ny", 0.8))
        
        if sessions:
            if "london" in [s[0] for s in sessions] and "ny" in [s[0] for s in sessions]:
                return "london_ny_overlap", 1.0
            
            if "tokyo" in [s[0] for s in sessions] and "london" in [s[0] for s in sessions]:
                return "tokyo_london_overlap", 0.85
            
            return max(sessions, key=lambda x: x[1])
        
        return "dead", 0.3

    def get_market_regime(self) -> str:
        """Detect market regime using multiple currency pairs for a broader read.

        Previous version only checked USD_JPY with extreme thresholds (0.5% move,
        1.5% vol) that almost never triggered — regime was 'neutral' 99.99% of time.

        Now uses USD_JPY (risk proxy) + EUR_USD + AUD_JPY (risk barometer) with
        realistic thresholds calibrated to actual forex volatility.
        """
        try:
            # Check multiple pairs for a composite read
            pairs_data = {}
            for pair in ['USD_JPY', 'EUR_USD', 'AUD_JPY']:
                df = self.client.get_candles(pair, 'H1', count=24)
                if not df.empty and len(df) >= 12:
                    change_pct = (df['close'].iloc[-1] / df['close'].iloc[-12] - 1) * 100
                    volatility = df['close'].pct_change().std() * 100
                    pairs_data[pair] = {'change': change_pct, 'vol': volatility}

            if not pairs_data:
                return "neutral"

            # AUD_JPY is the best single-pair risk barometer
            # Rising = risk-on (carry trade), Falling = risk-off
            audjpy = pairs_data.get('AUD_JPY', {})
            usdjpy = pairs_data.get('USD_JPY', {})
            eurusd = pairs_data.get('EUR_USD', {})

            # Composite risk score: AUD/JPY direction is the primary signal
            risk_score = 0.0
            if audjpy:
                risk_score += audjpy['change'] * 0.5  # AUD/JPY is primary
            if usdjpy:
                # Rising USD/JPY = risk-on (yen weakness), calibrated thresholds
                risk_score += usdjpy['change'] * 0.3
            if eurusd:
                risk_score += eurusd['change'] * 0.2

            # Average volatility across pairs
            avg_vol = np.mean([v['vol'] for v in pairs_data.values()])

            # Calibrated thresholds: 0.15% composite move is meaningful in forex
            if avg_vol > 0.12:
                return "volatile"
            elif risk_score > 0.15:
                return "risk_on"
            elif risk_score < -0.15:
                return "risk_off"
            else:
                return "neutral"

        except Exception as e:
            logger.debug(f"Error detecting market regime: {e}")
            return "neutral"
    
    def get_order_flow_bias(self, symbol: str) -> float:
        """Estimate order flow bias from price action microstructure.

        Previous version had a dead zone: if price was above VWAP but buying_pressure
        was <= 0.6 (or below VWAP with buying_pressure >= 0.4), bias returned 0.0.
        This happened ~80% of the time, making order_flow useless.

        New approach: score multiple independent components and combine them,
        so partial signals still produce a non-zero reading.
        """
        try:
            df = self.client.get_candles(symbol, 'M1', count=60)
            if df.empty or len(df) < 30:
                return 0.0

            current_price = df['close'].iloc[-1]

            # Component 1: VWAP position (continuous, not binary)
            vwap = ((df['high'] + df['low'] + df['close']) / 3).mean()
            price_range = df['high'].max() - df['low'].min()
            if price_range > 0:
                vwap_score = (current_price - vwap) / price_range  # -0.5 to +0.5 range
            else:
                vwap_score = 0.0

            # Component 2: Buying pressure (centered at 0.5, scaled to -0.5 to +0.5)
            buying_bars = len(df[df['close'] > df['open']])
            buying_pressure = (buying_bars / len(df)) - 0.5  # -0.5 to +0.5

            # Component 3: Volume-weighted close position within bar
            # Bars that close near high = buying, close near low = selling
            bar_positions = []
            for _, candle in df.iloc[-20:].iterrows():
                bar_range = candle['high'] - candle['low']
                if bar_range > 0:
                    close_pos = (candle['close'] - candle['low']) / bar_range - 0.5
                    bar_positions.append(close_pos)
            close_position_score = np.mean(bar_positions) if bar_positions else 0.0

            # Component 4: Wick rejections (last 10 bars)
            rejection_score = 0.0
            for i in range(-10, -1):
                candle = df.iloc[i]
                body = abs(candle['close'] - candle['open'])
                upper_wick = candle['high'] - max(candle['close'], candle['open'])
                lower_wick = min(candle['close'], candle['open']) - candle['low']
                if body > 0:
                    if upper_wick > body * 2:
                        rejection_score -= 0.05  # Selling pressure
                    if lower_wick > body * 2:
                        rejection_score += 0.05  # Buying pressure

            # Component 5: Recent momentum (last 5 vs previous 10 bars)
            recent_close = df['close'].iloc[-5:].mean()
            earlier_close = df['close'].iloc[-15:-5].mean()
            if earlier_close > 0:
                momentum_score = (recent_close / earlier_close - 1) * 100  # Scaled pct
                momentum_score = np.clip(momentum_score, -0.3, 0.3)
            else:
                momentum_score = 0.0

            # Weighted combination
            bias = (
                vwap_score * 0.25 +
                buying_pressure * 0.25 +
                close_position_score * 0.20 +
                rejection_score * 0.15 +
                momentum_score * 0.15
            )

            # Scale to [-1, 1] — the components sum to roughly [-0.5, 0.5]
            bias = np.clip(bias * 2.0, -1.0, 1.0)

            return float(bias)

        except Exception as e:
            logger.debug(f"Error calculating order flow for {symbol}: {e}")
            return 0.0
    
    def get_multi_timeframe_bias(self, symbol: str) -> float:
        """Calculate weighted multi-timeframe bias"""
        try:
            timeframe_weights = {
                'scalp': 0.05,
                'primary': 0.20,
                'trend': 0.30,
                'major': 0.25,
                'daily': 0.20
            }
            
            weighted_bias = 0
            total_weight = 0
            
            for tf_name, tf_value in Config.TIMEFRAMES.items():
                if tf_name in timeframe_weights:
                    trend = self._get_timeframe_trend(symbol, tf_value)
                    weight = timeframe_weights[tf_name]
                    weighted_bias += trend * weight
                    total_weight += weight
                    
                    logger.debug(f"{symbol} {tf_name}: trend={trend:.2f}, weight={weight:.2f}")
            
            if total_weight > 0 and total_weight != 1.0:
                weighted_bias = weighted_bias / total_weight
            
            weighted_bias = np.clip(weighted_bias, -1.0, 1.0)
            
            logger.debug(f"{symbol} MTF weighted bias: {weighted_bias:.3f}")
            return weighted_bias
            
        except Exception as e:
            logger.debug(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return 0.0
        
    def _get_timeframe_trend(self, symbol: str, timeframe: str) -> float:
        """Calculate trend direction for a single timeframe.

        Previous version returned only -1, 0, or +1 which made the weighted MTF
        bias collapse to a narrow set of values. Now returns a continuous score
        from -1 to +1 based on EMA relationship and price position.
        """
        df = self.get_market_data(symbol, timeframe, bars=200)
        if df.empty or len(df) < 20:
            return 0.0

        latest = df.iloc[-1]
        score = 0.0

        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            # EMA alignment: +/- 0.5
            if latest['ema_9'] > latest['ema_21']:
                score += 0.5
            else:
                score -= 0.5

            # Price position relative to EMAs: +/- 0.3
            if latest['close'] > latest['ema_9']:
                score += 0.3
            elif latest['close'] < latest['ema_9']:
                score -= 0.3

            # EMA separation magnitude (normalised by ATR): +/- 0.2
            if 'atr' in df.columns and latest['atr'] > 0:
                ema_sep = (latest['ema_9'] - latest['ema_21']) / latest['atr']
                score += np.clip(ema_sep * 0.1, -0.2, 0.2)

        return np.clip(score, -1.0, 1.0)
    
    def compute_hurst_exponent(self, symbol: str) -> float:
        """Compute rolling Hurst exponent from H1 close data using R/S analysis.
        
        H > 0.5 → trending/persistent (good for trend following)
        H < 0.5 → mean-reverting (bad for trend following)
        H ≈ 0.5 → random walk
        
        Uses the same algorithm as the backtested regime exit system.
        """
        try:
            df = self.get_market_data(symbol, 'H1', bars=Config.HURST_WINDOW + 50)
            if df.empty or len(df) < Config.HURST_WINDOW:
                logger.debug(f"{symbol}: Insufficient H1 data for Hurst ({len(df) if not df.empty else 0} bars)")
                return 0.5  # Assume random walk if no data
            
            closes = df['close'].values[-Config.HURST_WINDOW:]
            returns = np.diff(np.log(closes))
            
            if len(returns) < Config.HURST_MIN_WINDOW * 2:
                return 0.5
            
            # Log-spaced window sizes for R/S analysis
            window_sizes = np.unique(np.logspace(
                np.log10(Config.HURST_MIN_WINDOW),
                np.log10(min(Config.HURST_MAX_WINDOW, len(returns) // 2)),
                Config.HURST_NUM_WINDOWS
            ).astype(int))
            
            if len(window_sizes) < 3:
                return 0.5
            
            rs_values = []
            for w in window_sizes:
                rs_list = []
                n_windows = len(returns) // w
                for i in range(n_windows):
                    chunk = returns[i * w:(i + 1) * w]
                    mean_chunk = np.mean(chunk)
                    deviations = np.cumsum(chunk - mean_chunk)
                    R = np.max(deviations) - np.min(deviations)
                    S = np.std(chunk, ddof=1)
                    if S > 1e-10:
                        rs_list.append(R / S)
                if rs_list:
                    rs_values.append((w, np.mean(rs_list)))
            
            if len(rs_values) < 3:
                return 0.5
            
            log_sizes = np.log(np.array([x[0] for x in rs_values]))
            log_rs = np.log(np.array([x[1] for x in rs_values]))
            
            # Linear regression: slope = Hurst exponent
            n = len(log_sizes)
            sum_x = np.sum(log_sizes)
            sum_y = np.sum(log_rs)
            sum_xy = np.sum(log_sizes * log_rs)
            sum_x2 = np.sum(log_sizes ** 2)
            
            denom = n * sum_x2 - sum_x ** 2
            if abs(denom) < 1e-10:
                return 0.5
            
            hurst = (n * sum_xy - sum_x * sum_y) / denom
            hurst = float(np.clip(hurst, 0.0, 1.0))
            
            logger.debug(f"{symbol}: Hurst exponent = {hurst:.3f}")
            return hurst
            
        except Exception as e:
            logger.error(f"Error computing Hurst for {symbol}: {e}")
            return 0.5

    def find_support_resistance_levels(self, symbol: str) -> List[float]:
        df = self.get_market_data(symbol, Config.TIMEFRAMES['major'], bars=100)
        if df.empty or len(df) < 50:
            return []
        
        levels = []
        
        for i in range(10, len(df) - 10):
            if df['high'].iloc[i] == max(df['high'].iloc[i-10:i+10]):
                levels.append(df['high'].iloc[i])
            
            if df['low'].iloc[i] == min(df['low'].iloc[i-10:i+10]):
                levels.append(df['low'].iloc[i])
        
        current = df['close'].iloc[-1]
        if 'JPY' in symbol:
            step = 0.5
        else:
            step = 0.0050
        
        for i in range(-5, 6):
            level = round(current / step) * step + (i * step)
            levels.append(level)
        
        levels = sorted(list(set(levels)))
        
        return levels
    
    def is_spread_acceptable(self, symbol: str, atr: float, session_quality: float) -> bool:
        prices = self.client.get_prices([symbol])
        if symbol not in prices:
            return False
        
        spread = prices[symbol]['spread']
        
        max_spread_ratio = 0.25 * (2 - session_quality)
        
        if 'JPY' in symbol:
            absolute_max = 3.0 / 100
        else:
            absolute_max = 0.0003
        
        max_allowed = min(atr * max_spread_ratio, absolute_max)
        
        return spread <= max_allowed
    
    def rank_pairs_by_opportunity(self) -> List[str]:
        rankings = []
        
        for symbol in Config.SYMBOLS:
            try:
                condition = self.get_market_condition(symbol)
                if not condition['tradeable']:
                    logger.debug(f"{symbol} not tradeable: {condition.get('reason', 'unknown')}")
                    continue
                
                score = 0
                
                atr_pips = condition['volatility_pips']
                if 10 <= atr_pips <= 50:
                    score += 3
                elif 5 <= atr_pips <= 60:
                    score += 1
                
                trend = abs(condition['trend_strength'])
                score += trend * 5
                
                spread_atr_ratio = condition['spread_pips'] / atr_pips if atr_pips > 0 else 1
                if spread_atr_ratio < 0.05:
                    score += 3
                elif spread_atr_ratio < 0.1:
                    score += 1
                
                mtf_bias = abs(self.get_multi_timeframe_bias(symbol))
                score += mtf_bias * 4
                
                if self.is_optimal_session_for_pair(symbol):
                    score += 3
                
                order_flow = abs(condition['order_flow'])
                score += order_flow * 2
                
                rankings.append((symbol, score))
                
            except Exception as e:
                logger.debug(f"Error ranking {symbol}: {e}")
                continue
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        all_tradeable = [s for s, score in rankings]
        
        if rankings:
            logger.info(f"Tradeable pairs ({len(all_tradeable)}): {[(s, round(score, 2)) for s, score in rankings[:5]]}")
        
        return all_tradeable
    
    def is_optimal_session_for_pair(self, symbol: str) -> bool:
        session, quality = self.get_current_session()
        
        optimal_sessions = {
            'EUR_USD': ['london', 'london_ny_overlap'],
            'GBP_USD': ['london', 'london_ny_overlap'],
            'USD_JPY': ['tokyo', 'ny'],
            'AUD_USD': ['tokyo', 'ny'],
            'EUR_GBP': ['london'],
            'GBP_JPY': ['tokyo', 'london'],
        }
        
        base_pair = symbol.split('_')[0] + '_USD'
        if base_pair not in optimal_sessions:
            base_pair = 'USD_' + symbol.split('_')[1]
        
        optimal = optimal_sessions.get(symbol, optimal_sessions.get(base_pair, []))
        
        return session in optimal
    
    def get_market_data(self, symbol: str, timeframe: str, bars: int = 200) -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}_{bars}_v2"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        df = self.client.get_candles(symbol, timeframe, bars)
        
        if df.empty:
            logger.warning(f"Empty dataframe for {symbol} {timeframe}")
            return pd.DataFrame()
        
        if len(df) >= 45:
            df = self._add_indicators(df)
            self.cache[cache_key] = (df, time.time())
        else:
            logger.warning(f"Insufficient data for indicators: {symbol} has {len(df)} candles")
        
        return df
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            
            close_prices = df['close'].values.astype(float)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            volume = df['volume'].values.astype(float)
            
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['ema_9'] = talib.EMA(close_prices, timeperiod=9)
            df['ema_21'] = talib.EMA(close_prices, timeperiod=21)
            
            df['rsi'] = talib.RSI(close_prices, timeperiod=Config.RSI_PERIOD)
            
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices,
                fastperiod=Config.MACD_FAST,
                slowperiod=Config.MACD_SLOW,
                signalperiod=Config.MACD_SIGNAL
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=Config.BB_PERIOD,
                nbdevup=Config.BB_STDDEV,
                nbdevdn=Config.BB_STDDEV
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=Config.ATR_PERIOD)
            
            stoch_k, stoch_d = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            df['momentum'] = df['close'] - df['close'].shift(10)
            
            df['roc'] = talib.ROC(close_prices, timeperiod=10)
            
            if volume.sum() > 0:
                df['volume_sma'] = talib.SMA(volume, timeperiod=20)
                df['volume_ratio'] = np.where(df['volume_sma'] > 0, df['volume'] / df['volume_sma'], 1.0)
            else:
                df['volume_ratio'] = 1.0

            raw_atr = df['atr'].iloc[-1]
            logger.debug(f"Raw ATR value: {raw_atr:.8f}")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            
        return df
    
    def get_market_condition(self, symbol: str) -> Dict:
        blackout, reason = self.news_filter.is_news_blackout(symbol)
        if blackout:
            logger.debug(f"{symbol}: News blackout active - {reason}")
        
        df_primary = self.get_market_data(symbol, Config.TIMEFRAMES['primary'])
        
        if df_primary.empty or len(df_primary) < 50:
            logger.debug(f"{symbol}: Insufficient data for analysis")
            return {'tradeable': False, 'reason': 'insufficient_data'}
        
        prices = self.client.get_prices([symbol])
        if symbol not in prices:
            logger.debug(f"{symbol}: No price data available")
            return {'tradeable': False, 'reason': 'no_price_data'}
        
        latest = df_primary.iloc[-1]
        
        atr = latest['atr']
        if 'JPY' in symbol:
            atr_pips = atr * 100
        else:
            atr_pips = atr * 10000
        
        volatility_ok = Config.MIN_ATR_PIPS <= atr_pips <= Config.MAX_ATR_PIPS
        
        session, session_quality = self.get_current_session()
        
        spread_ok = self.is_spread_acceptable(symbol, atr, session_quality)
        
        order_flow = self.get_order_flow_bias(symbol)
        
        mtf_bias = self.get_multi_timeframe_bias(symbol)
        
        trend_strength = self._calculate_trend_strength(df_primary)
        
        sr_levels = self.find_support_resistance_levels(symbol)
        
        indicators_snapshot = {}
        for key in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'momentum', 'stoch_k', 'stoch_d']:
            if key in df_primary.columns:
                val = latest[key]
                indicators_snapshot[key] = float(val) if not pd.isna(val) else None
        
        tradeable = volatility_ok and spread_ok and session_quality > 0.5
        spread_pips = prices[symbol]['spread'] * (100 if 'JPY' in symbol else 10000)
        if not tradeable:
            if not volatility_ok:
                reason = f"ATR out of range: {atr_pips:.1f}p (need {Config.MIN_ATR_PIPS}-{Config.MAX_ATR_PIPS}p)"
            elif not spread_ok:
                reason = f"Spread too wide: {spread_pips:.1f}p"
            elif session_quality <= 0.5:
                reason = f"Poor session quality: {session_quality:.1f}"
            else:
                reason = "Multiple factors"
        else:
            reason = "OK"
        
        logger.info(f"{symbol}: Session={session}({session_quality:.1f}) ATR={atr_pips:.1f}p "
           f"Spread={spread_pips:.2f}p MTF={mtf_bias:.1f} Flow={order_flow:.2f} Trade={tradeable}")
        
        return {
            'tradeable': tradeable,
            'session': session,
            'reason': reason,
            'session_quality': session_quality,
            'volatility': atr,
            'volatility_pips': atr_pips,
            'spread_pips': prices[symbol]['spread'] * (100 if 'JPY' in symbol else 10000),
            'order_flow': order_flow,
            'multi_tf_bias': mtf_bias,
            'trend_strength': trend_strength,
            'trend_direction': 'bullish' if trend_strength > 0.2 else 'bearish' if trend_strength < -0.2 else 'neutral',
            'rsi': latest.get('rsi', 50),
            'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) if (latest['bb_upper'] - latest['bb_lower']) != 0 else 0.5,
            'current_price': latest['close'],
            'bid': prices[symbol]['bid'],
            'ask': prices[symbol]['ask'],
            'indicators': indicators_snapshot,
            'support_resistance': sr_levels,
            'regime': self.get_market_regime()
        }
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        if len(df) < 50:
            return 0.0
        
        latest = df.iloc[-1]
        
        trend_score = 0.0
        
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            if latest['ema_9'] > latest['ema_21']:
                trend_score += 0.3
            else:
                trend_score -= 0.3
        
        if 'sma_20' in df.columns:
            if latest['close'] > latest['sma_20']:
                trend_score += 0.2
            else:
                trend_score -= 0.2
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if latest['macd'] > latest['macd_signal']:
                trend_score += 0.25
            else:
                trend_score -= 0.25
        
        if 'momentum' in df.columns:
            if latest['momentum'] > 0:
                trend_score += 0.25
            else:
                trend_score -= 0.25
        
        return np.clip(trend_score, -1.0, 1.0)

# ==================== SIGNAL GENERATOR ====================
class EnhancedSignalGenerator:
    def __init__(self, analyzer: EnhancedMarketAnalyzer):
        self.analyzer = analyzer
        self.strategies = [
            self.momentum_pullback_signal,
            self.trend_following_signal,
            self.volatility_breakout_signal,
            # mean_reversion_signal REMOVED — structurally broken (shorts into bullish MTF)
            # order_flow_signal REMOVED — fired 13 times in 3 months, not worth the risk
        ]
        self.strategy_performance = {}
    
    def validate_strategy_performance(self, strategy_name: str) -> Tuple[bool, float]:
        """
        Validate strategy performance and return position sizing multiplier
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'wins': 0,
                'losses': 0,
                'pnl': 0,
                'total_trades': 0,
                'total_win': 0,
                'total_loss': 0,
                'enabled': True,
                'position_multiplier': 1.0
            }
            logger.info(f"New strategy {strategy_name} initialized with 100% position sizing")
            return True, 1.0
        
        perf = self.strategy_performance[strategy_name]
        
        if not perf.get('enabled', True):
            return False, 0
        
        if perf['total_trades'] < 10:
            logger.debug(f"{strategy_name}: {perf['total_trades']} trades - using 100% size")
            return True, 1.0
        
        win_rate = perf['wins'] / perf['total_trades'] if perf['total_trades'] > 0 else 0
        avg_win = perf['total_win'] / perf['wins'] if perf['wins'] > 0 else 0
        avg_loss = perf['total_loss'] / perf['losses'] if perf['losses'] > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = (perf['total_win'] / perf['total_loss']) if perf['total_loss'] > 0 else 0
        
        if perf['total_trades'] >= 20:
            if expectancy < -10 or win_rate < 0.25:
                perf['enabled'] = False
                logger.warning(f"Strategy {strategy_name} DISABLED - "
                             f"Expectancy: ${expectancy:.2f}, Win rate: {win_rate:.1%}")
                return False, 0
        
        if perf['total_trades'] >= 50 and expectancy > 20 and win_rate > 0.45:
            position_multiplier = 1.0
            tier = "PROVEN"
        elif perf['total_trades'] >= 30 and expectancy > 10 and win_rate > 0.40:
            position_multiplier = 1.0
            tier = "PERFORMING"
        elif perf['total_trades'] >= 20 and expectancy > 0:
            position_multiplier = 1.0
            tier = "DEVELOPING"
        else:
            position_multiplier = 1.0
            tier = "TESTING"
        
        perf['position_multiplier'] = position_multiplier
        
        if perf['total_trades'] % 10 == 0:
            logger.info(f"Strategy {strategy_name} [{tier}]: "
                       f"Trades={perf['total_trades']} WR={win_rate:.1%} "
                       f"Exp=${expectancy:.2f} PF={profit_factor:.2f} "
                       f"Size={position_multiplier:.0%}")
        
        return True, position_multiplier

    def generate_signals(self, symbol: str) -> Optional[Dict]:
        """Generate trading signals with time window filtering"""
        market_condition = self.analyzer.get_market_condition(symbol)
        
        if not market_condition['tradeable']:
            return None
        
        # CHECK TIME WINDOW FILTER
        in_window, window_reason = is_in_trading_window()
        if not in_window:
            logger.debug(f"{symbol}: {window_reason}")
            return None
        
        if abs(market_condition['multi_tf_bias']) < 0.5:
            logger.debug(f"{symbol}: Insufficient multi-timeframe alignment")
            return None
        
        # HURST REGIME FILTER — only trade in trending markets
        hurst = self.analyzer.compute_hurst_exponent(symbol)
        if hurst < Config.HURST_THRESHOLD:
            logger.info(f"{symbol}: Hurst {hurst:.3f} < {Config.HURST_THRESHOLD} — market not trending, skipping")
            return None
        
        signals = []
        for strategy in self.strategies:
            try:
                strategy_name = strategy.__name__.replace('_signal', '')
                
                can_trade, position_multiplier = self.validate_strategy_performance(strategy_name)
                if not can_trade:
                    logger.debug(f"Strategy {strategy_name} disabled for {symbol}")
                    continue
                
                signal = strategy(symbol, market_condition)
                if signal:
                    risk = abs(signal['entry'] - signal['stop_loss'])
                    reward = abs(signal['take_profit'] - signal['entry'])
                    rr_ratio = reward / risk if risk > 0 else 0
                    
                    if rr_ratio < Config.MIN_RISK_REWARD:
                        logger.debug(f"{symbol}: {signal['strategy']} R/R too low: {rr_ratio:.2f}")
                        continue
                    
                    signal['risk_reward'] = rr_ratio
                    signal['position_multiplier'] = position_multiplier
                    signal['market_conditions'] = {
                        'session': market_condition['session'],
                        'session_quality': market_condition['session_quality'],
                        'volatility': market_condition['volatility'],
                        'spread_pips': market_condition['spread_pips'],
                        'trend_strength': market_condition['trend_strength'],
                        'trend_direction': market_condition['trend_direction'],
                        'order_flow': market_condition['order_flow'],
                        'multi_tf_bias': market_condition['multi_tf_bias'],
                        'regime': market_condition['regime']
                    }
                    signal['indicators'] = market_condition['indicators']
                    
                    signal_record = SignalRecord(
                        timestamp=datetime.now().isoformat(),
                        symbol=symbol,
                        strategy=signal['strategy'],
                        direction=signal['direction'],
                        confidence=signal['confidence'],
                        entry_price=signal['entry'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit'],
                        risk_reward=rr_ratio,
                        market_conditions=signal['market_conditions'],
                        indicators=signal['indicators'],
                        traded=False,
                        not_traded_reason=f"Position multiplier: {position_multiplier:.2f}" if position_multiplier < 1 else None,
                        session=market_condition['session']
                    )
                    advanced_logger.log_signal(signal_record)
                    
                    # Store reference so execute_signal can update traded status
                    signal['_signal_record'] = signal_record
                    
                    signals.append(signal)
                    
                    logger.info(f"SIGNAL: {symbol} {signal['strategy']} {signal['direction']} "
                              f"conf={signal['confidence']:.2f} R/R={rr_ratio:.1f} "
                              f"size_mult={position_multiplier:.2f}")
                        
            except Exception as e:
                logger.error(f"Error in {strategy.__name__} for {symbol}: {e}")
        
        if not signals:
            return None
        
        valid_signals = [s for s in signals if s['confidence'] >= Config.CONFIDENCE_THRESHOLD]
        if not valid_signals:
            logger.debug(f"{symbol}: No signals above confidence threshold {Config.CONFIDENCE_THRESHOLD}")
            return None
        
        best_signal = max(valid_signals, 
                         key=lambda x: (x['confidence'] * x['risk_reward'] * x['position_multiplier']))
        
        logger.info(f"Selected {best_signal['strategy']} for {symbol} "
                   f"(multiplier: {best_signal['position_multiplier']:.2f})")
        
        return best_signal
    
    def momentum_pullback_signal(self, symbol: str, market_condition: Dict) -> Optional[Dict]:
        df = self.analyzer.get_market_data(symbol, Config.TIMEFRAMES['primary'])
        if df.empty or len(df) < 20:
            return None

        latest = df.iloc[-1]

        if abs(market_condition['multi_tf_bias']) < 0.5:
            return None

        momentum = (df['close'].iloc[-1] / df['close'].iloc[-12] - 1)
        if 'JPY' in symbol:
            momentum_pips = momentum * 100
        else:
            momentum_pips = momentum * 10000

        if abs(momentum_pips) < 10:
            return None

        signal = None
        confidence = 0.30  # Low base — earn it

        last_3_bars = df.iloc[-3:]

        if momentum_pips > 0 and market_condition['multi_tf_bias'] > 0:
            pullback = last_3_bars['low'].min() < df['close'].iloc[-4]
            if pullback and latest['rsi'] < 70 and market_condition['order_flow'] > -0.3:
                signal = 'buy'

                # Momentum magnitude: stronger momentum = higher confidence
                if abs(momentum_pips) > 30:
                    confidence += 0.15
                elif abs(momentum_pips) > 20:
                    confidence += 0.10
                else:
                    confidence += 0.05

                # RSI not overextended (ideal pullback zone 40-60)
                if 40 <= latest['rsi'] <= 60:
                    confidence += 0.10
                elif latest['rsi'] < 40:
                    confidence += 0.05  # Deep pullback — riskier

                if market_condition['session_quality'] > 0.8:
                    confidence += 0.05

                if market_condition['trend_direction'] == 'bullish':
                    confidence += 0.10

                confidence += market_condition['multi_tf_bias'] * 0.10

                if market_condition['order_flow'] > 0.2:
                    confidence += 0.05

        elif momentum_pips < 0 and market_condition['multi_tf_bias'] < 0:
            pullback = last_3_bars['high'].max() > df['close'].iloc[-4]
            if pullback and latest['rsi'] > 30 and market_condition['order_flow'] < 0.3:
                signal = 'sell'

                if abs(momentum_pips) > 30:
                    confidence += 0.15
                elif abs(momentum_pips) > 20:
                    confidence += 0.10
                else:
                    confidence += 0.05

                if 40 <= latest['rsi'] <= 60:
                    confidence += 0.10
                elif latest['rsi'] > 60:
                    confidence += 0.05

                if market_condition['session_quality'] > 0.8:
                    confidence += 0.05

                if market_condition['trend_direction'] == 'bearish':
                    confidence += 0.10

                confidence += abs(market_condition['multi_tf_bias']) * 0.10

                if market_condition['order_flow'] < -0.2:
                    confidence += 0.05

        if not signal:
            return None

        return self._create_signal_with_static_targets(
            symbol, signal, 'momentum_pullback', confidence,
            market_condition, df
        )
    
    def order_flow_signal(self, symbol: str, market_condition: Dict) -> Optional[Dict]:
        if abs(market_condition['order_flow']) < 0.5:
            return None
        
        if abs(market_condition['multi_tf_bias']) < 0.5:
            return None
        
        df = self.analyzer.get_market_data(symbol, Config.TIMEFRAMES['primary'])
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        
        signal = None
        confidence = 0.5 + abs(market_condition['order_flow']) * 0.3
        
        if market_condition['order_flow'] > 0.5 and market_condition['multi_tf_bias'] > 0:
            if latest['rsi'] < 70:
                signal = 'buy'
                if market_condition['regime'] == 'risk_on':
                    confidence += 0.1
        elif market_condition['order_flow'] < -0.5 and market_condition['multi_tf_bias'] < 0:
            if latest['rsi'] > 30:
                signal = 'sell'
                if market_condition['regime'] == 'risk_off':
                    confidence += 0.1
        
        if not signal:
            return None
        
        return self._create_signal_with_static_targets(
            symbol, signal, 'order_flow', confidence,
            market_condition, df
        )
    
    def trend_following_signal(self, symbol: str, market_condition: Dict) -> Optional[Dict]:
        if abs(market_condition['trend_strength']) < 0.3:
            return None

        if abs(market_condition['multi_tf_bias']) < 0.5:
            return None

        df = self.analyzer.get_market_data(symbol, Config.TIMEFRAMES['primary'])
        if df.empty or len(df) < 10:
            return None

        latest = df.iloc[-1]

        signal = None
        # Start at a LOW base — confidence must be earned by the quality of the setup
        confidence = 0.30

        if len(df) >= 2:
            prev = df.iloc[-2]

            macd_cross_up = latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']
            macd_cross_down = latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']

            momentum_up = latest['momentum'] > 0 and latest['rsi'] > 45
            momentum_down = latest['momentum'] < 0 and latest['rsi'] < 55

            if market_condition['trend_direction'] == 'bullish' and market_condition['multi_tf_bias'] > 0:
                if macd_cross_up or momentum_up:
                    signal = 'buy'

                    # MACD cross is a stronger signal than just momentum
                    if macd_cross_up:
                        confidence += 0.20
                    if momentum_up:
                        confidence += 0.10

                    # Order flow alignment (now actually produces values)
                    of = market_condition['order_flow']
                    if of > 0.3:
                        confidence += 0.10
                    elif of > 0.1:
                        confidence += 0.05

                    # MTF bias strength (0.5-1.0 range maps to 0.05-0.10)
                    confidence += market_condition['multi_tf_bias'] * 0.10

                    # Trend strength (0.3-1.0 range maps to 0.03-0.10)
                    confidence += abs(market_condition['trend_strength']) * 0.10

                    # Regime bonus
                    if market_condition.get('regime') == 'risk_on':
                        confidence += 0.05

                    # Session quality
                    if market_condition['session_quality'] >= 0.9:
                        confidence += 0.05

            elif market_condition['trend_direction'] == 'bearish' and market_condition['multi_tf_bias'] < 0:
                if macd_cross_down or momentum_down:
                    signal = 'sell'

                    if macd_cross_down:
                        confidence += 0.20
                    if momentum_down:
                        confidence += 0.10

                    of = market_condition['order_flow']
                    if of < -0.3:
                        confidence += 0.10
                    elif of < -0.1:
                        confidence += 0.05

                    confidence += abs(market_condition['multi_tf_bias']) * 0.10
                    confidence += abs(market_condition['trend_strength']) * 0.10

                    if market_condition.get('regime') == 'risk_off':
                        confidence += 0.05

                    if market_condition['session_quality'] >= 0.9:
                        confidence += 0.05

        if not signal:
            return None

        return self._create_signal_with_static_targets(
            symbol, signal, 'trend_following', confidence,
            market_condition, df
        )
    
    def mean_reversion_signal(self, symbol: str, market_condition: Dict) -> Optional[Dict]:
        if abs(market_condition['trend_strength']) > 0.4:
            return None
        
        df = self.analyzer.get_market_data(symbol, Config.TIMEFRAMES['primary'])
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        
        signal = None
        confidence = 0.0
        
        if latest['rsi'] < 30:
            signal = 'buy'
            confidence = 0.6 + (30 - latest['rsi']) / 50
        elif latest['rsi'] > 70:
            signal = 'sell'
            confidence = 0.6 + (latest['rsi'] - 70) / 50
        
        if not signal:
            return None
        
        bb_width = latest['bb_upper'] - latest['bb_lower']
        if bb_width > 0:
            if signal == 'buy' and latest['close'] <= latest['bb_lower'] * 1.01:
                confidence += 0.2
            elif signal == 'sell' and latest['close'] >= latest['bb_upper'] * 0.99:
                confidence += 0.2
        
        if signal == 'buy' and latest['stoch_k'] < 20:
            confidence += 0.1
        elif signal == 'sell' and latest['stoch_k'] > 80:
            confidence += 0.1
        
        if signal == 'buy' and market_condition['order_flow'] > 0:
            confidence += 0.1
        elif signal == 'sell' and market_condition['order_flow'] < 0:
            confidence += 0.1
        
        if confidence < Config.CONFIDENCE_THRESHOLD:
            return None
        
        return self._create_signal_with_static_targets(
            symbol, signal, 'mean_reversion', confidence,
            market_condition, df
        )
    
    def volatility_breakout_signal(self, symbol: str, market_condition: Dict) -> Optional[Dict]:
        df = self.analyzer.get_market_data(symbol, Config.TIMEFRAMES['primary'])
        if df.empty or len(df) < 20:
            return None

        latest = df.iloc[-1]

        recent_atr = df['atr'].iloc[-5:].mean()
        longer_atr = df['atr'].iloc[-20:].mean()

        if recent_atr <= longer_atr * 1.15:
            return None

        signal = None
        confidence = 0.30  # Low base

        # ATR expansion magnitude — stronger breakouts get more confidence
        atr_expansion = recent_atr / longer_atr if longer_atr > 0 else 1.0

        if latest['close'] > latest['bb_upper']:
            signal = 'buy'

            # BB breakout is the core signal
            confidence += 0.15

            # ATR expansion strength
            if atr_expansion > 1.5:
                confidence += 0.10
            elif atr_expansion > 1.25:
                confidence += 0.05

            # RSI in the right zone (strong but not exhausted)
            if 60 <= latest['rsi'] <= 75:
                confidence += 0.10
            elif 55 <= latest['rsi'] < 60:
                confidence += 0.05

            if market_condition['multi_tf_bias'] > 0:
                confidence += 0.10

        elif latest['close'] < latest['bb_lower']:
            signal = 'sell'

            confidence += 0.15

            if atr_expansion > 1.5:
                confidence += 0.10
            elif atr_expansion > 1.25:
                confidence += 0.05

            if 25 <= latest['rsi'] <= 40:
                confidence += 0.10
            elif 40 < latest['rsi'] <= 45:
                confidence += 0.05

            if market_condition['multi_tf_bias'] < 0:
                confidence += 0.10

        if not signal:
            return None

        if latest.get('volume_ratio', 1.0) > 1.5:
            confidence += 0.05

        # Order flow alignment
        of = market_condition['order_flow']
        if (signal == 'buy' and of > 0.2) or (signal == 'sell' and of < -0.2):
            confidence += 0.05

        if confidence < Config.CONFIDENCE_THRESHOLD:
            return None

        return self._create_signal_with_static_targets(
            symbol, signal, 'volatility_breakout', confidence,
            market_condition, df
        )
    
    def _create_signal_with_static_targets(self, symbol: str, direction: str, strategy: str,
                                           confidence: float, market_condition: Dict, df: pd.DataFrame) -> Dict:
        """FIXED: Use static 130/60 TP/SL without any overrides"""
        
        if direction == 'buy':
            current_price = market_condition['ask']
        else:
            current_price = market_condition['bid']
        
        # Calculate pip value
        if 'JPY' in symbol:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        # Set FIXED static TP/SL - 130/60
        if direction == 'buy':
            stop_loss = current_price - (Config.STATIC_SL_PIPS * pip_value)
            take_profit = current_price + (Config.STATIC_TP_PIPS * pip_value)
        else:
            stop_loss = current_price + (Config.STATIC_SL_PIPS * pip_value)
            take_profit = current_price - (Config.STATIC_TP_PIPS * pip_value)
        
        logger.info(f"TP/SL SET: {symbol} {direction} Entry={current_price:.5f} " +
                   f"SL={stop_loss:.5f} ({Config.STATIC_SL_PIPS}p) " +
                   f"TP={take_profit:.5f} ({Config.STATIC_TP_PIPS}p)")
        
        return {
            'strategy': strategy,
            'direction': direction,
            'confidence': min(confidence, 0.95),
            'entry': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'session': market_condition['session']
        }

# ==================== RISK MANAGER ====================
class EnhancedRiskManager:
    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_amounts = []
        self.loss_amounts = []
        self.last_reset = datetime.now().date()
        self.returns_history = []
        self.equity_history = []
        self.max_daily_trades = Config.MAX_DAILY_TRADES
    
    def reset_daily_stats(self):
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            if self.daily_pnl != 0:
                self.returns_history.append(self.daily_pnl)
                if len(self.returns_history) > 252:
                    self.returns_history = self.returns_history[-252:]
            
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = current_date
            logger.info("Daily stats reset")
    
    def should_pause_trading(self, current_balance: float) -> bool:
        self.equity_history.append(current_balance)
        if len(self.equity_history) > 100:
            self.equity_history = self.equity_history[-100:]
        
        if len(self.equity_history) < 20:
            return False
        
        current_equity = self.equity_history[-1]
        ma_20 = np.mean(self.equity_history[-20:])
        
        if current_equity < ma_20 * 0.95:
            logger.warning(f"Equity ({current_equity:.2f}) below MA ({ma_20:.2f}) - Pausing trading")
            return True
        
        return False
    
    def can_trade(self, account_balance: float, session_quality: float = 1.0) -> Tuple[bool, str]:
        self.reset_daily_stats()
        
        if self.daily_trades >= self.max_daily_trades:
            reason = f"Max daily trades ({self.max_daily_trades}) reached"
            return False, reason
        
        return True, "OK"

    def calculate_position_size_gbp(self, signal: Dict, account_balance: float, symbol: str, 
                                   oanda_client: OandaClient) -> float:
        """FIXED position sizing for GBP accounts with proper pip value calculation"""
        
        risk_amount = account_balance * Config.RISK_PER_TRADE
        
        if Config.USE_KELLY_CRITERION and len(self.win_amounts) >= 20:
            kelly_fraction = self._calculate_kelly_fraction()
            if kelly_fraction > 0:
                kelly_size = account_balance * min(kelly_fraction, Config.MAX_KELLY_FRACTION)
                risk_amount = min(risk_amount, kelly_size)
        
        if 'position_multiplier' in signal:
            risk_amount *= signal['position_multiplier']
            logger.debug(f"Strategy position multiplier: {signal['position_multiplier']}")
        
        risk_amount *= (0.5 + 0.5 * signal['confidence'])
        
        if 'session' in signal:
            session_quality = Config.SESSIONS.get(signal['session'], {}).get('quality', 1.0)
            risk_amount *= (0.8 + 0.4 * session_quality)
        
        entry = signal['entry']
        stop_loss = signal['stop_loss']
        
        if 'JPY' in symbol:
            pip_distance = abs(entry - stop_loss) * 100
        else:
            pip_distance = abs(entry - stop_loss) * 10000
        
        base_currency = symbol.split('_')[0]
        quote_currency = symbol.split('_')[1]
        
        if Config.ACCOUNT_CURRENCY == 'GBP':
            if quote_currency == 'GBP':
                if 'JPY' in symbol:
                    pip_value_per_unit = 0.01
                else:
                    pip_value_per_unit = 0.0001
                    
            elif base_currency == 'GBP':
                current_price = (entry + stop_loss) / 2
                if 'JPY' in symbol:
                    pip_value_per_unit = 0.01 / current_price
                else:
                    pip_value_per_unit = 0.0001 / current_price
                    
            else:
                gbp_usd_rate = oanda_client.get_current_exchange_rate('GBP', 'USD')
                
                if quote_currency == 'USD':
                    pip_value_per_unit = 0.0001 / gbp_usd_rate
                    
                elif base_currency == 'USD':
                    current_price = (entry + stop_loss) / 2
                    if 'JPY' in symbol:
                        pip_value_per_unit = 0.01 / current_price / gbp_usd_rate
                    else:
                        pip_value_per_unit = 0.0001 / current_price / gbp_usd_rate
                        
                else:
                    quote_to_gbp = oanda_client.get_conversion_rate(quote_currency, 'GBP')
                    if 'JPY' in symbol:
                        pip_value_per_unit = 0.01 * quote_to_gbp
                    else:
                        pip_value_per_unit = 0.0001 * quote_to_gbp
                        
        else:
            if quote_currency == 'USD':
                if 'JPY' in symbol:
                    pip_value_per_unit = 0.0001
                else:
                    pip_value_per_unit = 0.0001
            elif base_currency == 'USD':
                current_rate = (entry + stop_loss) / 2
                if 'JPY' in symbol:
                    pip_value_per_unit = 0.01 / current_rate
                else:
                    pip_value_per_unit = 0.0001 / current_rate
            else:
                conversion_rate = oanda_client.get_conversion_rate(quote_currency, 'USD')
                if 'JPY' in symbol:
                    pip_value_per_unit = 0.01 * conversion_rate
                else:
                    pip_value_per_unit = 0.0001 * conversion_rate
        
        if pip_distance > 0 and pip_value_per_unit > 0:
            units = risk_amount / (pip_distance * pip_value_per_unit)
        else:
            logger.error(f"Invalid position calculation: pip_distance={pip_distance}, pip_value={pip_value_per_unit}")
            return 0
        
        units = int(units)
        
        # MARGIN SAFETY CAP: ensure MAX_POSITIONS concurrent trades fit within margin budget
        # OANDA margin = (units / leverage) in base currency, converted to account currency
        # So: margin_in_GBP = (units / leverage) * base_to_GBP
        # Solving: max_units = budget * leverage / base_to_GBP
        try:
            margin_budget_per_trade = (account_balance * Config.MAX_MARGIN_USAGE_PCT) / Config.MAX_POSITIONS
            
            base_ccy = symbol.split('_')[0]
            if base_ccy == Config.ACCOUNT_CURRENCY:
                base_to_acct = 1.0
            else:
                base_to_acct = oanda_client.get_conversion_rate(base_ccy, Config.ACCOUNT_CURRENCY)
                if base_to_acct <= 0:
                    base_to_acct = 1.0  # Fallback
            
            max_units_for_margin = int(margin_budget_per_trade * Config.LEVERAGE / base_to_acct)
            
            if units > max_units_for_margin:
                logger.info(f"MARGIN CAP: {symbol} reduced {units} → {max_units_for_margin} units "
                           f"(budget £{margin_budget_per_trade:.0f} per trade, base_to_GBP={base_to_acct:.4f})")
                units = max_units_for_margin
        except Exception as e:
            logger.warning(f"Margin cap calculation failed: {e}")
        
        # Hard floor: 1 unit minimum, no artificial minimums that inflate risk
        units = max(units, 1)
        
        # Hard ceiling: 1M units
        units = min(units, 1000000)
        
        expected_loss = pip_distance * pip_value_per_unit * units
        if abs(expected_loss - risk_amount) > risk_amount * 0.5:
            logger.warning(f"Position validation: Expected loss £{expected_loss:.2f} vs Risk £{risk_amount:.2f}")
        
        logger.info(f"GBP Position Calc: {symbol} | {units} units | Risk: £{risk_amount:.2f} | "
                   f"Stop: {pip_distance:.1f}p | PipVal/unit: £{pip_value_per_unit:.8f}")
        
        return units

    def calculate_position_size(self, signal: Dict, account_balance: float, symbol: str, 
                              oanda_client: OandaClient) -> float:
        """Router to use GBP-specific calculation if needed"""
        if Config.ACCOUNT_CURRENCY == 'GBP':
            return self.calculate_position_size_gbp(signal, account_balance, symbol, oanda_client)
        
        risk_amount = account_balance * Config.RISK_PER_TRADE
        
        if Config.USE_KELLY_CRITERION and len(self.win_amounts) >= 20:
            kelly_fraction = self._calculate_kelly_fraction()
            if kelly_fraction > 0:
                kelly_size = account_balance * min(kelly_fraction, Config.MAX_KELLY_FRACTION)
                risk_amount = min(risk_amount, kelly_size)
        
        if 'position_multiplier' in signal:
            risk_amount *= signal['position_multiplier']
            logger.debug(f"Strategy position multiplier: {signal['position_multiplier']}")
        
        risk_amount *= (0.5 + 0.5 * signal['confidence'])
        
        if 'session' in signal:
            session_quality = Config.SESSIONS.get(signal['session'], {}).get('quality', 1.0)
            risk_amount *= (0.8 + 0.4 * session_quality)
        
        entry = signal['entry']
        stop_loss = signal['stop_loss']
        
        if 'JPY' in symbol:
            pip_distance = abs(entry - stop_loss) * 100
        else:
            pip_distance = abs(entry - stop_loss) * 10000

        base_currency = symbol.split('_')[0]
        quote_currency = symbol.split('_')[1]

        if quote_currency == 'USD':
            if 'JPY' in symbol:
                pip_value_per_unit = 0.0001
            else:
                pip_value_per_unit = 0.0001
        elif base_currency == 'USD':
            current_rate = (entry + stop_loss) / 2
            if 'JPY' in symbol:
                pip_value_per_unit = 0.01 / current_rate
            else:
                pip_value_per_unit = 0.0001 / current_rate
        else:
            conversion_rate = oanda_client.get_conversion_rate(quote_currency, 'USD')
            if 'JPY' in symbol:
                pip_value_per_unit = 0.01 * conversion_rate
            else:
                pip_value_per_unit = 0.0001 * conversion_rate
                
        if pip_distance > 0 and pip_value_per_unit > 0:
            units = risk_amount / (pip_distance * pip_value_per_unit)
        else:
            logger.error(f"Invalid position calculation: pip_distance={pip_distance}, pip_value={pip_value_per_unit}")
            return 0
        
        units = int(units)
        units = max(units, 1)
        units = min(units, 1000000)
        
        expected_loss = pip_distance * pip_value_per_unit * units
        if abs(expected_loss - risk_amount) > risk_amount * 0.5:
            logger.warning(f"Position validation: Expected loss ${expected_loss:.2f} vs Risk ${risk_amount:.2f}")
        
        logger.debug(f"Position: {units} units | Risk: ${risk_amount:.2f} | Pips: {pip_distance:.1f} | PipValue/unit: ${pip_value_per_unit:.8f}")
        
        return units
    
    def check_correlation(self, symbol: str, open_positions: List[Dict]) -> bool:
        """Enhanced correlation check with portfolio heat calculation"""
        if not open_positions:
            return True
        
        open_symbols = [pos['instrument'] for pos in open_positions]
        
        for group in Config.CORRELATION_GROUPS:
            if symbol in group:
                correlated_positions = [s for s in open_symbols if s in group]
                if len(correlated_positions) >= 2:
                    logger.debug(f"Skipping {symbol} - too many correlated positions in group")
                    return False
        
        if not self.calculate_portfolio_heat(symbol, open_positions):
            logger.debug(f"Skipping {symbol} - portfolio heat too high")
            return False
        
        return True

    def calculate_portfolio_heat(self, new_symbol: str, open_positions: List[Dict]) -> bool:
        """Calculate total portfolio correlation risk"""
        correlation_matrix = self.get_correlation_matrix()
        
        total_correlated_risk = 0
        account_balance = self.equity_history[-1] if self.equity_history else 10000
        
        new_position_risk = 0.01
        
        for position in open_positions:
            pos_symbol = position['instrument']
            position_risk = 0.01
            
            correlation = correlation_matrix.get(new_symbol, {}).get(pos_symbol, 0)
            
            correlated_risk = new_position_risk * position_risk * abs(correlation)
            total_correlated_risk += correlated_risk
        
        max_correlated_risk = Config.MAX_CORRELATION_RISK if hasattr(Config, 'MAX_CORRELATION_RISK') else 0.02
        
        if total_correlated_risk > max_correlated_risk:
            logger.warning(f"Portfolio heat too high: {total_correlated_risk:.3f} > {max_correlated_risk}")
            return False
        
        return True

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get simplified correlation matrix based on currency relationships"""
        matrix = {}
        
        for symbol1 in Config.SYMBOLS:
            matrix[symbol1] = {}
            for symbol2 in Config.SYMBOLS:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = 1.0
                else:
                    currencies1 = set(symbol1.split('_'))
                    currencies2 = set(symbol2.split('_'))
                    common = currencies1.intersection(currencies2)
                    
                    if len(common) == 2:
                        matrix[symbol1][symbol2] = 1.0
                    elif len(common) == 1:
                        if common.pop() in ['USD', 'EUR', 'GBP']:
                            matrix[symbol1][symbol2] = 0.5
                        else:
                            matrix[symbol1][symbol2] = 0.3
                    else:
                        risk_on = ['AUD', 'NZD', 'GBP']
                        risk_off = ['JPY', 'CHF', 'USD']
                        
                        curr1_risk = any(c in risk_on for c in currencies1)
                        curr2_risk = any(c in risk_on for c in currencies2)
                        curr1_safe = any(c in risk_off for c in currencies1)
                        curr2_safe = any(c in risk_off for c in currencies2)
                        
                        if (curr1_risk and curr2_risk) or (curr1_safe and curr2_safe):
                            matrix[symbol1][symbol2] = 0.2
                        else:
                            matrix[symbol1][symbol2] = 0.0
        
        return matrix

    def _calculate_kelly_fraction(self) -> float:
        if not self.win_amounts or not self.loss_amounts:
            return 0.0
        
        wins = len(self.win_amounts)
        losses = len(self.loss_amounts)
        total = wins + losses
        
        if total < 20:
            return 0.0
        
        win_prob = wins / total
        avg_win = np.mean(self.win_amounts)
        avg_loss = np.mean(self.loss_amounts)
        
        if avg_loss == 0:
            return Config.MAX_KELLY_FRACTION
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        kelly = kelly * 0.25
        
        return max(0, min(kelly, Config.MAX_KELLY_FRACTION))
    
    def calculate_sharpe_ratio(self) -> float:
        if len(self.returns_history) < 20:
            return 0.0
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return / std_return) * np.sqrt(252)
    
    def calculate_expectancy(self) -> float:
        if not self.win_amounts and not self.loss_amounts:
            return 0.0
        
        total_trades = len(self.win_amounts) + len(self.loss_amounts)
        if total_trades == 0:
            return 0.0
        
        win_rate = len(self.win_amounts) / total_trades
        avg_win = np.mean(self.win_amounts) if self.win_amounts else 0
        avg_loss = np.mean(self.loss_amounts) if self.loss_amounts else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return expectancy
    
    def update_trade_stats(self, pnl: float, strategy: str = None):
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            self.win_amounts.append(pnl)
        else:
            self.losing_trades += 1
            self.loss_amounts.append(abs(pnl))
        
        if len(self.win_amounts) > 100:
            self.win_amounts = self.win_amounts[-100:]
        if len(self.loss_amounts) > 100:
            self.loss_amounts = self.loss_amounts[-100:]

# ==================== TRADE EXECUTOR ====================
class EnhancedTradeExecutor:
    def __init__(self, oanda_client: OandaClient, risk_manager: EnhancedRiskManager,
                 persistent_state: 'PersistentState' = None):
        self.client = oanda_client
        self.risk_manager = risk_manager
        self.persistent_state = persistent_state
        self.trade_records = {}
        self.last_trade_time = {}
        self.last_tp_time = {}  # Track when each symbol last hit TP (for cooldown)
        self.partial_targets = {}
        self.trailing_stops = {}
        self.excursion_trackers = {}
        self.closed_trades_tracked = set()

        # Initialize blocked trade tracker
        self.blocked_trade_tracker = BlockedTradeTracker(oanda_client, risk_manager)
        logger.info("Blocked trade tracking enabled - Will analyze virtual outcomes with position sizing")
    
    def calculate_pnl_pips(self, symbol: str, entry_price: float, exit_price: float, direction: str) -> float:
        """Calculate P&L in pips"""
        price_diff = exit_price - entry_price
        if direction == 'sell':
            price_diff = -price_diff
        
        if 'JPY' in symbol:
            return price_diff * 100
        else:
            return price_diff * 10000
    
    def log_blocked_trade(self, signal: Dict, symbol: str, account_balance: float, blocked_reason: str):
        """Log a blocked trade attempt to CSV and start tracking virtual outcome"""
        trade_record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            trade_id=f"BLOCKED_{symbol}_{int(time.time())}",
            symbol=symbol,
            direction=signal['direction'],
            strategy=signal['strategy'],
            confidence=signal['confidence'],
            entry_price=signal['entry'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            position_size=0,
            risk_amount=0,
            market_conditions=signal.get('market_conditions', {}),
            indicators=signal.get('indicators', {}),
            status="BLOCKED",
            blocked_reason=blocked_reason,
            session=signal.get('session')
        )

        advanced_logger.log_trade(trade_record)
        logger.info(f"🚫 TRADE BLOCKED: {symbol} {signal['direction'].upper()} - {blocked_reason}")

        # Add to blocked trade tracker for virtual outcome analysis
        self.blocked_trade_tracker.add_blocked_trade(signal, symbol, blocked_reason, account_balance)
        logger.info(f"📊 Started tracking virtual outcome for blocked {symbol} trade")
    
    def execute_signal(self, signal: Dict, symbol: str, account_balance: float) -> bool:
        """Execute signal - CRITICAL FIX: NO TP/SL OVERRIDES"""
        if symbol in self.last_trade_time:
            if time.time() - self.last_trade_time[symbol] < 300:
                logger.debug(f"Cooldown active for {symbol}")
                return False

        # POST-TP COOLDOWN: Don't re-enter a pair that just hit TP — the move is exhausted
        if symbol in self.last_tp_time:
            elapsed = time.time() - self.last_tp_time[symbol]
            if elapsed < Config.POST_TP_COOLDOWN_SECONDS:
                remaining = (Config.POST_TP_COOLDOWN_SECONDS - elapsed) / 60
                logger.info(f"POST-TP COOLDOWN: {symbol} hit TP {elapsed/60:.0f}m ago, "
                           f"waiting {remaining:.0f}m more before re-entry")
                return False
        
        positions = self.client.get_open_positions()
        
        for position in positions:
            if position['instrument'] == symbol:
                logger.info(f"Already have position in {symbol}")
                return False
        
        if len(positions) >= Config.MAX_POSITIONS:
            logger.info("Maximum positions reached")
            return False
        
        if not self.risk_manager.check_correlation(symbol, positions):
            return False
        
        # Pre-trade margin check — stop the order before OANDA rejects it
        try:
            account_info = self.client.get_account_info()
            margin_available = float(account_info.get('marginAvailable', 0))
            prices_check = self.client.get_prices([symbol])
            if symbol in prices_check:
                est_price = prices_check[symbol]['ask']
                est_units = abs(self.risk_manager.calculate_position_size(signal, account_balance, symbol, self.client))
                est_margin_needed = est_units * est_price * 0.0333  # ~30:1 leverage
                base_ccy = symbol.split('_')[0]
                if base_ccy != 'GBP':
                    fx_approx = {'USD': 0.80, 'EUR': 0.86, 'AUD': 0.52, 'NZD': 0.48, 'CAD': 0.58, 'JPY': 0.0053}
                    est_margin_needed *= fx_approx.get(base_ccy, 0.80)
                margin_ratio = margin_available / est_margin_needed if est_margin_needed > 0 else 999
                if margin_ratio < 1.5:
                    logger.warning(f"🚫 PRE-CHECK: Insufficient margin for {symbol}. "
                                  f"Available: £{margin_available:.0f}, Needed: ~£{est_margin_needed:.0f} "
                                  f"(ratio: {margin_ratio:.2f}, need 1.5x)")
                    self.log_blocked_trade(signal, symbol, account_balance,
                                          f"Pre-check margin insufficient: {margin_ratio:.2f}x")
                    return False
                else:
                    logger.debug(f"Margin OK: {symbol} £{margin_available:.0f} avail, "
                                f"~£{est_margin_needed:.0f} needed ({margin_ratio:.1f}x)")
        except Exception as e:
            logger.warning(f"Margin pre-check failed (proceeding anyway): {e}")
        
        best_prices = None
        best_spread = float('inf')
        
        for attempt in range(5):
            prices = self.client.get_prices([symbol])
            if symbol in prices:
                current_spread = prices[symbol]['spread']
                if current_spread < best_spread:
                    best_prices = prices[symbol]
                    best_spread = current_spread
                
                spread_pips = best_spread * (100 if 'JPY' in symbol else 10000)
                if spread_pips <= 1.0:
                    break
            
            if attempt < 4:
                time.sleep(0.2)
        
        if not best_prices:
            return False
        
        spread_cost = self.calculate_spread_cost(symbol, {symbol: best_prices})
        
        units = self.risk_manager.calculate_position_size(signal, account_balance, symbol, self.client)
        if units == 0:
            logger.error(f"⚠️ POSITION SIZE CALCULATION FAILED for {symbol}")
            logger.error(f"Signal: entry={signal['entry']}, sl={signal['stop_loss']}, tp={signal['take_profit']}")
            logger.error(f"Account balance: {account_balance}, Risk per trade: {Config.RISK_PER_TRADE}")
            # This is a calculation error, not a margin issue, so don't track as blocked trade
            # These errors indicate code bugs that need fixing
            return False
        
        session_quality = signal.get('market_conditions', {}).get('session_quality', 1.0)
        expected_slippage = self.estimate_slippage(symbol, units, session_quality)
        
        # CRITICAL FIX: Keep original TP/SL from signal, don't recalculate!
        original_sl = signal['stop_loss']
        original_tp = signal['take_profit']
        
        # Adjust entry for costs but keep TP/SL pip distances exact
        if signal['direction'] == 'buy':
            adjusted_entry = best_prices['ask'] + expected_slippage
            units = abs(units)
            
            # Maintain EXACT pip distances (130/60)
            sl_pips = Config.STATIC_SL_PIPS
            tp_pips = Config.STATIC_TP_PIPS
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            
            signal['stop_loss'] = adjusted_entry - (sl_pips * pip_value)
            signal['take_profit'] = adjusted_entry + (tp_pips * pip_value)
        else:
            adjusted_entry = best_prices['bid'] - expected_slippage
            units = -abs(units)
            
            # Maintain EXACT pip distances (130/60)
            sl_pips = Config.STATIC_SL_PIPS
            tp_pips = Config.STATIC_TP_PIPS
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            
            signal['stop_loss'] = adjusted_entry + (sl_pips * pip_value)
            signal['take_profit'] = adjusted_entry - (tp_pips * pip_value)
        
        # Log the adjustments
        original_entry = best_prices['ask'] if signal['direction'] == 'buy' else best_prices['bid']
        total_cost_pips = abs(adjusted_entry - original_entry) * (100 if 'JPY' in symbol else 10000)
        logger.info(f"Entry adjustment: Original={original_entry:.5f} Adjusted={adjusted_entry:.5f} "
                   f"Cost={total_cost_pips:.2f} pips")
        
        # Verify TP/SL distances (should be 130/60)
        actual_sl_pips = abs(signal['stop_loss'] - adjusted_entry) * (100 if 'JPY' in symbol else 10000)
        actual_tp_pips = abs(signal['take_profit'] - adjusted_entry) * (100 if 'JPY' in symbol else 10000)
        logger.info(f"✅ VERIFIED: SL={actual_sl_pips:.1f}p (target {Config.STATIC_SL_PIPS}), "
                   f"TP={actual_tp_pips:.1f}p (target {Config.STATIC_TP_PIPS})")
        
        spread_pips = best_spread * (100 if 'JPY' in symbol else 10000)
        volatility = signal.get('market_conditions', {}).get('volatility', 0.001)
        use_limit = self.should_use_limit_order(symbol, spread_pips, session_quality, volatility)
        
        # Create order request with FIXED TP/SL (130/60)
        if use_limit:
            if signal['direction'] == 'buy':
                limit_price = original_entry - (spread_cost * 0.5)
            else:
                limit_price = original_entry + (spread_cost * 0.5)
            
            order_request = {
                'type': 'LIMIT',
                'instrument': symbol,
                'units': str(units),
                'price': str(round(limit_price, 3 if 'JPY' in symbol else 5)),
                'timeInForce': 'GTD',
                'gtdTime': (datetime.now() + timedelta(minutes=5)).isoformat() + 'Z',
                'positionFill': 'DEFAULT',
                'stopLossOnFill': {
                    'price': str(round(signal['stop_loss'], 3 if 'JPY' in symbol else 5))
                },
                'takeProfitOnFill': {
                    'price': str(round(signal['take_profit'], 3 if 'JPY' in symbol else 5))
                }
            }
            logger.info(f"LIMIT order: Entry={limit_price:.5f} SL={signal['stop_loss']:.5f} TP={signal['take_profit']:.5f}")
        else:
            order_request = {
                'type': 'MARKET',
                'instrument': symbol,
                'units': str(units),
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT',
                'stopLossOnFill': {
                    'price': str(round(signal['stop_loss'], 3 if 'JPY' in symbol else 5))
                },
                'takeProfitOnFill': {
                    'price': str(round(signal['take_profit'], 3 if 'JPY' in symbol else 5))
                }
            }
            logger.info(f"MARKET order: SL={signal['stop_loss']:.5f} TP={signal['take_profit']:.5f}")

        logger.info(f"🎯 ORDER TO OANDA: {json.dumps(order_request, indent=2)}")
        
        result = self.client.place_order(order_request)
        
        if result and 'orderFillTransaction' in result:
            fill = result['orderFillTransaction']
            trade_id = fill.get('tradeOpened', {}).get('tradeID', fill['id'])
            actual_price = float(fill['price'])
            
            actual_slippage = abs(actual_price - original_entry)
            actual_slippage_pips = actual_slippage * (100 if 'JPY' in symbol else 10000)
            
            logger.info(f"Fill analysis: Expected={original_entry:.5f} Actual={actual_price:.5f} "
                       f"Slippage={actual_slippage_pips:.2f} pips")
            
            self.excursion_trackers[trade_id] = ExcursionTracker(
                trade_id=trade_id,
                symbol=symbol,
                direction=signal['direction'],
                entry_price=actual_price,
                stop_loss=signal['stop_loss'],
                last_check=datetime.now().isoformat()
            )
            
            risk_amount = abs(actual_price - signal['stop_loss'])
            if 'JPY' in symbol:
                risk_pips = risk_amount * 100
            else:
                risk_pips = risk_amount * 10000
            
            logger.info(f"R:R Levels for {trade_id}: Risk={risk_pips:.1f} pips")
            logger.info(f"  1.0R: {risk_pips:.1f}p | TP target: {Config.STATIC_TP_PIPS}p")
            
            if Config.USE_TRAILING_STOP:
                self.trailing_stops[trade_id] = {
                    'activated': False,
                    'entry': actual_price,
                    'stop_loss': signal['stop_loss'],
                    'direction': signal['direction']
                }
            
            entry_time_iso = datetime.now().isoformat()
            trade_record = TradeRecord(
                timestamp=entry_time_iso,
                trade_id=trade_id,
                symbol=symbol,
                direction=signal['direction'],
                strategy=signal['strategy'],
                confidence=signal['confidence'],
                entry_price=actual_price,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                position_size=abs(units),
                risk_amount=account_balance * Config.RISK_PER_TRADE * signal.get('position_multiplier', 1.0),
                market_conditions=signal.get('market_conditions', {}),
                indicators=signal.get('indicators', {}),
                entry_time=entry_time_iso,  # ✅ FIXED - Set entry time
                session=signal.get('session'),
                order_flow_bias=signal.get('market_conditions', {}).get('order_flow', 0),
                multi_tf_alignment=abs(signal.get('market_conditions', {}).get('multi_tf_bias', 0)) > 0.5
            )
            
            self.trade_records[trade_id] = trade_record
            advanced_logger.log_trade(trade_record)
            logger.info(f"✅ Trade logged to CSV: {trade_id}")
            
            logger.info(f"🎯 *** TRADE EXECUTED: {symbol} {signal['direction'].upper()} @ {actual_price} " +
                       f"Strategy: {signal['strategy']} Session: {signal.get('session')} " +
                       f"Units: {units} TP/SL: {Config.STATIC_TP_PIPS}/{Config.STATIC_SL_PIPS} pips ***")
            
            self.last_trade_time[symbol] = time.time()
            self.risk_manager.daily_trades += 1
            
            # Fix logging: mark the signal as traded in memory AND on disk
            if '_signal_record' in signal:
                signal['_signal_record'].traded = True
                signal['_signal_record'].not_traded_reason = None
                advanced_logger.update_signal_traded(signal['_signal_record'])
            
            if signal['strategy'] not in advanced_logger.strategy_performance:
                advanced_logger.strategy_performance[signal['strategy']] = {
                    'wins': 0, 'losses': 0, 'pnl': 0, 'total_trades': 0,
                    'total_win': 0, 'total_loss': 0
                }
            advanced_logger.strategy_performance[signal['strategy']]['total_trades'] += 1
            
            return True
        
        elif result and 'orderCreateTransaction' in result:
            logger.info(f"Limit order created for {symbol}, waiting for fill...")
            return True
        elif result and 'error' in result:
            # Order was rejected by OANDA - check if it's margin-related
            error_text = str(result.get('error_text', '')).lower()
            error_json = result.get('error', {})
            error_message = str(error_json).lower()

            # ✅ IMPROVED MARGIN ERROR DETECTION - Check multiple patterns
            margin_keywords = ['margin', 'insufficient', 'funds', 'balance', 'closeout']
            is_margin_error = (
                any(keyword in error_text for keyword in margin_keywords) or
                any(keyword in error_message for keyword in margin_keywords) or
                'INSUFFICIENT_MARGIN' in str(error_json) or
                'MARGIN_CLOSEOUT_REQUIRED' in str(error_json) or
                'INSUFFICIENT_LIQUIDITY' in str(error_json) or
                result.get('status_code') in [400, 403, 422]  # Expanded status codes
            )

            if is_margin_error:
                logger.warning(f"🚫 Order rejected - INSUFFICIENT MARGIN: {error_text[:200]}")
                self.log_blocked_trade(signal, symbol, account_balance, f"OANDA rejection: {error_text[:100]}")
            else:
                # Not a margin error - log to error file for debugging
                logger.warning(f"⚠️ Order rejected - OTHER ERROR: {error_text[:200]}")
                logger.warning(f"Error details: {error_json}")
                # Don't block-track non-margin errors as they won't have valid outcomes

            return False

        # ✅ FALLBACK CASE - Unexpected result format
        logger.error(f"⚠️ Unexpected order result format: {result}")
        return False
    
    def update_trailing_stops(self):
        trades = self.client.get_open_trades()
        
        for trade in trades:
            trade_id = trade['id']
            
            if trade_id not in self.trailing_stops:
                continue
            
            trailing_info = self.trailing_stops[trade_id]
            current_price = float(trade['price'])
            entry_price = trailing_info['entry']
            unrealized_pl = float(trade.get('unrealizedPL', 0))
            
            symbol = trade['instrument']
            df = self.client.get_candles(symbol, Config.TIMEFRAMES['primary'], count=20)
            if df.empty or 'atr' not in df.columns:
                continue
            
            atr = df['atr'].iloc[-1]
            
            if trailing_info['direction'] == 'buy':
                if not trailing_info['activated']:
                    if current_price >= entry_price + abs(entry_price - trailing_info['stop_loss']):
                        trailing_info['activated'] = True
                        logger.info(f"Trailing stop activated for {trade_id}")
                
                if trailing_info['activated']:
                    new_stop = current_price - (Config.TRAILING_STOP_DISTANCE * atr)
                    current_stop = float(trade.get('stopLossOrder', {}).get('price', 0))
                    
                    if new_stop > current_stop:
                        if self.client.modify_trade(trade_id, stop_loss=new_stop):
                            logger.info(f"Trailing stop updated for {trade_id}: {current_stop:.5f} -> {new_stop:.5f}")
                            
            else:
                if not trailing_info['activated']:
                    if current_price <= entry_price - abs(trailing_info['stop_loss'] - entry_price):
                        trailing_info['activated'] = True
                        logger.info(f"Trailing stop activated for {trade_id}")
                
                if trailing_info['activated']:
                    new_stop = current_price + (Config.TRAILING_STOP_DISTANCE * atr)
                    current_stop = float(trade.get('stopLossOrder', {}).get('price', float('inf')))
                    
                    if new_stop < current_stop:
                        if self.client.modify_trade(trade_id, stop_loss=new_stop):
                            logger.info(f"Trailing stop updated for {trade_id}: {current_stop:.5f} -> {new_stop:.5f}")
    
    def check_partial_profits(self):
        if not Config.USE_PARTIAL_PROFITS:
            return
        
        trades = self.client.get_open_trades()
        if not trades:
            return
        logger.info(f"Open trades: {[t['id'] for t in trades]}, Tracking: {list(self.partial_targets.keys())}")
        current_prices = self.client.get_prices([t['instrument'] for t in trades])
        
        for trade in trades:
            trade_id = trade['id']
            
            if trade_id not in self.partial_targets:
                continue
            
            targets = self.partial_targets[trade_id]
            symbol = trade['instrument']
            
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]['bid'] if targets['direction'] == 'sell' else current_prices[symbol]['ask']
            
            for tp_name in ['tp1', 'tp2', 'tp3']:
                target = targets[tp_name]
                
                if target.hit:
                    continue
                
                hit = False
                if targets['direction'] == 'buy' and current_price >= target.price:
                    hit = True
                elif targets['direction'] == 'sell' and current_price <= target.price:
                    hit = True
                
                if hit:
                    if targets['direction'] == 'sell':
                        units_to_close = str(-int(target.units))
                    else:
                        units_to_close = str(int(target.units))
                    
                    if self.client.close_trade(trade_id, units_to_close):
                        target.hit = True
                        targets['remaining_units'] -= target.units
                        
                        logger.info(f"Partial profit taken for {trade_id} at {tp_name}: " +
                                  f"{units_to_close} units @ {current_price:.5f}")
                        
                        if tp_name == 'tp1':
                            entry = self.trailing_stops.get(trade_id, {}).get('entry')
                            if entry:
                                self.client.modify_trade(trade_id, stop_loss=entry)
                                logger.info(f"Stop moved to breakeven for {trade_id}")
    
    def update_closed_trades(self):
        """Check for recently closed trades and update records"""
        try:
            url = f"{Config.OANDA_API_URL}/v3/accounts/{self.client.account_id}/trades"
            params = {'count': 50}
            response = self.client.session.get(url, params=params)
            
            if response.status_code == 200:
                all_trades = response.json()['trades']
                
                for trade in all_trades:
                    trade_id = trade['id']
                    state = trade.get('state', 'UNKNOWN')
                    
                    if state == 'CLOSED' and trade_id not in self.closed_trades_tracked:
                        self.closed_trades_tracked.add(trade_id)
                        
                        if trade_id in self.trade_records:
                            record = self.trade_records[trade_id]
                            
                            record.exit_time = trade.get('closeTime', datetime.now().isoformat())
                            record.exit_price = float(trade.get('averageClosePrice', 0))
                            record.pnl = float(trade.get('realizedPL', 0))
                            record.pnl_pips = self.calculate_pnl_pips(record.symbol, record.entry_price, record.exit_price, record.direction)
                            record.exit_reason = self._determine_exit_reason(trade)
                            record.status = "CLOSED"

                            # Record TP hit time for post-TP cooldown
                            if record.exit_reason == 'take_profit':
                                self.last_tp_time[record.symbol] = time.time()
                                logger.info(f"POST-TP COOLDOWN: {record.symbol} TP hit — "
                                           f"{Config.POST_TP_COOLDOWN_SECONDS//60}min cooldown started")
                            
                            # Calculate duration using entry_time (or timestamp as fallback)
                            if record.exit_time and record.entry_time:
                                try:
                                    start = datetime.fromisoformat(record.entry_time.replace('T', ' ').replace('Z', ''))
                                    end = datetime.fromisoformat(record.exit_time.replace('T', ' ').replace('Z', '').split('.')[0])
                                    record.duration_minutes = int((end - start).total_seconds() / 60)
                                except Exception as e:
                                    logger.warning(f"Failed to calculate duration for {trade_id}: {e}")
                                    record.duration_minutes = 0
                            
                            self.risk_manager.update_trade_stats(record.pnl, record.strategy)

                            # Persist to disk (survives restarts)
                            if self.persistent_state:
                                self.persistent_state.record_trade_result(
                                    trade_id, record.pnl, record.strategy
                                )

                            strategy = record.strategy
                            if strategy in advanced_logger.strategy_performance:
                                perf = advanced_logger.strategy_performance[strategy]
                                if record.pnl > 0:
                                    perf['wins'] += 1
                                    perf['total_win'] += record.pnl
                                else:
                                    perf['losses'] += 1
                                    perf['total_loss'] += abs(record.pnl)
                                perf['pnl'] += record.pnl
                            
                            # Update MAE/MFE from excursion tracker if available
                            if trade_id in self.excursion_trackers:
                                tracker = self.excursion_trackers[trade_id]
                                tracker.exit_price = record.exit_price
                                tracker.exit_time = record.exit_time

                                # Calculate MAE/MFE in pips
                                if 'JPY' in record.symbol:
                                    record.max_favorable = tracker.max_favorable * 100
                                    record.max_adverse = tracker.max_adverse * 100
                                else:
                                    record.max_favorable = tracker.max_favorable * 10000
                                    record.max_adverse = tracker.max_adverse * 10000

                            # Log to old format (for backward compatibility)
                            advanced_logger.log_trade(record)

                            # ✅ ALSO LOG TO UNIFIED CSV - This is the main analysis file
                            advanced_logger.log_complete_trade(record, is_blocked=False)
                            logger.info(f"✅ Closed trade logged to unified CSV: {trade_id}")
                            
                            logger.info(f"Trade {trade_id} CLOSED: {record.symbol} "
                                      f"P&L=£{record.pnl:.2f} ({record.pnl_pips:.1f} pips) "
                                      f"Exit={record.exit_price:.5f} Reason={record.exit_reason}")
                        
        except Exception as e:
            logger.error(f"Error updating closed trades: {e}")
    
    def _determine_exit_reason(self, trade: Dict) -> str:
        """Determine why a trade closed"""
        if 'stopLossOrder' in trade and trade['stopLossOrder'].get('filledTime'):
            return 'stop_loss'
        elif 'takeProfitOrder' in trade and trade['takeProfitOrder'].get('filledTime'):
            return 'take_profit'
        elif trade.get('closingTransaction', {}).get('type') == 'MARKET_ORDER':
            return 'manual_close'
        else:
            return trade.get('state', 'unknown')
    
    def track_excursions(self):
        """Track maximum price excursions for all trades (open and closed)"""
        try:
            symbols = list(set([t.symbol for t in self.excursion_trackers.values()]))
            if not symbols:
                return
                
            prices = self.client.get_prices(symbols)
            
            for trade_id, tracker in self.excursion_trackers.items():
                if tracker.symbol not in prices:
                    continue
                
                if tracker.direction == 'buy':
                    current = prices[tracker.symbol]['bid']
                else:
                    current = prices[tracker.symbol]['ask']
                
                if tracker.direction == 'buy':
                    favorable = current - tracker.entry_price
                    adverse = tracker.entry_price - current
                else:
                    favorable = tracker.entry_price - current
                    adverse = current - tracker.entry_price
                
                tracker.max_favorable = max(tracker.max_favorable, favorable)
                tracker.max_adverse = max(tracker.max_adverse, adverse)
                
                risk = abs(tracker.entry_price - tracker.stop_loss)
                if risk > 0:
                    r_multiple = favorable / risk
                    
                    if r_multiple >= 1.0 and not tracker.hit_1r:
                        tracker.hit_1r = True
                        logger.info(f"🎯 Trade {trade_id} hit 1.0R")
                    if r_multiple >= 2.0 and not tracker.hit_2r:
                        tracker.hit_2r = True
                        logger.info(f"🎯 Trade {trade_id} hit 2.0R")
                    if r_multiple >= 2.5 and not tracker.hit_2_5r:
                        tracker.hit_2_5r = True
                        logger.info(f"🎯 Trade {trade_id} hit 2.5R")
                    if r_multiple >= 3.0 and not tracker.hit_3r:
                        tracker.hit_3r = True
                        logger.info(f"🎯🎯 Trade {trade_id} hit 3.0R!")
                    if r_multiple >= 3.5 and not tracker.hit_3_5r:
                        tracker.hit_3_5r = True
                        logger.info(f"🎯🎯 Trade {trade_id} hit 3.5R!!")
                    if r_multiple >= 4.0 and not tracker.hit_4r:
                        tracker.hit_4r = True
                        logger.info(f"🎯🎯🎯 Trade {trade_id} hit 4.0R!!!")
                    if r_multiple >= 5.0 and not tracker.hit_5r:
                        tracker.hit_5r = True
                        logger.info(f"🎯🎯🎯🎯 Trade {trade_id} hit 5.0R!!!!")
                
                tracker.last_check = datetime.now().isoformat()
                
        except Exception as e:
            logger.error(f"Error tracking excursions: {e}")

    def manage_open_positions(self):
        """Active management of open positions: stepped profit lock + momentum fade exit.

        This is NOT a trailing stop. It implements two specific exit improvements:

        1. STEPPED PROFIT LOCK: At discrete profit thresholds, ratchet the stop-loss
           upward to lock in profit. The trade keeps its original 130-pip TP target,
           but if price reverses, the ratcheted stop catches it at a profit.

        2. MOMENTUM-FADE EXIT: When a trade is significantly in profit (+50p) but
           momentum indicators show the move is exhausting, close the trade early
           to capture profit before it reverses back to the original SL.
        """
        if not Config.USE_PROFIT_LOCK and not Config.USE_MOMENTUM_FADE_EXIT:
            return

        trades = self.client.get_open_trades()
        if not trades:
            return

        for trade in trades:
            trade_id = trade['id']
            symbol = trade['instrument']
            units = float(trade['currentUnits'])
            entry_price = float(trade['price'])
            direction = 'buy' if units > 0 else 'sell'

            # Get current price
            prices = self.client.get_prices([symbol])
            if symbol not in prices:
                continue

            current_price = prices[symbol]['bid'] if direction == 'buy' else prices[symbol]['ask']
            pip_mult = 100 if 'JPY' in symbol else 10000

            # Calculate current profit in pips
            if direction == 'buy':
                profit_pips = (current_price - entry_price) * pip_mult
            else:
                profit_pips = (entry_price - current_price) * pip_mult

            # Get current SL from OANDA
            current_sl = float(trade.get('stopLossOrder', {}).get('price', 0))
            if current_sl == 0:
                continue

            pip_value = 0.01 if 'JPY' in symbol else 0.0001

            # --- STEPPED PROFIT LOCK ---
            if Config.USE_PROFIT_LOCK and profit_pips > 0:
                for trigger_pips, lock_pips in sorted(Config.PROFIT_LOCK_LEVELS, reverse=True):
                    if profit_pips >= trigger_pips:
                        # Calculate the new SL to lock in profit
                        if direction == 'buy':
                            new_sl = entry_price + (lock_pips * pip_value)
                        else:
                            new_sl = entry_price - (lock_pips * pip_value)

                        # Only move SL if the new one is better (closer to profit)
                        sl_improved = (
                            (direction == 'buy' and new_sl > current_sl) or
                            (direction == 'sell' and new_sl < current_sl)
                        )
                        if sl_improved:
                            new_sl_rounded = round(new_sl, 3 if 'JPY' in symbol else 5)
                            if self.client.modify_trade(trade_id, stop_loss=new_sl_rounded):
                                logger.info(
                                    f"PROFIT LOCK: {symbol} at +{profit_pips:.0f}p — "
                                    f"SL moved from {current_sl:.5f} to {new_sl_rounded:.5f} "
                                    f"(locking +{lock_pips}p profit)"
                                )
                                current_sl = new_sl_rounded  # Update for fade check below
                        break  # Only apply highest triggered level

            # --- MOMENTUM-FADE EXIT ---
            if Config.USE_MOMENTUM_FADE_EXIT and profit_pips >= Config.MOMENTUM_FADE_MIN_PROFIT_PIPS:
                try:
                    # Get M5 data for momentum assessment
                    df = self.client.get_candles(symbol, Config.TIMEFRAMES['primary'], count=30)
                    if df.empty or len(df) < 20:
                        continue

                    # Add indicators if not present
                    if 'rsi' not in df.columns:
                        close_prices = df['close'].values.astype(float)
                        df['rsi'] = talib.RSI(close_prices, timeperiod=14)
                        macd, macd_sig, macd_hist = talib.MACD(close_prices, 12, 26, 9)
                        df['macd_hist'] = macd_hist

                    latest_rsi = df['rsi'].iloc[-1]
                    prev_rsi_5 = df['rsi'].iloc[-6]  # RSI 5 bars ago
                    macd_hist_now = df['macd_hist'].iloc[-1]
                    macd_hist_prev = df['macd_hist'].iloc[-3]

                    should_fade_exit = False
                    fade_reason = ""

                    if direction == 'buy':
                        # Long trade fading: RSI was high and is dropping, MACD hist declining
                        rsi_reversal = prev_rsi_5 - latest_rsi
                        macd_declining = macd_hist_now < macd_hist_prev
                        if rsi_reversal > Config.MOMENTUM_FADE_RSI_REVERSAL and macd_declining:
                            should_fade_exit = True
                            fade_reason = f"RSI dropped {rsi_reversal:.0f} pts ({prev_rsi_5:.0f}->{latest_rsi:.0f}), MACD hist declining"
                    else:
                        # Short trade fading: RSI was low and is rising, MACD hist rising
                        rsi_reversal = latest_rsi - prev_rsi_5
                        macd_rising = macd_hist_now > macd_hist_prev
                        if rsi_reversal > Config.MOMENTUM_FADE_RSI_REVERSAL and macd_rising:
                            should_fade_exit = True
                            fade_reason = f"RSI rose {rsi_reversal:.0f} pts ({prev_rsi_5:.0f}->{latest_rsi:.0f}), MACD hist rising"

                    if should_fade_exit:
                        logger.info(
                            f"MOMENTUM FADE EXIT: {symbol} {direction} at +{profit_pips:.0f}p — {fade_reason}"
                        )
                        # Close the full position
                        close_units = str(-int(units)) if direction == 'sell' else str(int(units))
                        if self.client.close_trade(trade_id, close_units):
                            logger.info(
                                f"CLOSED for +{profit_pips:.0f}p profit (was targeting {Config.STATIC_TP_PIPS}p TP). "
                                f"Momentum fade exit saved potential reversal."
                            )
                        else:
                            logger.warning(f"Failed to close {trade_id} on momentum fade")

                except Exception as e:
                    logger.debug(f"Momentum fade check error for {trade_id}: {e}")

    def calculate_spread_cost(self, symbol: str, prices: Dict) -> float:
        """Calculate the spread cost in price terms"""
        if symbol not in prices:
            return 0
        spread = prices[symbol]['spread']
        return spread / 2
        
    def estimate_slippage(self, symbol: str, units: int, session_quality: float) -> float:
        """Estimate likely slippage based on position size and market conditions"""
        if abs(units) < 10000:
            base_slippage = 0.2
        elif abs(units) < 50000:
            base_slippage = 0.5
        elif abs(units) < 100000:
            base_slippage = 1.0
        else:
            base_slippage = 2.0
        
        session_multiplier = 2.0 - session_quality
        slippage_pips = base_slippage * session_multiplier
        
        if 'JPY' in symbol:
            return slippage_pips * 0.01
        else:
            return slippage_pips * 0.0001
            
    def should_use_limit_order(self, symbol: str, spread_pips: float, 
                              session_quality: float, volatility: float) -> bool:
        """Determine if we should use limit order instead of market"""
        if spread_pips > 1.5:
            return True
        if session_quality < 0.6:
            return True
        if 'JPY' in symbol:
            volatility_pips = volatility * 100
        else:
            volatility_pips = volatility * 10000
        if volatility_pips < 10:
            return True
        return False

# ==================== MAIN BOT ====================
class ProfessionalForexBot:
    def __init__(self):
        self.oanda_client = OandaClient()
        self.analyzer = EnhancedMarketAnalyzer(self.oanda_client)
        self.signal_generator = EnhancedSignalGenerator(self.analyzer)
        self.risk_manager = EnhancedRiskManager()
        self.state = PersistentState()  # Survives restarts
        self.executor = EnhancedTradeExecutor(self.oanda_client, self.risk_manager, self.state)
        self.running = False
        self.performance_tracker = {
            'start_balance': 0.0,
            'current_balance': 0.0,
            'total_trades': self.state.get('total_trades', 0),
            'total_pnl': self.state.get('total_pnl', 0.0),
            'max_drawdown': 0.0,
            'peak_balance': self.state.get('all_time_peak', 0.0),
        }
        self.last_session = None

        # Restore win/loss stats from persistent state into risk manager
        for amt in self.state.get('win_amounts', []):
            self.risk_manager.win_amounts.append(amt)
            self.risk_manager.winning_trades += 1
        for amt in self.state.get('loss_amounts', []):
            self.risk_manager.loss_amounts.append(amt)
            self.risk_manager.losing_trades += 1

        # Restore strategy performance into signal generator
        sp = self.state.get('strategy_performance', {})
        if sp:
            self.signal_generator.strategy_performance = sp
            logger.info(f"Restored strategy performance: {list(sp.keys())}")
    
    def initialize(self) -> bool:
        account_info = self.oanda_client.get_account_info()
        if not account_info:
            logger.error("Failed to connect to OANDA")
            return False

        balance = float(account_info['balance'])
        self.performance_tracker['start_balance'] = balance
        self.performance_tracker['current_balance'] = balance

        # Use persisted peak if it's higher than current balance (preserves history)
        persisted_peak = self.state.get('all_time_peak', 0.0)
        self.performance_tracker['peak_balance'] = max(balance, persisted_peak)
        self.state.update_peak(balance)
        self.state.save()
        
        account_currency = account_info.get('currency', 'USD')
        Config.ACCOUNT_CURRENCY = account_currency
        
        logger.info(f"Connected to OANDA. Balance: {account_currency} {balance:.2f}")
        logger.info(f"Account Currency: {account_currency} - Position sizing adjusted")
        logger.info("=" * 80)
        logger.info("🎯 130/60 TP/SL MODE — TREND FOLLOWING")
        logger.info("=" * 80)
        logger.info(f"TP/SL: {Config.STATIC_TP_PIPS}/{Config.STATIC_SL_PIPS} pips (FIXED)")
        logger.info(f"Time Windows: {len(Config.TRADE_WINDOWS)} optimal 30-min slots")
        logger.info("Based on analysis: 70 trades, £66,362 profit, 28.6% win rate")
        logger.info(f"Risk per trade: {Config.RISK_PER_TRADE*100}% | Min R/R: {Config.MIN_RISK_REWARD}:1")
        logger.info(f"Max positions: {Config.MAX_POSITIONS} | Max daily trades: {Config.MAX_DAILY_TRADES}")
        logger.info("=" * 80)
        
        return True
    
    def run(self):
        if not self.initialize():
            return
        
        self.running = True
        logger.info("🎯 BOT STARTED — 130/60 TP/SL")
        
        try:
            cycle_count = 0
            while self.running:
                session, session_quality = self.analyzer.get_current_session()
                
                if self.last_session != session:
                    logger.info(f"Trading session changed: {session} (quality: {session_quality:.1f})")
                    self.last_session = session
                
                self._trading_cycle()
                
                if cycle_count % 3 == 0:
                    self.executor.update_closed_trades()
                    self.executor.track_excursions()
                    self.executor.manage_open_positions()  # Profit lock + momentum fade
                    # Update blocked trades to check for virtual TP/SL hits
                    self.executor.blocked_trade_tracker.update_blocked_trades()

                if cycle_count % 10 == 0:
                    self._save_performance_snapshot()
                    self._monitor_open_positions()
                
                if cycle_count % 120 == 0:
                    advanced_logger.save_daily_analysis()
                    self._validate_system_health()
                
                cycle_count += 1
                
                if session_quality >= 0.9:
                    time.sleep(30)
                elif session_quality >= 0.7:
                    time.sleep(60)
                elif session_quality >= 0.5:
                    time.sleep(120)
                else:
                    time.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _trading_cycle(self):
        account_info = self.oanda_client.get_account_info()
        if not account_info:
            return

        current_balance = float(account_info['balance'])
        self.performance_tracker['current_balance'] = current_balance

        # Update persisted peak and drawdown
        dd_from_peak = self.state.update_peak(current_balance)
        if current_balance > self.performance_tracker['peak_balance']:
            self.performance_tracker['peak_balance'] = current_balance

        drawdown = (self.performance_tracker['peak_balance'] - current_balance) / self.performance_tracker['peak_balance']
        if drawdown > self.performance_tracker['max_drawdown']:
            self.performance_tracker['max_drawdown'] = drawdown

        # HARD DRAWDOWN CIRCUIT BREAKER — uses persisted all-time peak
        if self.state.is_drawdown_breached(current_balance):
            peak = self.state.get('all_time_peak', 0)
            logger.warning(
                f"CIRCUIT BREAKER: Balance {current_balance:.2f} is {dd_from_peak:.1%} below "
                f"all-time peak {peak:.2f} (limit: {Config.MAX_DRAWDOWN_PCT:.0%}). NO NEW TRADES."
            )
            return

        session, session_quality = self.analyzer.get_current_session()
        
        can_trade, reason = self.risk_manager.can_trade(current_balance, session_quality)
        if not can_trade:
            logger.info(f"Not trading: {reason}")
            return
        
        regime = self.analyzer.get_market_regime()
        logger.debug(f"Market regime: {regime}")
        
        top_pairs = self.analyzer.rank_pairs_by_opportunity()
        
        if not top_pairs:
            logger.debug("No suitable pairs identified")
            return
        
        signals_generated = 0
        trades_executed = 0
        
        for symbol in top_pairs:
            try:
                if trades_executed >= 1:
                    break
                
                signal = self.signal_generator.generate_signals(symbol)
                
                if signal:
                    signals_generated += 1
                    
                    logger.info(f"🎯 *** SIGNAL: {symbol} {signal['strategy']} " +
                              f"{signal['direction']} @ {signal['confidence']:.2f} " +
                              f"R/R: {signal.get('risk_reward', 0):.1f} ***")
                    
                    if signal['confidence'] >= Config.CONFIDENCE_THRESHOLD:
                        if self.executor.execute_signal(signal, symbol, current_balance):
                            trades_executed += 1
                            self.performance_tracker['total_trades'] += 1
                        
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        if signals_generated > 0 or trades_executed > 0:
            logger.info(f"Cycle complete: {signals_generated} signals, {trades_executed} trades executed")
            self._log_performance()
    
    def _monitor_open_positions(self):
        trades = self.oanda_client.get_open_trades()
        
        if not trades:
            return
        
        logger.info(f"=== Open Positions ({len(trades)}) ===")
        
        total_unrealized = 0
        for trade in trades:
            symbol = trade['instrument']
            units = float(trade['currentUnits'])
            entry = float(trade['price'])
            unrealized_pl = float(trade.get('unrealizedPL', 0))
            total_unrealized += unrealized_pl
            
            prices = self.oanda_client.get_prices([symbol])
            if symbol in prices:
                current = prices[symbol]['bid'] if units > 0 else prices[symbol]['ask']
                pips = (current - entry) * (100 if 'JPY' in symbol else 10000)
                if units < 0:
                    pips = -pips
                
                currency_symbol = '£' if Config.ACCOUNT_CURRENCY == 'GBP' else '$'
                logger.info(f"  {symbol}: {'LONG' if units > 0 else 'SHORT'} " +
                          f"Entry={entry:.5f} Current={current:.5f} " +
                          f"Pips={pips:.1f} P&L={currency_symbol}{unrealized_pl:.2f}")
        
        currency_symbol = '£' if Config.ACCOUNT_CURRENCY == 'GBP' else '$'
        logger.info(f"  Total Unrealized P&L: {currency_symbol}{total_unrealized:.2f}")

        # Report blocked trade tracking status
        blocked_stats = self.executor.blocked_trade_tracker.get_statistics()
        if blocked_stats['total_blocked'] > 0:
            logger.info(f"=== Blocked Trades Tracking ===")
            logger.info(f"  Active Virtual Trades: {blocked_stats['active']}")
            logger.info(f"  Completed Analysis: {blocked_stats['completed']}")
            if blocked_stats['would_have_filled'] > 0:
                logger.info(f"  Virtual Win Rate: {blocked_stats['virtual_win_rate']:.1%}")
                logger.info(f"  Total Virtual Pips: {blocked_stats['total_virtual_pips']:.1f}")

    def _validate_system_health(self):
        for strategy, perf in self.signal_generator.strategy_performance.items():
            if perf.get('total_trades', 0) >= 20:
                win_rate = perf['wins'] / perf['total_trades']
                if win_rate < 0.3:
                    logger.warning(f"Strategy {strategy} underperforming: {win_rate:.1%} win rate")
        
        if self.performance_tracker['max_drawdown'] > 0.15:
            logger.warning(f"High drawdown detected: {self.performance_tracker['max_drawdown']:.1%}")
        
        expectancy = self.risk_manager.calculate_expectancy()
        if expectancy < 0 and self.risk_manager.daily_trades > 10:
            logger.warning(f"Negative expectancy: {Config.ACCOUNT_CURRENCY}{expectancy:.2f} per trade")
    
    def _save_performance_snapshot(self):
        metrics = self.performance_tracker

        # Use PERSISTED stats (survive restarts) instead of in-memory only
        p_wins = self.state.get('winning_trades', 0)
        p_losses = self.state.get('losing_trades', 0)
        p_total = p_wins + p_losses
        win_rate = p_wins / p_total if p_total > 0 else 0

        win_amts = self.state.get('win_amounts', [])
        loss_amts = self.state.get('loss_amounts', [])
        total_wins_val = sum(win_amts) if win_amts else 0
        total_losses_val = sum(loss_amts) if loss_amts else 0
        profit_factor = total_wins_val / total_losses_val if total_losses_val > 0 else 0

        # Save state periodically
        self.state.update_peak(metrics['current_balance'])
        self.state.save()
        
        open_positions = len(self.oanda_client.get_open_positions())
        expectancy = self.risk_manager.calculate_expectancy()
        
        best_session = None
        worst_session = None
        if advanced_logger.session_performance:
            sessions_sorted = sorted(advanced_logger.session_performance.items(), 
                                   key=lambda x: x[1]['pnl'])
            if sessions_sorted:
                worst_session = sessions_sorted[0][0]
                best_session = sessions_sorted[-1][0]
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            balance=metrics['current_balance'],
            equity=metrics['current_balance'],
            daily_pnl=self.risk_manager.daily_pnl,
            daily_return_pct=(self.risk_manager.daily_pnl / metrics['start_balance']) * 100 if metrics['start_balance'] > 0 else 0,
            total_pnl=self.state.get('total_pnl', 0.0),
            total_return_pct=((metrics['current_balance'] - metrics['start_balance']) / metrics['start_balance']) * 100 if metrics['start_balance'] > 0 else 0,
            open_positions=open_positions,
            total_trades=self.state.get('total_trades', 0),
            winning_trades=p_wins,
            losing_trades=p_losses,
            win_rate=win_rate,
            avg_win=np.mean(win_amts) if win_amts else 0,
            avg_loss=np.mean(loss_amts) if loss_amts else 0,
            profit_factor=profit_factor,
            sharpe_ratio=self.risk_manager.calculate_sharpe_ratio(),
            max_drawdown=metrics['max_drawdown'],
            current_drawdown=(metrics['peak_balance'] - metrics['current_balance']) / metrics['peak_balance'] if metrics['peak_balance'] > 0 else 0,
            kelly_fraction=self.risk_manager._calculate_kelly_fraction(),
            expectancy=self.state.expectancy,
            best_session=best_session,
            worst_session=worst_session
        )
        
        advanced_logger.log_performance(snapshot)
    
    def _log_performance(self):
        m = self.performance_tracker
        positions = len(self.oanda_client.get_open_positions())
        expectancy = self.risk_manager.calculate_expectancy()
        currency_symbol = '£' if Config.ACCOUNT_CURRENCY == 'GBP' else '$'
        
        logger.info(f"Balance: {currency_symbol}{m['current_balance']:.2f} | " +
                   f"Daily: {currency_symbol}{self.risk_manager.daily_pnl:.2f} | " +
                   f"Trades: {self.risk_manager.daily_trades} | " +
                   f"Positions: {positions}/{Config.MAX_POSITIONS} | " +
                   f"Expectancy: {currency_symbol}{expectancy:.2f}")

    def run_margin_aware_analysis(self):
        """
        Run comprehensive margin-aware portfolio simulation.
        Analyzes performance with different exit strategies and margin constraints.
        """
        logger.info("=" * 80)
        logger.info("🔬 RUNNING MARGIN-AWARE PORTFOLIO ANALYSIS")
        logger.info("=" * 80)

        try:
            # Get account balance for simulation
            account_info = self.oanda_client.get_account_info()
            if account_info:
                starting_balance = float(account_info['balance'])
            else:
                starting_balance = self.performance_tracker['start_balance']

            # Initialize simulator
            simulator = MarginAwarePortfolioSimulator(
                starting_balance=starting_balance,
                max_leverage=30.0  # Adjust based on your account
            )

            # Load trade data
            trades_csv = f"{Config.ANALYSIS_DIR}/trades_log.csv"
            blocked_initial_csv = f"{Config.ANALYSIS_DIR}/blocked_trades_initial.csv"
            blocked_outcomes_csv = f"{Config.ANALYSIS_DIR}/blocked_trades_outcomes.csv"

            logger.info("Loading trade data from CSVs...")
            trades_data = simulator.load_trades_from_csv(
                trades_csv, blocked_initial_csv, blocked_outcomes_csv
            )

            if not trades_data['executed'] and not trades_data['blocked']:
                logger.warning("No trade data available for analysis")
                return

            # Run strategy comparison
            logger.info("Comparing exit strategies...")
            comparison_results = simulator.compare_strategies(trades_data)

            # Generate report
            report_file = f"{Config.ANALYSIS_DIR}/margin_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_text = simulator.generate_report(comparison_results, report_file)

            # Print summary
            logger.info("\n" + report_text)
            logger.info("=" * 80)
            logger.info(f"📊 Full analysis saved to: {report_file}")
            logger.info("=" * 80)

            return comparison_results

        except Exception as e:
            logger.error(f"Failed to run margin-aware analysis: {e}", exc_info=True)
            return None

    def shutdown(self):
        self.running = False

        self._save_performance_snapshot()
        self.state.save()  # Final state save

        analysis_file = advanced_logger.save_daily_analysis()

        m = self.performance_tracker
        total_pnl = self.state.get('total_pnl', 0.0)
        total_trades = self.state.get('total_trades', 0)
        currency_symbol = '£' if Config.ACCOUNT_CURRENCY == 'GBP' else '$'

        logger.info("=" * 80)
        logger.info("BOT — SHUTDOWN SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Start Balance:    {currency_symbol}{m['start_balance']:.2f}")
        logger.info(f"Final Balance:    {currency_symbol}{m['current_balance']:.2f}")
        logger.info(f"Session P&L:      {currency_symbol}{m['current_balance'] - m['start_balance']:.2f}")
        logger.info(f"Lifetime P&L:     {currency_symbol}{total_pnl:.2f}")
        logger.info(f"Lifetime Trades:  {total_trades}")
        logger.info(f"All-Time Peak:    {currency_symbol}{self.state.get('all_time_peak', 0):.2f}")
        logger.info(f"Max Drawdown:     {m['max_drawdown']:.2%}")

        wr = self.state.win_rate
        if wr > 0:
            logger.info(f"Win Rate:         {wr:.1%} ({self.state.get('winning_trades',0)}W / {self.state.get('losing_trades',0)}L)")

        logger.info(f"Expectancy:       {currency_symbol}{self.state.expectancy:.2f}")
        logger.info(f"Analysis saved:   {analysis_file}")

        # Report blocked trade statistics
        blocked_stats = self.executor.blocked_trade_tracker.get_statistics()
        if blocked_stats['total_blocked'] > 0:
            logger.info("=" * 80)
            logger.info("🚫 BLOCKED TRADES ANALYSIS")
            logger.info("=" * 80)
            logger.info(f"Total Blocked:         {blocked_stats['total_blocked']}")
            logger.info(f"Completed Analysis:    {blocked_stats['completed']}")
            logger.info(f"Still Active:          {blocked_stats['active']}")
            if blocked_stats['would_have_filled'] > 0:
                logger.info(f"Would Have Filled:     {blocked_stats['would_have_filled']}")
                logger.info(f"Would Have Won:        {blocked_stats['would_have_won']}")
                logger.info(f"Would Have Lost:       {blocked_stats['would_have_lost']}")
                logger.info(f"Virtual Win Rate:      {blocked_stats['virtual_win_rate']:.1%}")
                logger.info(f"Total Virtual Pips:    {blocked_stats['total_virtual_pips']:.1f}")
                logger.info(f"Avg Pips/Trade:        {blocked_stats['avg_pips_per_trade']:.1f}")
                logger.info("---")
                logger.info(f"📊 Check CSV files in {Config.ANALYSIS_DIR}:")
                logger.info(f"   - blocked_trades_initial.csv (all blocked attempts)")
                logger.info(f"   - blocked_trades_outcomes.csv (complete virtual outcomes)")

        logger.info("=" * 80)
        logger.info("🎯 BOT shutdown complete")

# ==================== ENTRY POINT ====================
def main():
    bot = ProfessionalForexBot()
    
    import signal
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        bot.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    bot.run()

if __name__ == "__main__":
    main()
