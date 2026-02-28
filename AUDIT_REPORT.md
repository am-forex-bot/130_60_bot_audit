# FORENSIC AUDIT REPORT: Forex Bot 130/60

**Date:** 2026-02-28
**Bot:** forex_bot_130_60.py (4,134 lines)
**Data Period:** 2025-10-28 to 2026-02-28 (~123 days)
**Files Analyzed:** forex_bot_130_60.py, performance_log.csv (12,351 rows), signals_log.csv (30,211 rows), 5 daily_analysis JSON files

---

## EXECUTIVE SUMMARY

This bot has been running for 4 months on an OANDA practice account denominated in GBP. It uses a 130-pip take-profit / 60-pip stop-loss trend-following strategy across 12 currency pairs.

**Bottom line: The account is down -3.99% overall, but this masks a catastrophic 59.4% drawdown from peak-to-trough. The bot has severe architectural bugs that render all internal performance metrics non-functional, it has been restarted 31 times fragmenting all statistics, and for the last ~3 weeks it has been essentially dormant with 5 stale open positions and zero new trades.**

| Metric | Value |
|---|---|
| Starting Balance | £66,072.08 |
| Current Balance | £63,433.78 |
| Net P&L | **-£2,638.30 (-3.99%)** |
| All-Time High | £70,417.92 (Nov 5, 2025) |
| All-Time Low | £28,572.37 (Jan 2, 2026) |
| True Peak-to-Trough Drawdown | **-59.4% (-£41,846)** |
| Bot Restarts | 31 times |
| Estimated Total Trades | ~3,200 |
| Internal Win/Loss Tracking | **Completely broken (always 0)** |
| Current Open Positions | 5 |
| Last New Trade | ~Feb 27 |

---

## 1. STRATEGY ARCHITECTURE

### 1.1 Core Strategy: 130/60 TP/SL
- **Take Profit:** 130 pips (fixed, hardcoded)
- **Stop Loss:** 60 pips (fixed, hardcoded)
- **Risk/Reward:** 2.17:1
- This requires a **win rate of only ~31.6% to break even** (before costs)
- Backtest expectation: 70 trades, £66,362 profit, 28.6% win rate

### 1.2 Active Signal Strategies (3)
| Strategy | Signal Share | Description |
|---|---|---|
| trend_following | 89.4% | MACD crossover + momentum with MTF alignment |
| momentum_pullback | 9.4% | Pullback into trend continuation |
| volatility_breakout | 1.1% | Bollinger Band breakout with ATR expansion |

Two strategies were removed (commented out): `mean_reversion` ("structurally broken - shorts into bullish MTF") and `order_flow` ("fired 13 times in 3 months").

### 1.3 Entry Filter Stack (Very Aggressive)
Signals must pass ALL of these simultaneously:
1. ATR between 5-100 pips
2. Spread acceptable for session quality
3. Session quality > 0.5
4. Multi-timeframe bias > |0.5| (weighted across M1, M5, M15, H1, H4)
5. Hurst exponent > 0.52 (market must be trending)
6. Strategy-specific conditions (MACD cross, momentum, BB break)
7. Confidence >= 0.60
8. Risk/reward >= 2.0
9. No news blackout
10. No existing position in same symbol
11. Max 8 positions
12. Correlation check passes
13. Margin pre-check passes (1.5x safety buffer)
14. 5-minute cooldown per symbol

### 1.4 Disabled Features
- `USE_TIME_WINDOWS = False` (14 optimal 30-min windows identified but not used)
- `USE_KELLY_CRITERION = False` ("noisy with small samples")
- `USE_TRAILING_STOP = False` ("broken and counterproductive")
- `USE_PARTIAL_PROFITS = False` ("DISABLED until fixed")

### 1.5 Risk Parameters
- Risk per trade: 1% of account balance
- Max positions: 8
- Max daily trades: 30
- Max margin usage: 60%
- Leverage: 50:1
- Account currency: GBP

---

## 2. PERFORMANCE ANALYSIS

### 2.1 Balance Trajectory

```
£70,418 (peak, Nov 5)
  |\
  | \
  |  \______ £48,764
  |           \
  |            \____ £28,572 (bottom, Jan 2)
  |                    /
  |                   / (sharp recovery)
  |             ____/
  |            / £60,997 (late Jan)
  |           /      \___/ oscillation
  |__________/                        £63,434 (now)
  Oct     Nov     Dec     Jan      Feb
```

### 2.2 Monthly Breakdown

| Month | Open | Close | Low | High | Max Trades | Max DD | Change |
|---|---|---|---|---|---|---|---|
| 2025-10 | £66,072 | £64,614 | £60,565 | £66,072 | 11 | 4.9% | -£1,458 |
| 2025-11 | £64,614 | £41,193 | £41,193 | £70,418 | 54 | 22.8% | **-£23,421** |
| 2025-12 | £41,193 | £29,924 | £29,897 | £41,620 | 1,118 | 28.2% | **-£11,269** |
| 2026-01 | £29,924 | £54,422 | £28,572 | £62,559 | 1,137 | 30.3% | **+£24,498** |
| 2026-02 | £54,422 | £63,434 | £54,422 | £66,092 | 154 | 9.4% | +£9,012 |

### 2.3 Critical Phases

**Phase of Destruction (Nov 5 - Jan 2): -£41,846 (-59.4%)**
- 58 days of near-continuous losses
- Balance dropped from £70,418 to £28,572
- Over 1,000 trades in December alone
- The bot was overtrading during a losing streak with no circuit breaker

**Phase of Recovery (Jan 14 - Jan 28): +£26,592 (+77%)**
- One single 14-day phase generated massive gains
- 615 trades in this window
- This one phase single-handedly rescued the account from near-destruction

**Current Phase (Feb 11+): Effectively dormant**
- Bot has been restarted 13 times in 17 days
- Only 1-3 trades per restart (quickly restarts lose context)
- Balance stuck at ~£63,434 with 5 open positions

---

## 3. CRITICAL BUGS AND ISSUES

### 3.1 BUG: Win/Loss Tracking Completely Non-Functional
**Severity: CRITICAL**

Across all 12,351 performance log rows: `winning_trades=0`, `losing_trades=0`, `win_rate=0`, `avg_win=0`, `avg_loss=0`, `profit_factor=0`, `sharpe_ratio=0`, `expectancy=0`.

**Root cause:** The performance metrics are stored in `EnhancedRiskManager` instance memory. The `update_trade_stats()` method (line 3079) only gets called when `update_closed_trades()` detects a closed trade AND the trade exists in `self.trade_records` (also in memory). Every restart wipes both. With 31 restarts in 123 days, statistics never accumulate.

**Impact:** The bot literally has no idea whether it is winning or losing. All adaptive features (strategy disabling at 25% win rate, Kelly criterion, strategy performance tiers) are non-functional because they depend on data that is always zero.

### 3.2 BUG: Signals Log Shows 0% Execution Rate (Misleading)
**Severity: HIGH**

All 30,211 signals show `traded=False` with no `not_traded_reason`. This is a **logging bug**, not a trading bug:
- `log_signal()` (line 1193) writes to CSV with `traded=False` immediately
- `execute_signal()` (line 3391) later sets `signal['_signal_record'].traded = True` in memory
- But the CSV row was already written and is **never retroactively updated**

So the signals CSV is useless for determining actual execution rate. Trades ARE happening (the balance proves it), they're just not reflected in this file.

### 3.3 BUG: daily_pnl and daily_return_pct Always Zero
**Severity: MEDIUM**

The `_save_performance_snapshot()` method (line 3943) calculates daily P&L from `self.risk_manager.daily_pnl`, which resets to 0 on every restart AND on every new calendar day via `reset_daily_stats()`. Since the bot restarts so frequently, this is always 0.

### 3.4 BUG: Total P&L Resets on Restart
**Severity: HIGH**

`total_pnl` is calculated as `current_balance - start_balance` (line 3975), but `start_balance` is set fresh to the current OANDA balance on every restart. So `total_pnl` only reflects gains/losses since the last restart, not lifetime. The daily JSON files showing `total_return: 0.0` confirm this.

### 3.5 ISSUE: 31 Restarts in 123 Days
**Severity: HIGH**

The bot has been restarted approximately every 4 days on average (more frequently in February: 13 times in 17 days). Each restart:
- Wipes all in-memory trade records
- Resets win/loss counters
- Resets daily P&L tracking
- Orphans any trade_records needed for closed-trade detection
- Fragments all performance statistics

### 3.6 ISSUE: Hurst Filter May Be Too Restrictive
**Severity: MEDIUM**

The Hurst exponent threshold of 0.52 requires the market to be demonstrably trending. Combined with the MTF bias > |0.5| requirement, this creates a very narrow window where trades can fire. The signals log shows the bot generates thousands of signals that pass all other filters but may be getting blocked at execution by margin/position constraints rather than signal quality.

---

## 4. SIGNALS ANALYSIS

### 4.1 Signal Volume
- 30,211 signals over 122 days = **248 signals/day average**
- 90.4% of signals fire within 2 minutes of the previous one (signal spam)
- Over 55% are part of rapid bursts (>10 for same symbol in 30 minutes)
- The bot re-evaluates and re-logs the same signal on every cycle

### 4.2 Signal Distribution
| Metric | Value |
|---|---|
| Direction split | 50.04% sell / 49.96% buy (nearly perfect balance) |
| Top symbols | USD_JPY (28.9%), GBP_USD (23.5%), EUR_USD (14.2%) |
| Confidence range | 0.685 - 0.950 (93% in 0.85-0.95 band) |
| Sessions | Tokyo 42.9%, London-NY overlap 36.6%, NY 12.8% |

### 4.3 Risk/Reward Distribution
Only 3 unique R:R values exist (all hardcoded from the fixed SL/TP):
| R:R | Share | SL/TP |
|---|---|---|
| 2.17 | 41.0% | 60-pip SL / 130-pip TP (the "130/60" config) |
| 3.21 | 37.8% | 28-pip SL / 90-pip TP |
| 4.82 | 21.2% | 28-pip SL / 135-pip TP |

**Note:** The code hardcodes 130/60 only, but the signals show two other SL/TP tiers. The 28-pip SL / 90-135-pip TP values come from JPY pairs where the pip calculation produces different absolute distances. This is correct behavior due to the `0.01` vs `0.0001` pip multiplier for JPY pairs.

### 4.4 Frozen Indicators
- `regime` is "neutral" for 99.99% of signals (30,208/30,211)
- `order_flow` is 0.0 for every single signal
- These suggest the market regime detection and order flow calculations may be non-functional or too narrowly calibrated

---

## 5. DAILY JSON FILES (Feb 24-28, 2026)

All 5 daily analysis files show the same pattern:
- Balance declining: £65,473 -> £64,889 -> £64,893 -> £63,462 -> £63,434
- 0 trades recorded per day
- 5-6 open positions
- All metrics zero
- Recommendations always: "Win rate below 40%", "Profit factor below 1.5"

The bot is alive and logging hourly but completely dormant in terms of new trade execution.

---

## 6. ARCHITECTURAL CONCERNS

### 6.1 No State Persistence
The bot stores ALL operational state in memory (trade records, win/loss stats, daily P&L, strategy performance). Nothing persists to disk except the CSV logs. Every restart is a clean slate. For a bot that restarts this frequently, this is fatal to performance tracking.

### 6.2 No Circuit Breaker for Drawdowns
The `should_pause_trading()` method (line 2713) checks if equity is below the 20-period MA by 5%, but the equity_history is in-memory and resets on restart. During the -59.4% drawdown, the bot kept trading through the entire collapse.

### 6.3 Overtrading in Bad Conditions
December 2025 saw 1,118 trades in a single restart phase while the account lost -£11,269. The 30 daily trade limit exists but doesn't prevent large cumulative losses over weeks.

### 6.4 Signal Spam
Generating 248 signals/day that are 90%+ redundant wastes API calls and obscures the log. The same signal is logged every 60-90 seconds for the same pair in the same direction.

### 6.5 Disconnected Performance Metrics
The bot has extensive performance tracking infrastructure (PerformanceSnapshot, Sharpe ratio, Kelly criterion, strategy tiers) but none of it works because the underlying data never populates. The infrastructure is decorative.

---

## 7. WHAT HAPPENED SINCE LAST EDIT (~Feb 18)

The bot file was last modified on Feb 18, 2026. Since then:

1. **10+ restarts** in 10 days (Feb 18 → Feb 28)
2. **~10 total trades** across all restart cycles (1-3 per cycle)
3. Balance went from ~£62,835 to £63,434 (roughly flat, +£600)
4. **5 positions remain open** and have not closed
5. The bot generates signals continuously but barely executes
6. The last balance change was ~Feb 27 (the last 24+ hours are completely flat)

The frequent restarts suggest either infrastructure instability or manual intervention. Each restart orphans the previous trade records, so the bot can't track the outcomes of its own trades.

---

## 8. RECOMMENDATIONS

### Critical Fixes
1. **Persist trade records to disk** - Store `trade_records`, win/loss stats, and `start_balance` in a JSON/SQLite file that survives restarts
2. **Fix the signals CSV logging** - Update the CSV row after execution, or write execution status to a separate file
3. **Implement a hard drawdown circuit breaker** - e.g., halt all new trades if account is down >15% from a persisted high-water mark
4. **Reduce restart frequency** - Each restart destroys operational context; stabilize the deployment

### Performance Improvements
5. **Deduplicate signal logging** - Only log a signal once per symbol/direction/timeframe, not every 60 seconds
6. **Investigate the frozen indicators** - order_flow always 0.0 and regime always "neutral" suggests broken or overly conservative calculations
7. **Re-evaluate the Hurst filter** - May be too restrictive; consider lowering to 0.50 or making it a confidence modifier rather than a hard gate
8. **Enable time windows** - The backtest identified 14 profitable 30-minute windows; this feature is built but disabled
9. **Investigate the November collapse** - What caused -£23,421 in a single month? Was the strategy fundamentally wrong for those market conditions, or was there a bug?

### Monitoring
10. **Add balance high-water mark persistence** - Track the true all-time peak and drawdown across restarts
11. **Add a daily email/alert summary** - The bot has no way to notify you of problems
12. **Track actual vs. expected win rate** - The 130/60 strategy needs ~31.6% wins to break even; you need to know if you're hitting that

---

*Report generated by forensic audit of all bot files, performance data, signals data, and daily analysis JSONs.*
