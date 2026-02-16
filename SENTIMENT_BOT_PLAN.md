# Crypto Sentiment Bot - Project Plan

## Project Constraints (MUST FOLLOW)

- **Location**: India - All transactions in INR
- **Budget**: Total project cost < 100 USD (including deployed capital)
- **Capital Protection**: MUST prioritize capital preservation
- **Slippage & Fees**: Must be factored into calculations
- **Exchange**: Binance (preferred for low fees)

---

## Current System Understanding

### Architecture Overview
The existing bot is a modular Python trading system with:

```
CryptoBot/
├── core/               # Core utilities
│   ├── data_fetcher.py    # Fetches OHLCV from exchanges via CCXT
│   ├── indicators.py     # RSI, Bollinger Bands calculations
│   └── strategies.py     # Backtesting.py strategy definitions
├── backtesting/        # Historical testing
│   ├── backtest.py      # Single pair backtest
│   └── batch_runner.py  # Multi-pair batch testing
├── live/               # Real-time trading
│   ├── trader.py       # Main trading logic (analyze + execute)
│   ├── account_manager.py  # Track positions & balance
│   └── bot_runner.py   # Orchestrates the trading loop
├── configs/            # YAML configs per bot (BTC, ETH, SOL)
├── data/               # Historical CSV data
├── config.py           # Legacy config file
└── requirements.txt    # Dependencies
```

### Current Trading Logic
1. **Data**: Fetch OHLCV (1h candles) via CCXT
2. **Indicators**: Calculate RSI(14) + Bollinger Bands(20,2)
3. **Signal Generation**:
   - BUY: Price < Lower BB AND RSI < 30
   - SELL: Price > Upper BB AND RSI > 70
4. **Execution**: Paper trading or live Binance orders

### Execution Modes
- **Backtest**: Historical simulation
- **Paper**: Simulated trades with virtual money
- **Live**: Real Binance API orders

### IMPORTANT: Previous Strategy DISCARDED
**Bollinger Band + RSI Strategy**: Tested but discarded
- **Reason**: Too few trades, not profitable
- **Result**: -29.82% net return, only 3 trades, 0% win rate on BTC
- **Lesson**: Technical indicators alone are insufficient; need sentiment-driven approach

---

## Planned Adaptation: Sentiment-Driven Trading

### New Architecture
```
CryptoBot/
├── core/
│   ├── data_fetcher.py    # Existing - OHLCV data
│   ├── indicators.py      # Existing - Technical indicators
│   ├── sentiment.py       # NEW - Twitter sentiment fetch & analysis
│   └── signals.py        # NEW - Combine sentiment + optionally technicals
├── bots/
│   └── sentiment_bot.py   # NEW - Main autonomous bot
├── wallet/
│   └── wallet_manager.py # NEW - Manage dedicated wallet funds
├── configs/
│   └── sentiment_*.yaml  # NEW - Bot configurations
└── ...
```

### Sentiment Analysis Pipeline (Twitter)

1. **Data Collection**
   - Fetch tweets using Twitter API (X API v2)
   - Keywords: `$BTC`, `$ETH`, `$SOL`, `bitcoin`, `ethereum`, `solana`, etc.
   - Configurable time windows (last 1h, 6h, 24h)

2. **Sentiment Processing**
   - Use LLM (OpenAI GPT) or local model (HuggingFace)
   - Classify: Bullish / Bearish / Neutral
   - Aggregate score: Weighted average or majority vote

3. **Signal Generation**
   - BULLISH threshold: >60% positive → BUY signal
   - BEARISH threshold: >60% negative → SELL signal
   - Else: HOLD

### Autonomous Wallet System

1. **Wallet Setup**
   - Dedicated Binance sub-account or main account
   - Initial capital: User-defined (e.g., $1000)
   - Trade size: Configurable % per trade (e.g., 95%)

2. **Risk Management**
   - Max daily loss limit
   - Stop-loss (optional, based on technicals)
   - Position cooldown between trades
   - Maximum open positions

3. **24/7 Operation**
   - Scheduled runs (e.g., every 15 min, 1h)
   - Docker container for cloud deployment
   - Health checks & auto-restart
   - Logging & alerts (Telegram/Discord)

### Data Flow
```
Twitter API → Tweet Fetch → LLM Sentiment → Score
                                          ↓
                              Signal (BUY/SELL/HOLD)
                                          ↓
Binance API ← Execute Trade ← Wallet Manager ← Current Price (CCXT)
```

### Key Components to Build

| Component | Purpose |
|-----------|---------|
| `core/sentiment.py` | Twitter API client + sentiment analysis (✓ Built) |
| `backtesting/sentiment_backtest.py` | Backtest sentiment strategy (✓ Built) |
| `core/signals.py` | Combine sentiment into tradeable signals |
| `bots/sentiment_bot.py` | Main loop: fetch → analyze → trade |
| `wallet/wallet_manager.py` | Track PnL, positions, capital |
| `configs/sentiment_*.yaml` | Per-pair configs (thresholds, keywords) |

---

## Cost Structure & Budget (India)

### Estimated Costs (Must stay under 100 USD total)

| Item | Cost (USD) | Notes |
|------|-----------|-------|
| Deployed Capital | ~$50-80 | Main trading capital |
| OpenAI API (sentiment) | ~$5-10/month | GPT-3.5 for sentiment analysis |
| Twitter API (if needed) | ~$0-100/month | Free tier available |
| Cloud/Hosting | ~$0-10/month | Can run locally or free tier |
| Exchange Fees | ~0.1% per trade | Binance spot fees |

### INR Handling
- Convert INR to USDT via Indian exchanges (WazirX, CoinDCX, ZebPay)
- All calculations in USDT for simplicity
- Track INR equivalent for reporting

### Fee Calculation Formula
```
Total Cost = Trade Value * (1 + Maker/Taker Fee + Slippage)
- Binance Spot Fee: 0.1% (maker/taker)
- Estimated Slippage: 0.05-0.1% for liquid pairs
```

---

## Risk Management & Capital Protection

### Capital Preservation Rules
1. **Max Daily Loss**: Stop trading if daily loss > 5%
2. **Max Drawdown**: Auto-stop if portfolio drawdown > 15%
3. **Position Size**: Never risk more than 10% per trade
4. **Stop Loss**: 2-3% hard stop on every trade
5. **Cooldown**: 1 hour minimum between trades

### Emergency Controls
- Manual kill switch
- Auto-pause on high volatility
- Position size limits
- Trading hours restriction (avoid late-night volatility)

### Configuration Structure (YAML)
```yaml
bot:
  name: "BTC Sentiment Bot"
  pair: "BTC/USDT"
  exchange: "binance"

sentiment:
  source: "twitter"
  keywords: ["$BTC", "bitcoin", "#BTC"]
  time_window: "1h"
  min_tweets: 10
  bullish_threshold: 0.6
  bearish_threshold: 0.6
  cooldown_minutes: 60

execution:
  paper_trading: false
  allocated_capital: 1000
  trade_size_pct: 0.95

risk:
  max_daily_loss_pct: 5
  max_drawdown_pct: 15
  max_position_pct: 10
  stop_loss_pct: 3
  cooldown_minutes: 60
  max_positions: 1
```

---

## Backtesting Adaptation

### Challenge
- Binance API limits to ~1000 candles per fetch
- Historical Twitter data requires paid API access

### Solution
- **Live Sentiment Backtest**: For each hour in backtest period, fetch live Twitter sentiment at that moment
- Single pair test (BTC/USDT) first
- Use OpenAI for sentiment classification

### Data Available
- 5 pairs: BTC, ETH, SOL, BNB, XRP (USDT pairs)
- Each has ~1000 candles (1h timeframe) = ~41 days

### Execution
```
for each hour in backtest_data:
    1. Fetch live tweets about the pair
    2. Send to OpenAI for sentiment analysis
    3. Generate BUY/SELL/HOLD signal
    4. Execute simulated trade if signal != HOLD
    5. Track metrics
```

### Metrics to Output (matching current format)
- Start Equity / Final Equity
- Net Return %
- Buy & Hold Return %
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown %
- # Trades
- Win Rate %

---

## Session Notes & Updates

*(This section is updated regularly as the project progresses)*

### 2026-02-15: Initial Planning
- Analyzed existing Bollinger+RSI strategy - DISCARDED (too few trades, -29.82% return)
- Planned sentiment-based approach using Twitter + OpenAI
- Created `core/sentiment.py` with Twitter fetch & sentiment analysis
- Created `backtesting/sentiment_backtest.py` for backtesting
- **Twitter backtest failed** - Demo mode (random sentiment) gave poor results (-35% return)
- **Switched to NewsAPI** for backtesting (user has API access)

### Constraints Registered
- India-based: INR transactions, use USDT for trading
- Budget: < $100 total (including capital)
- Capital protection: Priority #1
- Include slippage & fees in all calculations
- Slippage: 0.1%, Binance Fee: 0.1%

### 2026-02-15: Derivatives Strategy Implemented
- Created `derivatives_backtest/` module with real Binance API integration
- **Real Data Sources**:
  - Spot OHLCV: Real from Binance
  - Funding Rate: Real from `/fapi/v1/fundingRate`
  - Long/Short Ratio: Real from `/futures/data/globalLongShortAccountRatio`
  - Open Interest: Real from `/futures/data/openInterestHist`
- **Data cached locally** in `derivatives_backtest/data/`

### 2026-02-15: Initial Batch Results (Small Samples)
- First test on 1,000 candles showed promising results for SOL/XRP but with very low trade counts.
- Strategy was too restrictive (AND logic on 4+ factors).

### 2026-02-16: Data Fixes & Strategy Refinement
- **Forward Pagination**: Fixed `data_fetcher.py` to use forward pagination. Now funding rates cover the full 14-month range, and L/S / OI cover the maximum available 30-day window.
- **Streamlined Signals**: Removed redundant hard z-score filters in `signal_generator.py`. The strategy now trades purely on `sentiment_score > threshold` + `Price > EMA-50`.
- **Better Normalization**: Increased the rolling min-max window for the sentiment score from 100 to 500 candles to filter out noise.
- **Rebalanced Weights**: Increased focus on Momentum (ROC 25%, Vol 20%) while keeping contrarian derivatives at 15% each.
- **OI Corrected**: High Open Interest during flat regimes is now correctly treated as accumulation (bullish).

### 2026-02-16: Batch Backtest Results (Full 10K Candles)
*Test Period: Dec 2024 — Feb 2026 (Mostly Bearish)*

| Pair | Test Return% | B&H% | Sharpe | Trades | Win Rate% | Max DD% |
|------|-------------|-------------|--------|-----------|----------|---------|
| BTC | -54.48% | -38.98% | -5.91 | 52 | 21.2% | -54.48% |
| ETH | -74.17% | -51.83% | -7.19 | 61 | 19.7% | -74.52% |
| SOL | -63.26% | -57.51% | -5.40 | 48 | 18.8% | -63.80% |
| BNB | -53.15% | -49.24% | -5.85 | 52 | 26.9% | -54.04% |
| XRP | -81.11% | -40.87% | -7.96 | 57 | 8.8% | -81.17% |
| **AVG** | **-65.23%** | **-47.69%** | **-6.46** | **54** | **19.1%** | **-65.60%** |

**Observation**: While trade frequency is now healthy (~4 trades/month), the strategy struggled in the sustained downtrend of late 2025. Capital preservation was the previous strength; the current "momentum" tilt needs a stronger bearish filter or a switch to shorting.

**Conclusion from 10K Test**: The current configuration is too aggressive in capturing local rallies within a macro downtrend. Needs better bearish exclusion logic.

### 2026-02-16: Strategy Refinement & Data Expansion
- **Cleanup**: Removed Architecture 1 (BB+RSI) files and consolidated into `derivatives_backtest/`.
- **Enhanced Data**: Integrated historical Fear & Greed Index (daily) and paginated derivatives data.
- **Improved Metrics**: Added exit fees and slippage (0.1% + 0.05%) to backtests for realism.
- **Composite Score**: Shifted to a 6-component weighted score (Funding, L/S, OI, Volume, F&G, Momentum).

#### Data Gap Discovery
- **Observation**: During 10,000 candle (14-month) backtesting, we found that:
  - **L/S Ratio & Open Interest**: Historical data is limited to ~30 days (744 records) via the public Binance API.
  - **Funding Rate**: API pagination logic was fetching oldest data first; needs alignment with OHLCV range.
  - **Result**: Significant portions of the backtest were using forward-filled default values, leading to low trade frequency and unreliable results in the early 90% of the period.

### 2026-02-16: XGBoost ML Research Results
**Goal**: Determine if non-linear ML modeling can solve the "-65% drawdown" problem by filtering out bad trades.

#### ML Model Performance (XGBoost Classifier — 70/30 Time-Series Split)
| Pair | AUC | Precision | Recall | F1 | Train Pos% | Test Pos% |
|------|-----|-----------|--------|-----|-----------|----------|
| BTC | 0.620 | 0.323 | 0.525 | 0.400 | 27.6% | 25.8% |
| ETH | 0.640 | 0.421 | 0.375 | 0.397 | 36.5% | 29.7% |
| SOL | **0.684** | **0.446** | **0.659** | **0.532** | 39.6% | 32.3% |
| BNB | 0.653 | 0.356 | 0.682 | 0.468 | 33.1% | 27.9% |
| XRP | 0.579 | 0.341 | 0.645 | 0.446 | 35.7% | 29.8% |

#### ML-Filtered Backtest Comparison
| Pair | Det. Return | ML-Filtered Return | B&H Return | Det. Trades | ML Trades | ML Max DD |
|------|------------|-------------------|------------|------------|-----------|----------|
| BTC | -74.73% | **-5.08%** | -38.98% | 58 | 12 | -7.95% |
| ETH | -77.95% | **0.00%** | -51.83% | 53 | 0 | 0.00% |
| SOL | -84.35% | **-5.21%** | -57.51% | 56 | 5 | -5.43% |
| BNB | -80.51% | **-9.15%** | -49.24% | 51 | 11 | -9.17% |
| XRP | -91.41% | **-49.87%** | -40.87% | 68 | 31 | -50.77% |
| **AVG** | **-81.79%** | **-13.86%** | **-47.69%** | 54 | 12 | -14.66% |

**Key Finding**: ML filter improved average return by **+67.93%** vs deterministic. The XGBoost model successfully pruned the majority of "falling knife" trades.

#### Top Predictive Features (Aggregated)
1. **ROC (Price Momentum)** — 11.06% importance — Strongest single predictor
2. **ROC Z-Score** — 7.90%
3. **Fear & Greed Z-Score** — 5.53%
4. **F&G Normalized** — 5.20%
5. **200-EMA Distance** — 4.71%
6. **ATR %** — 4.18%
7. **Funding Rate** — 3.73%

**Insight**: L/S Ratio and Open Interest had near-zero importance (data only covers 30 days). The model relies primarily on **momentum + sentiment + trend context** to make predictions. SOL had the best AUC (0.684).

#### Revised Goals (Post-ML Research)
1. **Adopt ML-Filtered Strategy**: Use XGBoost as a mandatory trade gate (P > 0.55) alongside sentiment threshold.
2. **Retrain Periodically**: Implement walk-forward optimization to prevent model staleness.
3. **Address XRP Leak**: XRP's ML filter still lost -49.87% — investigate why (possibly too many false positives).
4. **Drop L/S and OI from ML features**: Near-zero importance due to 30-day data limit. Replace with on-chain metrics if available.
5. **Test Higher Probability Thresholds**: Try P > 0.60 and P > 0.65 to further reduce bad trades.

### 2026-02-16: Proxy Futures Backtest (Long/Short 1x via Spot Data)
**Goal**: Test if adding short legs (via 200-EMA + ROC trend signals) improves returns over long-only.

**Setup**: Spot Close treated as Futures Mark Price. Funding applied every 8h. 3% intra-candle stop loss. 0.1% fee + 0.05% slippage per side.

| Pair | Long-Only | Long+Short | Buy & Hold | Short PnL | Shorts Won? |
|------|-----------|------------|------------|-----------|-------------|
| BTC | -79.42% | -91.94% | -29.95% | -$11,358 | ❌ |
| ETH | -63.48% | -84.96% | -44.81% | -$8,099 | ❌ |
| SOL | -76.77% | -85.45% | -60.62% | **+$4,700** | ✅ |
| BNB | -68.07% | -88.83% | -14.29% | -$18,069 | ❌ |
| XRP | -49.45% | -86.86% | -40.42% | -$32,444 | ❌ |
| **AVG** | **-67.44%** | **-87.61%** | **-38.02%** | **-$13,054** | ❌ |

**Verdict**: ❌ Short legs **bled money** on 4/5 pairs (avg -$13K). Only SOL shorts were profitable. The 200-EMA + ROC signal generates too many whipsaw trades (~800/pair) in a ranging/bearish market. Stop losses were rarely triggered (1-6% of exits).

**Key Insight**: The problem is **signal quality**, not direction. Both long and short sides lose because the trend filter flips too frequently on 1h candles.

#### Data Integrity Audit
| Source | Coverage | Notes |
|--------|----------|-------|
| Spot OHLCV | ✅ 100% real | Full 10K candles from Binance |
| Fear & Greed | ✅ 97% real | ~288/10K rows default (50) |
| Funding Rate | ⚠️ ~84% real | 16% default (0.0001), varies by pair |
| L/S Ratio | ❌ ~4% real | Only last 30 days — rest forward-filled |
| Open Interest | ❌ ~5% real | Only last 30 days — rest forward-filled |

### 2026-02-16: Dual-Core ML Backtest (Bull + Bear XGBoost)
**Goal**: Train separate Bull and Bear XGBoost models with high-conviction probability gates to create "Sniper" entries.

**Setup**: Bull model predicts +1% in 24h, Bear model predicts -2% in 24h. Entry thresholds: Long requires Bull_P > 0.70 AND Bear_P < 0.30. Short requires Bear_P > 0.70 AND Bull_P < 0.30. Test period is the last 30% of data (Oct 2025 — Feb 2026).

#### Model Training Summary
| Pair | Bull AUC | Bear AUC | Bull F1 | Bear F1 |
|------|----------|----------|---------|---------|
| BTC | 0.632 | 0.692 | 0.428 | 0.397 |
| ETH | 0.646 | 0.701 | 0.427 | 0.489 |
| SOL | 0.696 | **0.714** | **0.541** | 0.482 |
| BNB | 0.672 | **0.718** | 0.494 | 0.456 |
| XRP | 0.602 | 0.671 | 0.496 | 0.458 |

#### 🔥 Dual-Core Backtest Results
| Pair | Raw (L+S) | ML Shield | **Dual-Core** | B&H | Dual Trades | Dual WR% |
|------|-----------|-----------|--------------|------|-------------|----------|
| BTC | -99.14% | -35.90% | **+69.82%** | -38.13% | 220 | 59.1% |
| ETH | -99.74% | -2.72% | **+570.22%** | -50.50% | 227 | 59.9% |
| SOL | -99.68% | -34.38% | **+387.08%** | -56.11% | 274 | 63.1% |
| BNB | -98.85% | -49.13% | **+161.75%** | -47.88% | 282 | 58.2% |
| XRP | -99.82% | -61.84% | **+165.72%** | -38.96% | 249 | 61.0% |
| **AVG** | **-99.44%** | **-36.79%** | **+270.92%** | **-46.32%** | 250 | 60.3% |

#### Direction Breakdown (All Pairs Profitable on Both Sides)
| Pair | Long PnL | Short PnL |
|------|----------|-----------|
| BTC | +$97,267 | +$16,363 |
| ETH | +$89,343 | +$562,211 |
| SOL | +$407,048 | +$83,713 |
| BNB | +$118,528 | +$127,198 |
| XRP | +$173,078 | +$59,198 |
| **Total Short PnL** | | **+$848,683** |

**Verdict**: ✅ Dual-Core **crushed all benchmarks**. +271% avg return vs -99% raw and -37% shield. Shorts ADDED value on all 5 pairs. Win rate ~60%. The high-conviction probability gates (0.70/0.30) successfully reduced trades from ~800 to ~250 and eliminated whipsaw.

⚠️ **Caution**: These results warrant careful scrutiny for potential data leakage or overfitting. Next step should be walk-forward validation.

---

## DEPLOYMENT & RESEARCH STATUS

### Current Status
*Deployment is paused. Four experiments have been run:*
1. **Deterministic Sentiment** → -65% avg (too many falling knife entries)
2. **ML Long-Only Shield** → -37% avg (improved but still negative)
3. **Proxy Futures (Raw L+S)** → -99% avg (worst — shorts amplified bad signals)
4. **🔥 Dual-Core ML (Bull+Bear)** → **+271% avg** (breakthrough — both sides profitable)

### Recommended Path Forward
1. **Validate Dual-Core**: Run walk-forward validation to confirm results aren't overfit.
2. **Paper Trade**: Deploy Dual-Core on live paper account for 2-4 weeks.
3. **Model Retraining**: Implement walk-forward optimization (retrain monthly).
4. **Risk Management**: Add drawdown-based position sizing (reduce size during DD).
5. **Live Deployment**: If paper trading confirms, deploy with minimal capital (1-2% portfolio).

---

## Technical Context
### Data Sources
- **Spot OHLCV**: Binance Spot API (paginated to 10K candles).
- **Funding Rate**: Binance Futures API (`/fapi/v1/fundingRate`) — ~84% real coverage.
- **L/S Ratio**: Binance Futures API (30-day limit) — ❌ mostly forward-filled.
- **Open Interest**: Binance Futures API (30-day limit) — ❌ mostly forward-filled.
- **Sentiment**: Alternative.Me Fear & Greed Index — ✅ ~97% real.

### Feature Pipeline
1. **Fetch**: `DerivativesDataFetcher` retrieves and merges all 5 sources.
2. **Engineer**: `FeatureEngineer` computes Z-scores, ATR, EMA, ROC, ADX, and ML features.
3. **Score**: Composite `sentiment_score` (rebalanced weights).
4. **Train**: `train_models.py` trains Bull + Bear XGBoost models per pair.
5. **Backtest**: `dual_core_test.py` runs Dual-Core ML-gated backtest.

### Project Files
```
derivatives_backtest/
├── core/
│   ├── data_fetcher.py        # Fetches and caches all data sources
│   ├── feature_engineering.py # Base + ML features, sentiment score, both targets
│   └── signal_generator.py   # BacktestSignalExecutor (entry/exit logic)
├── backtester.py              # Single-pair backtester
├── batch_runner.py            # Multi-pair batch runner with param tuning
├── ml_research.py             # XGBoost training, feature importance, ML-filtered backtest
├── train_models.py            # Dual-Core model trainer (Bull + Bear per pair)
├── dual_core_test.py          # Dual-Core ML backtest (L+S with probability gates)
├── proxy_futures_test.py      # Long/Short 1x simulation via spot data (raw)
├── models/                    # Saved XGBoost models (JSON) + metadata
├── data/                      # Cached CSVs (10K candles per pair)
└── results/                   # Backtest outputs and ML plots
```


