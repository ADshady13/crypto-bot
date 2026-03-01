# CryptoBot V3 — Architecture Documentation

## System Overview

CryptoBot is a **Triple-Core XGBoost derivatives trading system** utilizing intelligent position sizing, dynamic Chandelier Exits (trailing stops), and specialized ML models as high-conviction entry gates. It trades long and short on Binance Futures.

### The Triple-Core Strategy (V3)

Three XGBoost models collaborate to predict barrier touches within a 6-hour horizon:

- **Bull Model (`y_bull`)**: Predicts if a LONG position will hit a 1.5×ATR upward threshold before a 1.0×ATR downward threshold.
- **Bear Model (`y_bear`)**: Predicts if a SHORT position will hit a 1.5×ATR downward threshold before a 1.0×ATR upward threshold.
- **Time/Chop Model (`y_time`)**: Predicts if the market is dead/ranging — no barrier hit within 6 hours.

**Signal Logic:**

| Signal | Condition | Meaning |
|--------|-----------|---------|
| **LONG** | `Bull_P > threshold` AND `Time_P < chop_thresh` | High conviction long + market is moving |
| **SHORT** | `Bear_P > threshold` AND `Time_P < chop_thresh` | High conviction short + market is moving |
| **FLAT** | Everything else | No edge or choppy market — stay out |

### Position Sizing & Exit Math (Dynamic Logic)

```
Alloc Limit = 10,000 INR (MAX_ALLOCATION_PER_BOT)
Base Sizing = 50% of Alloc Limit
Size Scled  = Scales from Base up to 100% based on Distance(Probability, Threshold)

Trailing Activation = 1.0 × ATR in profit
Trailing Pullback   = 0.5 × ATR behind Peak Profit (MFE)
Hard SL             = 1.0 × ATR from Entry

Horizon = 12 hourly candles (max hold time)
Fees = 0.08% round-trip (Binance Futures taker + slippage)
```

---

## Architecture

```
crypto_bot/
├── main.py                      ← CLI entry point (argparse)
│
├── .agent/                      ← AGENTIC CAPABILITIES
│   └── skills/                  ← Localized AI skills
│       ├── market-sentiment     ← Crypto RSS news & sentiment
│       ├── trading-strategist   ← Strategy generator
│       ├── stock-market-pro     ← Yahoo Finance analytics
│       └── alphaear-*           ← Advanced financial trackers
│
├── core/                        ← THE ENGINE
│   ├── config.py                ← Environment + constants (1x leverage hard-coded)
│   ├── data_manager.py          ← Historical CSV + live Binance feed
│   ├── execution.py             ← Live execution w/ trailing stops & Kelly sizing
│   ├── logging_setup.py         ← Rotating file handler (5MB × 3)
│   └── notification.py          ← Telegram alerts (optional)
│
├── strategies/                  ← THE BRAIN
│   ├── feature_engineering.py   ← SINGLE SOURCE OF TRUTH for features (34 features)
│   └── inference.py             ← Hot-reloading ModelLoader
│
├── scripts/                     ← UTILITIES
│   └── download_clean_data.py   ← Fetch 20K candles + funding + F&G from APIs
│
├── jobs/                        ← CRON TASKS
│   └── retrain.py               ← Champion vs Challenger pipeline
│
├── models/                      ← ARTIFACTS (gitignored)
│   ├── xgb_bull_{PAIR}.json     ← Active Bull model
│   ├── xgb_bear_{PAIR}.json     ← Active Bear model
│   ├── xgb_time_{PAIR}.json     ← Active Time/Chop model
│   └── *_meta.json              ← Training metadata
│
├── data/                        ← CACHED DATA (gitignored)
│   ├── {PAIR}_20k.csv           ← 20,000 hourly candles (Nov 2023 → Mar 2026)
│   └── ...
│
├── logs/                        ← RUNTIME LOGS
├── bot.service                  ← Systemd service file
├── deploy.sh                    ← DigitalOcean deployment script
├── requirements.txt             ← Pinned production dependencies
├── .env.example                 ← Template for environment variables
└── .gitignore                   ← Security rules
```

---

## Data Flow

```
                    ┌─────────────┐
                    │  Binance    │
                    │  Spot API   │
                    └──────┬──────┘
                           │ OHLCV (hourly)
                    ┌──────▼──────┐     ┌───────────────┐
                    │  Binance    │     │  alternative   │
                    │  Futures    │     │  .me API       │
                    └──────┬──────┘     └──────┬────────┘
                           │ Funding Rate      │ Fear & Greed
                    ┌──────▼──────────────────▼──────┐
                    │       DataManager              │
                    │  (Historical CSV + Live Feed)  │
                    └──────────────┬─────────────────┘
                                   │ Raw DataFrame (7 cols)
                    ┌──────────────▼─────────────────┐
                    │     FeatureEngineer.transform() │
                    │  34 features: ROC, BB, RSI,    │
                    │  F&G, Funding, EMA, ADX, ATR,  │
                    │  Volume, Sentiment, Candles     │
                    └──────────────┬─────────────────┘
                                   │ Feature Matrix
                    ┌──────────────▼─────────────────┐
                    │     ModelLoader.predict()       │
                    │  ┌────────┐ ┌────────┐ ┌────┐ │
                    │  │Bull XGB│ │Bear XGB│ │Time│ │
                    │  └───┬────┘ └───┬────┘ └──┬─┘ │
                    │      │         │         │    │
                    │  bull_p     bear_p    time_p  │
                    └──────┬─────────┬────────┬─────┘
                           │         │        │
                    ┌──────▼─────────▼────────▼─────┐
                    │   ExecutionEngine.evaluate()    │
                    │  Bull>T & Time<C → LONG         │
                    │  Bear>T & Time<C → SHORT        │
                    │  Otherwise → FLAT                │
                    └──────────────┬─────────────────┘
                                   │ Signal
                    ┌──────────────▼─────────────────┐
                    │  ExecutionEngine.execute()      │
                    │  Paper: simulated fill           │
                    │  Live: Binance Futures API       │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │  Logger + Telegram Notifier     │
                    └────────────────────────────────┘
```

---

## Feature Set (34 Features)

| # | Category | Feature | Source | Description |
|---|----------|---------|--------|-------------|
| 1 | Momentum | `roc` | Close | ROC(12h) — rate of change |
| 2 | | `roc_4h` | Close | ROC(4h) |
| 3 | | `roc_24h` | Close | ROC(24h) |
| 4 | | `roc_72h` | Close | ROC(72h) |
| 5 | | `roc_zscore` | Close | Z-score of ROC(12h) |
| 6 | Sentiment | `fear_greed_value` | API | Raw F&G index (0-100) |
| 7 | | `fg_zscore` | API | F&G z-score (720h window) |
| 8 | Derivatives | `funding_rate` | Binance | Raw funding rate |
| 9 | | `funding_zscore` | Binance | Funding z-score |
| 10 | | `funding_delta_4h` | Binance | 4h funding rate change |
| 11 | | `funding_delta_24h` | Binance | 24h funding rate change |
| 12 | Trend | `ema_200_dist` | Close | % distance from 200-EMA |
| 13 | | `adx` | H/L/C | ADX(14) trend strength |
| 14 | Volatility | `atr_pct` | H/L/C | ATR(14) as % of price |
| 15 | | `atr_pct_zscore` | H/L/C | Z-score of atr_pct |
| 16 | Volume | `volume_zscore` | Volume | Volume z-score (30h) |
| 17 | | `volume_delta_4h` | Volume | 4h volume % change |
| 18 | Composite | `sentiment_score` | Derived | Composite score [0,1] |
| 19 | BB (20h) | `bb_width_20` | Close | 20h band width (squeeze/expansion) |
| 20 | | `bb_pct_20` | Close | 20h %B (position in bands) |
| 21 | | `bb_zscore_20` | Close | Z-score of bb_width_20 |
| 22 | BB (50h) | `bb_width_50` | Close | 50h band width |
| 23 | | `bb_pct_50` | Close | 50h %B |
| 24 | | `bb_zscore_50` | Close | Z-score of bb_width_50 |
| 25 | RSI | `rsi_14` | Close | Standard RSI(14h) |
| 26 | | `rsi_7` | Close | Fast RSI(7h) |
| 27 | | `rsi_delta` | Close | RSI(7) - RSI(21) divergence |
| 28 | | `rsi_zscore` | Close | Z-score of RSI(14) |
| 29 | Candles | `cdl_doji` | OHLC | Doji pattern (indecision) |
| 30 | | `cdl_hammer` | OHLC | Hammer (bullish reversal) |
| 31 | | `cdl_invhammer` | OHLC | Inverted hammer |
| 32 | | `cdl_engulf_bull` | OHLC | Bullish engulfing |
| 33 | | `cdl_engulf_bear` | OHLC | Bearish engulfing |
| 34 | | `cdl_body_ratio` | OHLC | Body/range ratio (conviction) |

---

## Model Performance (500-Trial HP Tuning)

OOT Period: Nov 2025 → Mar 2026 (2,300 hourly candles per pair)

| Pair | Bull AUC | Bear AUC | Time AUC |
|------|----------|----------|----------|
| BTC  | 0.623    | 0.561    | **0.734** |
| ETH  | 0.538    | 0.557    | **0.713** |
| SOL  | 0.522    | 0.536    | **0.695** |
| BNB  | 0.587    | 0.569    | **0.686** |
| XRP  | **0.622** | 0.555   | **0.715** |

## Backtest Results (V3 Dynamic Engine)

Realistic simulation leveraging **Trailing Exits** (1.0 ATR activation, 0.5 ATR pullback) and **Intelligent Position Sizing** (50%-100% of a 10K INR max allocation based on confidence). Includes 0.04% taker fees/side + 0.03% slippage/side.

*OOT Period: Nov 2025 → Mar 2026*

| Pair | Best Thresholds | Net PnL | Max DD | Win Rate | PF | Avg Hold |
|------|----------------|---------|--------|----------|-----|----------|
| **ETH** | B≥0.50 R≥0.60 T<0.50 | **₹+1,320** (+13.20%) | 6.8% | 51.5% | 1.16 | ~2h |
| **XRP** | B≥0.55 R≥0.50 T<0.55 | **₹+1,216** (+12.17%) | 12.1% | 53.2% | 1.14 | ~1h |
| **SOL** | B≥0.50 R≥0.45 T<0.45 | **₹+789** (+7.89%) | 6.0% | 46.5% | 1.20 | ~2h |
| **BNB** | B≥0.55 R≥0.50 T<0.40 | **₹+137** (+1.38%) | 10.9% | 49.8% | 1.03 | ~3h |
| BTC | B≥0.50 R≥0.45 T<0.35 | ₹-12 (-0.12%) | 8.2% | 51.3% | 1.00 | ~3h |

---
## Risk Controls

### Dynamic Safeguards

| Control | Value | Purpose |
|---------|-------|---------|
| **Max Leverage** | **1x** | Prevents liquidation entirely. |
| **Intelligent Sizing** | **5,000 to 10,000 INR** | Position size scales linearly based on probability score `P` above threshold. |
| **Trailing Stop** | **0.5 ATR Pullback** | Locks in profits aggressively during strong trends. Prevents a winning trade from becoming a loser. |
| **Chop Filter** | **Time model** | Vetoes entries if the market is paralyzed sideways. |

---

## Indian Tax Compliance & Fees

CryptoBot V3 operates under a highly optimized fee mathematical framework for Indian citizens on Binance Futures constraint sets:

1. **1% TDS Exemption:** Under Section 194S of the Indian Income Tax Act, the 1% TDS on Virtual Digital Assets (VDAs) applies only to spot transfers. Crypto derivative futures contracts are strictly classified as **derivative instruments and are EXEMPT from the 1% TDS.**
2. **Tax Classification:** Profits are treated as **Speculative Business Income** (Section 43(5)) and taxed according to standard individual slab rates.
3. **Deductibles:** Because this is business income, the bot's execution fees, API costs, and server costs act as legal business deductibles to lower net tax burden.

*Bot Calculation Rate Table:*
| Fee Component | Rate | Side |
|----------|------|------|
| Exchange Taker Fee | 0.04% | Entry & Exit |
| Modeled Slippage | 0.03% | Entry & Exit |
| **Total Round-Trip Friction** | **0.14%** | Total Cost of Trading |

---

## Training Data

| Pair | Rows | Timeframe | Date Range | File |
|------|------|-----------|------------|------|
| BTCUSDT | 20,000 | 1h | Nov 2023 → Mar 2026 | `BTCUSDT_20k.csv` |
| ETHUSDT | 20,000 | 1h | Nov 2023 → Mar 2026 | `ETHUSDT_20k.csv` |
| SOLUSDT | 20,000 | 1h | Nov 2023 → Mar 2026 | `SOLUSDT_20k.csv` |
| BNBUSDT | 20,000 | 1h | Nov 2023 → Mar 2026 | `BNBUSDT_20k.csv` |
| XRPUSDT | 20,000 | 1h | Nov 2023 → Mar 2026 | `XRPUSDT_20k.csv` |

All data is real (no synthetic), sourced from Binance Spot API (OHLCV), Binance Futures API (funding rate), and alternative.me (Fear & Greed index).

---

## Agentic Capabilities (Skills)

CryptoBot incorporates local AI agent integration via `.agent/skills/` directory for macro overriding and advanced heuristics. These skills operate alongside the deterministic execution engine:

- `market-sentiment`: Aggregates real-time news from RSS to gauge the macro temperature and protect against sudden flash crashes.
- `trading-strategist`: Contextualizes current market conditions via LLM synthesis.
- `alphaear-news` & `alphaear-sentiment`: Fetch and analyze pure financial news to identify sentiment regimes (Bull Market vs Bear Market contexts).
- `stock-market-pro`: Hooks into external Yahoo Finance feeds to detect cross-market correlations (e.g., S&P 500 crashes bleeding into Crypto).

---

## CLI Reference

```bash
# Paper trading (default)
python main.py --mode paper --pair BTCUSDT

# Live trading
python main.py --mode live --pair SOLUSDT

# Custom capital for paper trading
python main.py --mode paper --pair ETHUSDT --capital 50000

# Retrain a single pair
python main.py --retrain --pair BTCUSDT

# Retrain all pairs
python main.py --retrain --all

# Help
python main.py --help
```
