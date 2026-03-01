# CryptoBot V1 — Architecture Documentation

## System Overview

CryptoBot is a **Dual-Core XGBoost derivatives trading system** operating as a fixed-size, rigid-target bot for trading long and short on Binance Futures.

### The Dual-Core Strategy (V1)

Two XGBoost models predict fixed-distance targets within a 12-hour horizon:

- **Bull Model (`y_bull`)**: Predicts if a LONG position will hit a +2.0% taking profit before a -1.0% stop loss.
- **Bear Model (`y_bear`)**: Predicts if a SHORT position will hit a -2.0% taking profit before a +1.0% stop loss.

**Signal Logic:**

| Signal | Condition | Meaning |
|--------|-----------|---------|
| **LONG** | `Bull_P > threshold` | High conviction long |
| **SHORT** | `Bear_P > threshold` | High conviction short |
| **FLAT** | Everything else | No edge |

### Position Sizing & Exits (Fixed Logic)

```
Alloc Limit = 95% of Available Wallet Balance
Sizing = Fixed 95% (All-In)

Hard TP             = 2.0% from Entry
Hard SL             = 1.0% from Entry

Horizon = 12 hourly candles (max hold time)
Fees = 0.08% round-trip (Binance Futures taker + slippage)
```

---

## Architecture

```
crypto_bot/
├── main.py                      ← CLI entry point (argparse)
│
│
├── core/                        ← THE ENGINE
│   ├── config.py                ← Environment + constants (1x leverage hard-coded)
│   ├── data_manager.py          ← Historical CSV + live Binance feed
│   ├── execution.py             ← Live execution w/ fixed 2% TP and 1% SL
│   ├── logging_setup.py         ← Rotating file handler (5MB × 3)
│   └── notification.py          ← Telegram alerts (optional)
│
├── strategies/                  ← THE BRAIN
│   ├── feature_engineering.py   ← SINGLE SOURCE OF TRUTH for features (22 features)
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

## Feature Set (22 Features Legacy)

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
| 22 | Candlesticks | `cdl_body_ratio` | OHLC | Body/range ratio (conviction) |

---

## Model Performance (Legacy V1)

Performance was extremely poor out-of-time (OOT), with negative PnL across all pairs due to the rigidity of the 2% TP and 1% SL rules. The lack of a chop filter led to significant bleed in ranging markets.

| Pair | Bull AUC | Bear AUC | Time AUC |
|------|----------|----------|----------|
| BTC  | 0.623    | 0.561    | **0.734** |
| ETH  | 0.538    | 0.557    | **0.713** |
| SOL  | 0.522    | 0.536    | **0.695** |
| BNB  | 0.587    | 0.569    | **0.686** |
| XRP  | **0.622** | 0.555   | **0.715** |

All data is real (no synthetic), sourced from Binance Spot API (OHLCV), Binance Futures API (funding rate), and alternative.me (Fear & Greed index).

---

## Findings from the V1 Exercise

1. **Fixed Targets Don't Work:** Using a hard 2% Take Profit and 1% Stop Loss across all pairs is deeply flawed. 1% on BTC is a major move, while 1% on XRP is just standard 1-hour noise. The bot was getting stopped out constantly on volatile pairs and not reaching TP on stable pairs.
2. **"All-In" Sizing is Dangerous:** Betting 95% of the wallet on every single cross of the 0.50 threshold meant massive volatility in portfolio swings without respecting the machine learning model's confidence level.
3. **Missing "Chop" Filter:** The bot would bleed chips exactly sideways because it was forced to be in a Long or Short at all times if a threshold was crossed. We urgently needed a "Do Not Trade" time/chop classifier.

*These findings directly spurred the development of V2 (Triple Barrier Targets & Time Filter) and V3 (Dynamic Trading Exits).*

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
