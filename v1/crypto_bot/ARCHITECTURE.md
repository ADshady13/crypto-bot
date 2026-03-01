# CryptoBot — Architecture Documentation

## System Overview

CryptoBot is a **Dual-Core XGBoost derivatives trading system** that uses two specialized ML models (Bull & Bear) as high-conviction entry gates for long and short trades.

### The Dual-Core Strategy

Instead of one model deciding "buy or sell," two models collaborate:

- **Bull Model (`xgb_bull`)**: Predicts probability of a ≥1% price increase in the next 24h
- **Bear Model (`xgb_bear`)**: Predicts probability of a ≥2% price decrease in the next 24h

**Entry Logic (Probability Gates):**

| Signal | Condition | Meaning |
|--------|-----------|---------|
| **LONG** | `Bull_P > 0.70` AND `Bear_P < 0.30` | High confidence bullish |
| **SHORT** | `Bear_P > 0.70` AND `Bull_P < 0.30` | High confidence bearish |
| **FLAT** | Everything else | No edge detected — stay out |

This dual-gate approach drastically reduces trade frequency (fewer whipsaws) and only enters trades with high conviction from *both* models.

### Validated Performance

Walk-Forward Validation (17 expanding windows, 90-day initial train, 30-day steps):

| Metric | Result |
|--------|--------|
| Average OOS Win Rate | **61.2%** |
| Average Max Drawdown | **-19.7%** |
| Profit Factor | **2.62** |
| Reverse Polarity (stress test) | **PASS** — all 5 pairs |

---

## Architecture

```
crypto_bot/
├── main.py                      ← CLI entry point (argparse)
│
├── core/                        ← THE ENGINE
│   ├── config.py                ← Environment + constants (1x leverage hard-coded)
│   ├── data_manager.py          ← Historical CSV + live Binance feed
│   ├── execution.py             ← Order safety, SL logic, paper/live modes
│   ├── logging_setup.py         ← Rotating file handler (5MB × 3)
│   └── notification.py          ← Telegram alerts (optional)
│
├── strategies/                  ← THE BRAIN
│   ├── feature_engineering.py   ← SINGLE SOURCE OF TRUTH for features
│   └── inference.py             ← Hot-reloading ModelLoader
│
├── jobs/                        ← CRON TASKS
│   └── retrain.py               ← Champion vs Challenger pipeline
│
├── models/                      ← ARTIFACTS (gitignored)
│   ├── xgb_bull_BTC.json        ← Active Bull model
│   ├── xgb_bear_BTC.json        ← Active Bear model
│   └── ...
│
├── data/                        ← CACHED DATA (gitignored)
│   ├── BTCUSDT_data.csv         ← Historical 15K candles
│   └── ...
│
├── research/                    ← ARCHIVE (old backtest scripts)
│
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
                                   │ Raw DataFrame
                    ┌──────────────▼─────────────────┐
                    │     FeatureEngineer.transform() │
                    │  ROC, F&G, Funding, 200-EMA,   │
                    │  ADX, ATR, Volume, Sentiment   │
                    └──────────────┬─────────────────┘
                                   │ Feature Matrix
                    ┌──────────────▼─────────────────┐
                    │     ModelLoader.predict()       │
                    │  ┌─────────┐  ┌─────────┐     │
                    │  │Bull XGB │  │Bear XGB │     │
                    │  └────┬────┘  └────┬────┘     │
                    │       │            │           │
                    │  bull_prob     bear_prob       │
                    └──────┬────────────┬────────────┘
                           │            │
                    ┌──────▼────────────▼────────────┐
                    │   ExecutionEngine.evaluate()    │
                    │  Bull>0.70 & Bear<0.30 → LONG  │
                    │  Bear>0.70 & Bull<0.30 → SHORT │
                    │  Otherwise → FLAT               │
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

## The "Self-Healing" Mechanism

The bot automatically adapts to changing market conditions through the **Champion vs. Challenger** retraining pipeline.

### How It Works

```
                    ┌─────────────────────────────────────────┐
                    │          retrain.py (cron job)           │
                    │                                         │
                    │  1. Load historical + live data          │
                    │  2. Train CHALLENGER on [0 : -30 days]  │
                    │  3. Test CHALLENGER on last 30 days      │
                    │  4. Test CHAMPION on same 30 days        │
                    │                                         │
                    │  IF Challenger_Sharpe > Champion_Sharpe  │
                    │  AND Challenger_WR > 55%:               │
                    │                                         │
                    │    ┌──────────────────────┐             │
                    │    │ os.replace()         │ ◄── Atomic  │
                    │    │ temp.json → model.json│    Rename  │
                    │    └──────────────────────┘             │
                    └─────────────────┬───────────────────────┘
                                      │
                                      │ Model file mtime changes
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │        ModelLoader._check_reload()       │
                    │                                         │
                    │  Before EVERY prediction:                │
                    │  if os.path.getmtime() != stored_mtime: │
                    │    → Reload model from disk              │
                    │    → Log: "🔄 Hot-reloaded"              │
                    │                                         │
                    │  ZERO DOWNTIME                           │
                    └─────────────────────────────────────────┘
```

### Retraining Schedule

Recommended: **Weekly** (Sunday 03:00 UTC via cron):

```bash
# /etc/crontab or crontab -e
0 3 * * 0 cd /opt/crypto_bot && /opt/crypto_bot/venv/bin/python -m jobs.retrain --all
```

---

## Deployment Guide

### Prerequisites

- **Server**: DigitalOcean Droplet (Ubuntu 22.04, 1GB RAM minimum)
- **API Keys**: Binance API key & secret (with Futures trading enabled)
- **Optional**: Telegram bot token for alerts

### Step-by-Step Deployment

#### 1. Create DigitalOcean Droplet

```bash
# Recommended: Basic Droplet, 1GB RAM, 25GB SSD, Ubuntu 22.04
# Region: Singapore or any low-latency location to Binance
```

#### 2. Clone & Deploy

```bash
# On the server
git clone <YOUR_REPO_URL> /tmp/crypto_bot_src
cd /tmp/crypto_bot_src/crypto_bot

# Run deployment script (creates swap, user, venv, systemd)
sudo chmod +x deploy.sh
sudo ./deploy.sh
```

#### 3. Configure Environment

```bash
# Copy and edit the .env file
sudo cp .env.example /opt/crypto_bot/.env
sudo nano /opt/crypto_bot/.env

# Fill in your API keys:
# BINANCE_API_KEY=your_actual_key
# BINANCE_SECRET=your_actual_secret
# TRADE_MODE=paper     ← Start with paper!
```

#### 4. Copy Data & Models

```bash
# From your local machine
scp data/*.csv root@<SERVER_IP>:/opt/crypto_bot/data/
scp models/*.json root@<SERVER_IP>:/opt/crypto_bot/models/
```

#### 5. Start the Bot

```bash
# Start
sudo systemctl start crypto_bot

# Check status
sudo systemctl status crypto_bot

# View live logs
sudo journalctl -u crypto_bot -f

# Stop
sudo systemctl stop crypto_bot
```

#### 6. Set Up Retraining Cron

```bash
sudo crontab -e
# Add:
0 3 * * 0 cd /opt/crypto_bot && /opt/crypto_bot/venv/bin/python -m jobs.retrain --all >> /opt/crypto_bot/logs/retrain.log 2>&1
```

---

## Risk Controls

### Hard-Coded Safety (Cannot Be Overridden)

| Control | Value | Location | Purpose |
|---------|-------|----------|---------|
| **Max Leverage** | **1x** | `config.py` + `execution.py` | Prevents liquidation |
| **Stop Loss** | **3%** | `config.py` + `execution.py` | Caps per-trade loss |
| **Position Sizing** | **95% of capital** | `execution.py` | 5% buffer for fees |
| **Min Win Rate Gate** | **55%** | `config.py` + `retrain.py` | Prevents bad model promotion |
| **Dual-Gate Entry** | **Bull>0.7 & Bear<0.3** | `config.py` | High-conviction only |

### Position Sizing Rules

1. **Max 1 position per pair** — no pyramiding
2. **95% of available capital** — 5% reserved for fees and slippage
3. **No margin** — all trades are 1x (spot-equivalent risk)

### Stop Loss Mechanics

- **Long**: SL triggers at `entry_price × (1 - 0.03)` = -3%
- **Short**: SL triggers at `entry_price × (1 + 0.03)` = -3%
- Checked on **every tick** before signal evaluation
- Uses intra-candle High/Low for accurate fills (not just Close)

### Model Safety

- Models **cannot** be manually edited in production
- New models must pass the **Champion vs Challenger** gate:
  - Challenger Sharpe > Champion Sharpe
  - Challenger Win Rate > 55%
- Model updates are **atomic** (`os.replace()`) — crash-safe
- Bot auto-detects model changes via file timestamp monitoring

### Emergency Procedures

```bash
# Stop the bot immediately
sudo systemctl stop crypto_bot

# Disable auto-restart
sudo systemctl disable crypto_bot

# Check open positions on Binance
# (Manual review via Binance app or API)
```

---

## Feature Set (The "Shield" Features)

These 22 features were selected through ML research and validated via walk-forward testing:

| Category | Features | Source |
|----------|----------|--------|
| **Momentum** | `roc`, `roc_4h`, `roc_24h`, `roc_72h`, `roc_zscore` | Price close |
| **Sentiment** | `fear_greed_value`, `fg_normalized`, `fg_zscore` | alternative.me |
| **Derivatives** | `funding_rate`, `funding_zscore`, `funding_delta_4h`, `funding_delta_24h` | Binance Futures |
| **Trend** | `ema_200_dist`, `adx` | Price |
| **Volatility** | `atr`, `atr_pct`, `atr_pct_zscore` | Price |
| **Volume** | `volume_zscore`, `volume_delta_4h`, `volume_ratio_24h` | Spot volume |
| **Composite** | `sentiment_score`, `sentiment_raw` | Derived |

**Not used**: L/S Ratio, Open Interest (only 30 days of historical data — near-zero ML importance).

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
