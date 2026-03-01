# CryptoBot V2 (Current Dynamic XGBoost) - Deployment Guide

This folder contains the active **V2/V3 CryptoBot** using Triple-Core XGBoost models with Dynamic Position Sizing (Kelly Criterion metrics) and Trailing Chandelier Exits.

## Deployment Instructions

### 1. Prerequisites
Ensure you are running Python 3.10+ and have your `.env` configured inside this folder.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
You must create a `.env` file in the root `crypto_bot` folder with the following:
```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TG_BOT_TOKEN=your_telegram_bot_token
TG_CHAT_ID=your_chat_id
```

### 3. Running Locally (Paper Trading)
Run the bot locally to verify the execution logic without spending real money.
```bash
cd crypto_bot
python main.py --mode paper --pair BTCUSDT
```

### 4. Deploying to DigitalOcean VPS
If deploying this specific version to the remote server, use the included deployment script:
```bash
cd crypto_bot
chmod +x deploy.sh
./deploy.sh
```

The systemd service `bot.service` is configured to run `main.py --mode live`. 

### Check Logs
Once live, you can monitor the bot hitting the exchange via:
```bash
tail -f logs/crypto_bot_live.log
```
