# CryptoBot V1 (Legacy Dual-Core) - Deployment Guide

This folder contains the **Legacy V1 Dual-Core** version of CryptoBot. This codebase utilizes fixed targets (Hard 2% TP, Hard 1% SL) and static 95% position sizing.
**This version has been deprecated in favor of V2 due to severe limitations in sizing logic and chop recognition.**

## Redeployment Instructions (For Archival/Testing Purposes)

### 1. Prerequisites
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Requires a `.env` file in the root `crypto_bot` folder for execution matching the V1 format.
```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

### 3. Execution
To run the static execution legacy backend:
```bash
cd crypto_bot
python main.py --mode paper --pair BTCUSDT
```

*Note: There is no deploy.sh or systemd file configured natively in this archival folder as it is not meant for remote VPS live-action deployment.*
