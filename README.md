# Crypto Trading Bot

A Python-based crypto trading bot using CCXT and Backtesting.py.

## Features
- **Strategy**: Bollinger Bands + RSI
- **Backtesting**: Simulate performance with historical data.
- **Paper Trading**: Test in real-time without risking funds.
- **Live Trading**: Execute trades on Binance (or other exchanges supported by CCXT).

## Setup
1.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure `.env` with your API keys.

## Running
- **Fetch Data**: `python3 core/data_fetcher.py`
- **Backtest**: `python3 backtest.py`
- **Run Bot**: `python3 main.py`
# crypto-bot
