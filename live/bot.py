#!/usr/bin/env python3
"""
Crypto Sentiment Bot - Live Trading
24/7 trading bot for SOL/XRP using derivatives sentiment
"""

import os
import sys
import time
import logging
import yaml
from datetime import datetime
from pathlib import Path

# Add parent directory to path for derivatives_backtest
sys.path.insert(0, str(Path(__file__).parent.parent / "derivatives_backtest"))

# Import from derivatives_backtest
from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer
from core.signal_generator import BacktestSignalExecutor
import ccxt


class TradingBot:
    def __init__(self, config_path="live/config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.positions = {}  # {symbol: {'entry_price': x, 'entry_time': y, 'size': z}}
        self.daily_pnl = {}
        self.last_trade_time = {}

        # Initialize exchange
        self.exchange = self.init_exchange()

        self.logger.info("Bot initialized")
        self.logger.info(f"Mode: {self.config['execution']['mode']}")
        self.logger.info(
            f"Pairs: {[p['symbol'] for p in self.config['pairs'] if p['enabled']]}"
        )

    def load_config(self, config_path):
        """Load configuration from YAML"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Set up logging"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler()
                if log_config.get("console", True)
                else logging.NullHandler(),
                logging.FileHandler(log_config.get("file", "logs/bot.log"))
                if log_config.get("file")
                else logging.NullHandler(),
            ],
        )
        self.logger = logging.getLogger("TradingBot")

    def init_exchange(self):
        """Initialize Binance exchange"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET")

        testnet = self.config["binance"].get("testnet", True)

        if testnet:
            # Use testnet
            exchange = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot", "testnet": True},
                }
            )
            self.logger.info("Using Binance TESTNET")
        else:
            exchange = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            self.logger.info("Using Binance LIVE")

        return exchange

    def get_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance["total"].get("USDT", 0)
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0

    def fetch_data(self, symbol):
        """Fetch current data for symbol"""
        fetcher = DerivativesDataFetcher(symbol)
        df = fetcher.fetch_all_data(limit=self.config["data"]["lookback_candles"])

        # Engineer features
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)
        df = engineer.compute_sentiment_score(df)

        return df

    def check_entry_signal(self, df, params):
        """Check if entry signal is present"""
        latest = df.iloc[-1]

        conditions = (
            latest["sentiment_score"] > params["sentiment_entry_threshold"]
            and latest["funding_zscore"] < params["funding_zscore_entry"]
            and latest["long_short_ratio_zscore"] < params["lsr_zscore_entry"]
            and latest["price_above_ema"] == True
        )

        return conditions, latest

    def check_exit_signal(self, df, position, params):
        """Check if exit signal is present"""
        latest = df.iloc[-1]
        entry_price = position["entry_price"]
        current_price = latest["Close"]
        atr = latest["atr"]

        # Exit conditions
        if latest["sentiment_score"] < params["sentiment_exit_threshold"]:
            return True, "SENTIMENT_EXIT"

        # Stop loss
        sl_price = entry_price * (1 - self.config["risk"]["stop_loss_pct"] / 100)
        if current_price <= sl_price:
            return True, "STOP_LOSS"

        # Take profit (ATR based)
        tp_price = entry_price + (atr * params["atr_tp_multiplier"])
        if current_price >= tp_price:
            return True, "TAKE_PROFIT"

        return False, None

    def execute_buy(self, symbol, amount):
        """Execute buy order"""
        mode = self.config["execution"]["mode"]

        if mode == "paper":
            self.logger.info(f"[PAPER] BUY {amount} {symbol}")
            return {
                "status": "filled",
                "price": self.exchange.fetch_ticker(symbol)["last"],
            }
        else:
            try:
                order = self.exchange.create_market_buy_order(symbol, amount)
                self.logger.info(f"LIVE BUY: {order}")
                return order
            except Exception as e:
                self.logger.error(f"Buy error: {e}")
                return None

    def execute_sell(self, symbol, amount):
        """Execute sell order"""
        mode = self.config["execution"]["mode"]

        if mode == "paper":
            self.logger.info(f"[PAPER] SELL {amount} {symbol}")
            return {
                "status": "filled",
                "price": self.exchange.fetch_ticker(symbol)["last"],
            }
        else:
            try:
                order = self.exchange.create_market_sell_order(symbol, amount)
                self.logger.info(f"LIVE SELL: {order}")
                return order
            except Exception as e:
                self.logger.error(f"Sell error: {e}")
                return None

    def run_cycle(self):
        """Run one trading cycle"""
        self.logger.info("=" * 50)
        self.logger.info(f"Trading cycle started at {datetime.now()}")

        balance = self.get_balance()
        self.logger.info(f"Balance: ${balance:.2f} USDT")

        for pair_config in self.config["pairs"]:
            if not pair_config["enabled"]:
                continue

            symbol = pair_config["symbol"]
            params = pair_config["params"]

            self.logger.info(f"\n--- Checking {symbol} ---")

            try:
                # Fetch data
                df = self.fetch_data(symbol)
                current_price = df.iloc[-1]["Close"]
                self.logger.info(f"Current price: ${current_price:.4f}")
                self.logger.info(
                    f"Sentiment score: {df.iloc[-1]['sentiment_score']:.3f}"
                )
                self.logger.info(
                    f"Funding z-score: {df.iloc[-1]['funding_zscore']:.2f}"
                )
                self.logger.info(
                    f"L/S ratio z-score: {df.iloc[-1]['long_short_ratio_zscore']:.2f}"
                )

                # Check if we have a position
                if symbol in self.positions:
                    # Check exit signal
                    should_exit, reason = self.check_exit_signal(
                        df, self.positions[symbol], params
                    )
                    if should_exit:
                        position = self.positions[symbol]
                        self.logger.info(f"EXIT SIGNAL: {reason}")

                        result = self.execute_sell(symbol, position["size"])
                        if result:
                            pnl = (
                                result["price"] - position["entry_price"]
                            ) * position["size"]
                            self.logger.info(f"Closed position: PnL = ${pnl:.2f}")
                            del self.positions[symbol]
                    else:
                        self.logger.info("No exit signal")

                # Check entry signal (if no position)
                else:
                    should_entry, latest = self.check_entry_signal(df, params)

                    if should_entry:
                        self.logger.info("ENTRY SIGNAL!")

                        # Calculate position size
                        trade_pct = self.config["execution"]["trade_size_pct"]
                        usdt_amount = balance * trade_pct
                        amount = usdt_amount / current_price

                        self.logger.info(
                            f"Executing buy: ${usdt_amount:.2f} ({amount:.4f} {symbol.replace('USDT', '')})"
                        )

                        result = self.execute_buy(symbol, amount)
                        if result:
                            self.positions[symbol] = {
                                "entry_price": result["price"],
                                "entry_time": datetime.now(),
                                "size": amount,
                            }
                            self.logger.info(
                                f"Opened position at ${result['price']:.4f}"
                            )
                    else:
                        self.logger.info("No entry signal")

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                import traceback

                traceback.print_exc()

        self.logger.info(f"\nOpen positions: {len(self.positions)}")
        for sym, pos in self.positions.items():
            self.logger.info(f"  {sym}: {pos['size']} @ ${pos['entry_price']:.4f}")

    def run_forever(self):
        """Run bot continuously"""
        interval = self.config["execution"]["check_interval_minutes"] * 60

        self.logger.info(f"Starting bot - checking every {interval / 60:.0f} minutes")
        self.logger.info("Press Ctrl+C to stop")

        try:
            while True:
                self.run_cycle()
                self.logger.info(f"Sleeping for {interval / 60:.0f} minutes...")
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")

        # Close any open positions on exit
        if self.positions:
            self.logger.warning("Closing open positions...")
            for symbol, position in self.positions.items():
                self.execute_sell(symbol, position["size"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Sentiment Bot")
    parser.add_argument("--config", default="live/config.yaml", help="Config file path")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")

    args = parser.parse_args()

    bot = TradingBot(args.config)

    if args.once:
        bot.run_cycle()
    else:
        bot.run_forever()


if __name__ == "__main__":
    main()
