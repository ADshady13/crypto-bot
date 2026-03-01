"""
Crypto Bot — V3 Triple-Core XGBoost System

Single CLI entry point for the production trading bot.

Usage:
    python main.py --mode paper --pairs "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"

The PortfolioManager controls isolated 10K INR wallets internally for each pair.
All API state logic, signal computations, and trade execution occur strictly sequentially to avoid rate limits and permit a single consolidated hourly Telegram ping.
"""

import argparse
import logging
import signal
import sys
import time
import os

# Ensure crypto_bot is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.logging_setup import setup_logging
from core.data_manager import DataManager
from core.execution import PortfolioManager
from core.notification import Notifier
from strategies.inference import ModelLoader

logger = logging.getLogger("crypto_bot")

# Global shutdown flag
shutdown_event = False

class GlobalRunner:
    """Synchronous runner for the entire 5-pair crypto portfolio."""

    def __init__(self, pairs: list[str], mode: str):
        self.pairs = pairs
        self.mode = mode

        # Global Components
        self.notifier = Notifier()
        self.portfolio = PortfolioManager(pairs, mode)
        
        # Load all static machine learning models once
        self.model_loader = ModelLoader(pairs)
        
        # Isolated Data Managers
        self.data_managers = {p: DataManager(p) for p in pairs}

    def initialize(self):
        """Load historical continuous cache for all pairs."""
        logger.info(f"Initializing V3 Portfolio in {self.mode.upper()} mode...")

        if self.mode == "live":
            errors = Config.validate()
            if errors:
                for e in errors:
                    logger.error(f"Config error: {e}")
                return False

        # Load historical cache for feature engineering warmth
        for p, dm in self.data_managers.items():
            dm.load_historical()
            if not self.model_loader.is_ready(p):
                logger.error(f"Missing models for {p}. Aborting.")
                return False

        logger.info(f"✅ Portfolio initialized. 10,000 INR isolated wallets provisioned for {len(self.pairs)} pairs.")
        self.notifier.send(f"🤖 V3 Portfolio Boot: Monitoring {len(self.pairs)} pairs in {self.mode.upper()} mode with Trailing Exits.")
        return True

    def run_portfolio_tick(self):
        """Runs the sequential cycle exactly on the hour rollover."""
        logger.info("--- Hourly Portfolio Tick Initiated ---")
        
        # 1. Fetch BTC Macro data first (REQUIRED for V3 Alts)
        btc_dm = self.data_managers["BTCUSDT"]
        btc_raw = btc_dm.update()
        if btc_raw.empty or len(btc_raw) < 200:
            logger.warning("BTC Macro data insufficient. Cannot process ticks.")
            return
            
        # Extract features for BTC Macro linkage
        btc_features = self.model_loader.feature_engineer.transform(btc_raw.copy(), verbose=False)

        # 2. Iterate through all bots
        for pair in self.pairs:
            bot = self.portfolio.bots[pair]
            dm = self.data_managers[pair]
            
            # Update the individual coin
            df = dm.update()
            if df.empty or len(df) < 250:
                logger.warning(f"[{pair}] Insufficient data, skipping tick")
                continue
                
            current_price = float(df["Close"].iloc[-1])
            high = float(df["High"].iloc[-1])
            low = float(df["Low"].iloc[-1])
            
            # We need the ATR at this very moment to update our Trailing logic mathematically
            # Rather than calculating all features, we just grab the last ATR
            atr_val = self.model_loader.feature_engineer.atr(df["High"], df["Low"], df["Close"], Config.ATR_PERIOD).iloc[-1] if hasattr(Config, "ATR_PERIOD") else 0.0
            
            # --- 2a. Tick Open Positions (Stop Losses / Trailing Stops / Timeouts) ---
            exit_result = bot.process_tick_updates(current_price, high, low)
            if exit_result:
                self.portfolio.log_trade(exit_result)
                
            # --- 2b. Evaluate ML Models for New/Reversal Signals ---
            window = dm.get_feature_window(lookback=500)
            
            # Inject BTC features explicitly if this isn't BTC
            bull_prob, bear_prob, time_prob = self.model_loader.predict(window, pair, btc_features if pair != "BTCUSDT" else None)
            
            action = bot.evaluate_signal(current_price, atr_val, bull_prob, bear_prob, time_prob)
            logger.info(f"[{pair}] B:{bull_prob:.2f} R:{bear_prob:.2f} T:{time_prob:.2f} -> {action}")
            
            # Stagger Binance API hits slightly
            time.sleep(1)

        # 3. Save internal JSON state
        self.portfolio.save_state()

        # 4. Blast the consolidated Telegram message
        tg_msg = self.portfolio.generate_telegram_summary()
        self.notifier.send(tg_msg)
        
        logger.info("--- Hourly Portfolio Tick Completed ---")

    def run_forever(self):
        """Main time-aligned execution loop."""
        global shutdown_event
        
        logger.info(f"Starting V3 Loop (Interval: {Config.CHECK_INTERVAL_SECONDS}s)")

        while not shutdown_event:
            try:
                self.run_portfolio_tick()
            except Exception as e:
                logger.error(f"Global Tick Error: {e}", exc_info=True)
                self.notifier.send(f"⚠️ V3 Portfolio Crash Loop Error:\n{e}")

            # Sleep until the NEXT hour rollover + 30s buffer for Binance candle settling
            now = time.time()
            next_hour = ((int(now) // 3600) + 1) * 3600
            sleep_duration = next_hour - now + 30
            
            logger.info(f"Sleeping {sleep_duration:.0f}s until next candle close...")
            
            while sleep_duration > 0 and not shutdown_event:
                chunk = min(sleep_duration, 10)
                time.sleep(chunk)
                sleep_duration -= chunk

        logger.info("Graceful Shutdown. Saving finalizing state...")
        self.portfolio.save_state()
        self.notifier.send("🛑 CryptoBot V3 Portfolio Engine Offline.")


# ======================================================================
#  CLI ENTRY POINT
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Crypto Bot V3 Portfolio Runner")

    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--pairs", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
        help="Comma separated list of pairs (e.g. BTCUSDT,ETHUSDT)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    Config.ensure_dirs()
    
    # Needs a default ATR period value for fast access, checking if exists, else append to Config directly on boot
    if not hasattr(Config, "ATR_PERIOD"):
        Config.ATR_PERIOD = 14

    pairs_list = [p.strip().upper() for p in args.pairs.split(",")]

    logger.info("=" * 60)
    logger.info("  CRYPTO BOT V3 — Portfolio Manager Engine")
    logger.info("=" * 60)
    logger.info(f"  Mode: {args.mode} | Pairs: {len(pairs_list)}")
    logger.info("=" * 60)

    # ---- Signal handler ----
    def handle_shutdown(signum, frame):
        global shutdown_event
        logger.info("Shutdown signal received...")
        shutdown_event = True

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Instantiate the monolith runner and go
    runner = GlobalRunner(pairs_list, args.mode)
    if not runner.initialize():
        sys.exit(1)
        
    runner.run_forever()


if __name__ == "__main__":
    main()
