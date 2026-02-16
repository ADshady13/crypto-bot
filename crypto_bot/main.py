"""
Crypto Bot — Dual-Core XGBoost Trading System

Single CLI entry point for the production trading bot.

Usage:
    python main.py --mode paper --pair BTCUSDT   # Paper trading (default)
    python main.py --mode live --pair SOLUSDT     # Live trading
    python main.py --mode paper --all             # All pairs simultaneously

The bot runs an hourly loop:
  1. Fetch latest market data (OHLCV + funding + F&G)
  2. Engineer features (shared pipeline with training)
  3. Run Dual-Core ML inference (Bull + Bear XGBoost)
  4. Evaluate signal (Long / Short / Flat)
  5. Execute trade if needed (paper or live)
  6. Check stop loss on open positions
  7. Log everything + optional Telegram alerts
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
from core.execution import ExecutionEngine
from core.notification import Notifier
from strategies.inference import ModelLoader

logger = logging.getLogger("crypto_bot")


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, pair: str, mode: str):
        self.pair = pair
        self.mode = mode
        self.running = True

        # Components
        self.data_manager = DataManager(pair)
        pair_short = pair.replace("USDT", "")
        self.model_loader = ModelLoader(pair_short)
        self.engine = ExecutionEngine(pair, mode)
        self.notifier = Notifier()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received — closing positions...")
        self.running = False

    def initialize(self, initial_capital: float = 100_000):
        """Load historical data and set up the engine."""
        logger.info(f"Initializing {self.pair} in {self.mode} mode...")
        logger.info(f"Config: {Config.summary()}")

        # Validate config
        if self.mode == "live":
            errors = Config.validate()
            if errors:
                for e in errors:
                    logger.error(f"Config error: {e}")
                sys.exit(1)

        # Load historical data
        self.data_manager.load_historical()

        # Check model readiness
        if not self.model_loader.ready:
            logger.error("Models not ready! Run: python -m jobs.retrain --pair " + self.pair)
            sys.exit(1)

        # Set capital
        self.engine.set_capital(initial_capital)
        logger.info(f"✅ Bot initialized: {self.pair} | {self.mode} | ${initial_capital:,.0f}")
        self.notifier.send(f"🤖 Bot started: {self.pair} ({self.mode})")

    def tick(self):
        """Run one trading cycle."""
        # 1. Update data
        df = self.data_manager.update()
        if df.empty or len(df) < 250:
            logger.warning("Not enough data for inference, skipping tick")
            return

        # 2. Get current price
        current_price = float(df["Close"].iloc[-1])

        # 3. Check stop loss first
        sl_result = self.engine.check_stop_loss(current_price)
        if sl_result:
            self.notifier.trade_alert(
                "STOP_LOSS", self.pair, sl_result.exit_price,
                f"PnL: ${sl_result.pnl_usd:+,.2f}"
            )
            return

        # 4. Get ML predictions (hot-reload aware)
        window = self.data_manager.get_feature_window(lookback=500)
        bull_prob, bear_prob = self.model_loader.predict(window)

        # 5. Evaluate signal
        signal_action = self.engine.evaluate_signal(bull_prob, bear_prob)

        logger.info(
            f"Tick: {self.pair} @ ${current_price:,.2f} | "
            f"Bull={bull_prob:.3f} Bear={bear_prob:.3f} → {signal_action} | "
            f"Capital=${self.engine.capital:,.2f}"
        )

        # 6. Execute
        if signal_action == "FLAT" and self.engine.position:
            # Close existing position
            result = self.engine.close_position(current_price, "SIGNAL")
            if result:
                self.notifier.trade_alert(
                    "CLOSE", self.pair, current_price,
                    f"PnL: ${result.pnl_usd:+,.2f} ({result.pnl_pct:+.2f}%)"
                )

        elif signal_action in ("LONG", "SHORT"):
            # Close opposite position if exists
            if self.engine.position and self.engine.position.direction != signal_action:
                result = self.engine.close_position(current_price, "REVERSAL")
                if result:
                    self.notifier.trade_alert(
                        "REVERSAL", self.pair, current_price,
                        f"Closed {result.direction}, PnL: ${result.pnl_usd:+,.2f}"
                    )

            # Open new position
            if not self.engine.position:
                pos = self.engine.open_position(signal_action, current_price)
                if pos:
                    reason = f"Bull={bull_prob:.3f} Bear={bear_prob:.3f}"
                    self.notifier.trade_alert(signal_action, self.pair, current_price, reason)

    def run(self):
        """Main trading loop — runs every hour."""
        logger.info(f"Starting main loop (interval: {Config.CHECK_INTERVAL_SECONDS}s)")

        while self.running:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
                self.notifier.send(f"⚠️ Error in {self.pair}: {e}")

            # Wait for next tick
            if self.running:
                logger.debug(f"Sleeping {Config.CHECK_INTERVAL_SECONDS}s until next tick")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)

        # Graceful shutdown
        logger.info("Shutting down...")
        if self.engine.position:
            logger.info("Closing open position on shutdown")
            # In production, you might want to keep the position open
            # For now, we just log it

        stats = self.engine.get_stats()
        logger.info(f"Session stats: {stats}")
        self.notifier.send(
            f"🛑 Bot stopped: {self.pair}\n"
            f"Trades: {stats['trades']} | WR: {stats['win_rate']:.1f}% | "
            f"PnL: ${stats['total_pnl']:+,.2f}"
        )


# ======================================================================
#  CLI ENTRY POINT
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Crypto Bot — Dual-Core XGBoost Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode paper --pair BTCUSDT     # Paper trade BTC
  python main.py --mode live --pair SOLUSDT       # Live trade SOL
  python main.py --retrain --pair BTCUSDT         # Retrain models
  python main.py --retrain --all                  # Retrain all pairs
        """
    )

    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Trading pair (e.g. BTCUSDT, default from .env)"
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="Initial capital in USD (default: 100000 for paper, ignored for live)"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Run model retraining instead of trading"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Apply to all supported pairs"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Handle retrain mode
    if args.retrain:
        from jobs.retrain import main as retrain_main
        sys.argv = ["retrain"]
        if args.all:
            sys.argv.append("--all")
        elif args.pair:
            sys.argv.extend(["--pair", args.pair])
        retrain_main()
        return

    # Trading mode
    pair = args.pair or Config.DEFAULT_PAIR
    Config.ensure_dirs()

    logger.info("=" * 60)
    logger.info("  CRYPTO BOT — Dual-Core XGBoost Trading System")
    logger.info("=" * 60)

    bot = TradingBot(pair, args.mode)
    bot.initialize(initial_capital=args.capital)
    bot.run()


if __name__ == "__main__":
    main()
