"""
Crypto Bot — Dual-Core XGBoost Trading System

Single CLI entry point for the production trading bot.

Usage:
    python main.py --mode paper --pair BTCUSDT                    # Single pair
    python main.py --mode paper --pairs BTCUSDT:50000,SOLUSDT:50000  # Multi-pair
    python main.py --mode paper --all --capital 100000            # All pairs (equal split)
    python main.py --retrain --all                                # Retrain all models

Multi-pair mode runs each pair in its own thread with individual capital
allocations. They share the same Binance API connection but track PnL
and positions independently.
"""

import argparse
import logging
import signal
import sys
import time
import os
import threading

# Ensure crypto_bot is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.logging_setup import setup_logging
from core.data_manager import DataManager
from core.execution import ExecutionEngine
from core.notification import Notifier
from strategies.inference import ModelLoader

logger = logging.getLogger("crypto_bot")

# Global shutdown flag (shared across all threads)
shutdown_event = threading.Event()


class TradingBot:
    """Trading bot for a single pair. Multiple instances run concurrently."""

    def __init__(self, pair: str, mode: str, capital: float):
        self.pair = pair
        self.mode = mode
        self.capital = capital

        # Components
        self.data_manager = DataManager(pair)
        pair_short = pair.replace("USDT", "")
        self.model_loader = ModelLoader(pair_short)
        self.engine = ExecutionEngine(pair, mode)
        self.notifier = Notifier()

    def initialize(self):
        """Load historical data and set up the engine."""
        logger.info(f"[{self.pair}] Initializing in {self.mode} mode...")

        # Validate config for live mode
        if self.mode == "live":
            errors = Config.validate()
            if errors:
                for e in errors:
                    logger.error(f"Config error: {e}")
                return False

        # Load historical data
        self.data_manager.load_historical()

        # Check model readiness
        if not self.model_loader.ready:
            logger.error(f"[{self.pair}] Models not ready! Run: python -m jobs.retrain --pair {self.pair}")
            return False

        # Set capital
        self.engine.set_capital(self.capital)
        logger.info(f"✅ [{self.pair}] Initialized | {self.mode} | ${self.capital:,.0f}")
        self.notifier.send(f"🤖 Bot started: {self.pair} ({self.mode}) — ${self.capital:,.0f}")
        return True

    def tick(self):
        """Run one trading cycle."""
        # 1. Update data
        df = self.data_manager.update()
        if df.empty or len(df) < 250:
            logger.warning(f"[{self.pair}] Not enough data, skipping tick")
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
            f"[{self.pair}] @ ${current_price:,.2f} | "
            f"Bull={bull_prob:.3f} Bear={bear_prob:.3f} → {signal_action} | "
            f"Capital=${self.engine.capital:,.2f}"
        )

        # 6. Execute
        if signal_action == "FLAT" and self.engine.position:
            result = self.engine.close_position(current_price, "SIGNAL")
            if result:
                self.notifier.trade_alert(
                    "CLOSE", self.pair, current_price,
                    f"PnL: ${result.pnl_usd:+,.2f} ({result.pnl_pct:+.2f}%)"
                )

        elif signal_action in ("LONG", "SHORT"):
            if self.engine.position and self.engine.position.direction != signal_action:
                result = self.engine.close_position(current_price, "REVERSAL")
                if result:
                    self.notifier.trade_alert(
                        "REVERSAL", self.pair, current_price,
                        f"Closed {result.direction}, PnL: ${result.pnl_usd:+,.2f}"
                    )

            if not self.engine.position:
                pos = self.engine.open_position(signal_action, current_price)
                if pos:
                    reason = f"Bull={bull_prob:.3f} Bear={bear_prob:.3f}"
                    self.notifier.trade_alert(signal_action, self.pair, current_price, reason)

    def run(self):
        """Main trading loop — runs until shutdown_event is set."""
        logger.info(f"[{self.pair}] Starting loop (interval: {Config.CHECK_INTERVAL_SECONDS}s)")

        while not shutdown_event.is_set():
            try:
                self.tick()
            except Exception as e:
                logger.error(f"[{self.pair}] Tick error: {e}", exc_info=True)
                self.notifier.send(f"⚠️ Error in {self.pair}: {e}")

            # Wait for next tick — check shutdown_event every 10s for fast exits
            remaining = Config.CHECK_INTERVAL_SECONDS
            while remaining > 0 and not shutdown_event.is_set():
                time.sleep(min(remaining, 10))
                remaining -= 10

        # Graceful shutdown
        logger.info(f"[{self.pair}] Shutting down...")
        stats = self.engine.get_stats()
        logger.info(f"[{self.pair}] Session: {stats}")
        self.notifier.send(
            f"🛑 Bot stopped: {self.pair}\n"
            f"Trades: {stats['trades']} | WR: {stats['win_rate']:.1f}% | "
            f"PnL: ${stats['total_pnl']:+,.2f}"
        )


def run_bot_thread(pair: str, mode: str, capital: float):
    """Thread entry point for running a single pair's bot."""
    bot = TradingBot(pair, mode, capital)
    if not bot.initialize():
        logger.error(f"[{pair}] Failed to initialize — thread exiting")
        return
    bot.run()


def parse_pairs_string(pairs_str: str) -> dict:
    """
    Parse --pairs format: 'BTCUSDT:50000,SOLUSDT:50000'
    Returns {pair: capital} dict.
    """
    pairs = {}
    for entry in pairs_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            pair, cap = entry.split(":", 1)
            pairs[pair.strip().upper()] = float(cap.strip())
        else:
            raise ValueError(
                f"Invalid pair format: '{entry}'. "
                f"Use PAIR:CAPITAL format, e.g. 'BTCUSDT:50000,SOLUSDT:50000'"
            )
    return pairs


# ======================================================================
#  CLI ENTRY POINT
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Crypto Bot — Dual-Core XGBoost Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode paper --pair BTCUSDT                       # Single pair
  python main.py --mode paper --pairs BTCUSDT:50000,SOLUSDT:50000  # Multi-pair
  python main.py --mode paper --all --capital 100000               # All 5 pairs (equal split)
  python main.py --retrain --pair BTCUSDT                          # Retrain models
  python main.py --retrain --all                                   # Retrain all
        """
    )

    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Single trading pair (e.g. BTCUSDT)"
    )
    parser.add_argument(
        "--pairs", type=str, default=None,
        help="Multi-pair with allocations: 'BTCUSDT:50000,SOLUSDT:50000'"
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="Capital per pair for --pair, or total for --all (default: 100000)"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Run model retraining instead of trading"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Trade all supported pairs (equal capital split)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    Config.ensure_dirs()

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

    # ---- Build pair→capital mapping ----
    ALL_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

    if args.pairs:
        # Explicit multi-pair: --pairs BTCUSDT:50000,SOLUSDT:50000
        pair_allocations = parse_pairs_string(args.pairs)
    elif args.all:
        # All pairs with equal split
        cap_each = args.capital / len(ALL_PAIRS)
        pair_allocations = {p: cap_each for p in ALL_PAIRS}
    elif args.pair:
        # Single pair
        pair_allocations = {args.pair: args.capital}
    else:
        # Default pair from .env
        pair_allocations = {Config.DEFAULT_PAIR: args.capital}

    # ---- Banner ----
    logger.info("=" * 60)
    logger.info("  CRYPTO BOT — Dual-Core XGBoost Trading System")
    logger.info("=" * 60)
    logger.info(f"  Mode: {args.mode} | Pairs: {len(pair_allocations)}")
    for pair, cap in pair_allocations.items():
        logger.info(f"    {pair}: ${cap:,.0f}")
    logger.info(f"  Total Capital: ${sum(pair_allocations.values()):,.0f}")
    logger.info("=" * 60)

    # ---- Signal handler for graceful shutdown ----
    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping all pairs...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ---- Launch ----
    if len(pair_allocations) == 1:
        # Single pair — run in main thread (simpler)
        pair, cap = next(iter(pair_allocations.items()))
        bot = TradingBot(pair, args.mode, cap)
        if not bot.initialize():
            sys.exit(1)
        bot.run()
    else:
        # Multi-pair — one thread per pair
        threads = []
        for pair, cap in pair_allocations.items():
            t = threading.Thread(
                target=run_bot_thread,
                args=(pair, args.mode, cap),
                name=f"bot-{pair}",
                daemon=True,
            )
            threads.append(t)
            t.start()
            time.sleep(2)  # Stagger API calls to avoid rate limits

        logger.info(f"All {len(threads)} pair threads started")

        # Wait for shutdown signal
        try:
            while not shutdown_event.is_set():
                shutdown_event.wait(timeout=60)

                # Log combined status every 60s
                alive = sum(1 for t in threads if t.is_alive())
                if alive > 0:
                    logger.debug(f"Heartbeat: {alive}/{len(threads)} threads alive")
                else:
                    logger.warning("All threads dead — exiting")
                    break
        except KeyboardInterrupt:
            shutdown_event.set()

        # Wait for all threads to finish graceful shutdown
        for t in threads:
            t.join(timeout=30)

        logger.info("All pair threads stopped. Goodbye.")


if __name__ == "__main__":
    main()
