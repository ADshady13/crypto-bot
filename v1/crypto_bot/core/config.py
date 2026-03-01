"""
Configuration — Loads environment variables safely with validation.

All sensitive values come from .env; this module provides typed,
validated access with sensible defaults for non-secret settings.
"""

import os
from dotenv import load_dotenv

# Load .env from the crypto_bot directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))
# Also try parent directory (repo root) .env
load_dotenv(os.path.join(os.path.dirname(_ROOT), ".env"))


class Config:
    """Centralized, validated configuration."""

    # --- Binance ---
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")

    # --- Telegram (optional) ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- Trading ---
    TRADE_MODE: str = os.getenv("TRADE_MODE", "paper")  # paper | live
    DEFAULT_PAIR: str = os.getenv("DEFAULT_PAIR", "BTCUSDT")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Paths (relative to crypto_bot/) ---
    DATA_DIR: str = os.path.join(_ROOT, os.getenv("DATA_DIR", "data"))
    MODEL_DIR: str = os.path.join(_ROOT, os.getenv("MODEL_DIR", "models"))
    LOG_DIR: str = os.path.join(_ROOT, "logs")

    # --- Strategy Constants (hard-coded for safety) ---
    LEVERAGE: int = 1               # HARD-CODED: Never > 1x
    FEE_PCT: float = 0.001          # 0.1% taker fee
    SLIPPAGE_PCT: float = 0.0005    # 0.05% estimated slippage
    SL_PCT: float = 0.03            # 3% stop loss
    FUNDING_INTERVAL_H: int = 8     # Hours between funding payments

    # --- ML Thresholds (Dual-Core) ---
    BULL_ENTRY_PROB: float = 0.70   # Bull prob > this → long
    BEAR_ENTRY_PROB: float = 0.70   # Bear prob > this → short
    BULL_BLOCK_PROB: float = 0.30   # Bull prob must be < this to allow short
    BEAR_BLOCK_PROB: float = 0.30   # Bear prob must be < this to allow long

    # --- Retraining ---
    RETRAIN_TEST_DAYS: int = 30     # Champion vs Challenger test window
    RETRAIN_MIN_WIN_RATE: float = 55.0

    # --- Polling ---
    CHECK_INTERVAL_SECONDS: int = 3600  # 1 hour

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of configuration errors (empty = valid)."""
        errors = []
        if not cls.BINANCE_API_KEY or cls.BINANCE_API_KEY == "your_api_key_here":
            errors.append("BINANCE_API_KEY is not set")
        if not cls.BINANCE_SECRET:
            errors.append("BINANCE_SECRET is not set")
        if cls.TRADE_MODE not in ("paper", "live"):
            errors.append(f"TRADE_MODE must be 'paper' or 'live', got '{cls.TRADE_MODE}'")
        if cls.LEVERAGE != 1:
            errors.append(f"LEVERAGE must be 1 (hard-coded safety), got {cls.LEVERAGE}")
        return errors

    @classmethod
    def ensure_dirs(cls):
        """Create required directories if they don't exist."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

    @classmethod
    def summary(cls) -> str:
        """Human-readable config summary (no secrets)."""
        return (
            f"Mode={cls.TRADE_MODE} | Pair={cls.DEFAULT_PAIR} | "
            f"Leverage={cls.LEVERAGE}x | SL={cls.SL_PCT*100}% | "
            f"Bull>{cls.BULL_ENTRY_PROB} Bear>{cls.BEAR_ENTRY_PROB} | "
            f"API={'SET' if cls.BINANCE_API_KEY else 'MISSING'}"
        )
