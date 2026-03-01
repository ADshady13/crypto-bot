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

    # --- Strategy Constants (V3 Dynamic) ---
    LEVERAGE: int = 1               # HARD-CODED: Never > 1x
    FEE_PCT: float = 0.0004         # 0.04% taker fee Binance
    SLIPPAGE_PCT: float = 0.0003    # 0.03% estimated slippage
    
    # Trailing Stop Constants
    SL_ATR_MULT: float = 1.0          # Initial hard stop loss distance
    TRAILING_ACTIVATION: float = 1.0  # Activate trailing stop when profit hits 1.0×ATR
    TRAILING_PULLBACK: float = 0.5    # Trail behind MFE by 0.5×ATR
    MAX_HOLD_HOURS: int = 12

    # --- ML Thresholds (V3 Macro Optuna Grid) ---
    # These are the best performing thresholds exactly as evaluated in OOT
    THRESHOLDS = {
        "ETHUSDT": {"bull": 0.50, "bear": 0.60, "time": 0.50},
        "XRPUSDT": {"bull": 0.55, "bear": 0.50, "time": 0.55},
        "SOLUSDT": {"bull": 0.50, "bear": 0.45, "time": 0.45},
        "BNBUSDT": {"bull": 0.55, "bear": 0.50, "time": 0.40},
        "BTCUSDT": {"bull": 0.50, "bear": 0.45, "time": 0.35},
    }

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
            f"Mode={cls.TRADE_MODE} | Leverage={cls.LEVERAGE}x | "
            f"Trail={cls.TRAILING_ACTIVATION}xATR / {cls.TRAILING_PULLBACK}xATR | "
            f"API={'SET' if cls.BINANCE_API_KEY else 'MISSING'}"
        )
