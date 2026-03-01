"""
Logging Setup — Rotating file handler + console output.

Provides structured logging with automatic file rotation,
preventing unbounded log growth on VPS deployments.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from core.config import Config


def setup_logging(name: str = "crypto_bot") -> logging.Logger:
    """
    Configure and return the application logger.

    - Console: colored, human-readable
    - File: rotating, max 5MB × 3 backups = 15MB max disk usage
    """
    Config.ensure_dirs()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

    # Prevent duplicate handlers on re-init
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console Handler ---
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # --- Rotating File Handler ---
    log_path = os.path.join(Config.LOG_DIR, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized → {log_path}")
    return logger
