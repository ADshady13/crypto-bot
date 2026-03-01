"""
Data Manager — Unified data loading for both static (historical) and live feeds.

Responsibilities:
  1. Load cached historical OHLCV + Funding + Fear & Greed from CSV
  2. Fetch latest live candles from Binance (appending to history)
  3. Deduplicate and maintain the master DataFrame
  4. Provide a clean window of data for feature engineering
"""

import os
import logging
import time

import ccxt
import pandas as pd
import numpy as np
import requests

from core.config import Config

logger = logging.getLogger("crypto_bot")


class DataManager:
    """Handles all data ingestion — historical cache + live updates."""

    def __init__(self, pair: str):
        self.pair = pair
        self.spot_symbol = f"{pair.replace('USDT', '')}/USDT"
        self.cache_path = os.path.join(Config.DATA_DIR, f"{pair}_data.csv")
        self.df: pd.DataFrame = pd.DataFrame()

        # Initialize exchange
        config = {"enableRateLimit": True}
        if Config.BINANCE_API_KEY:
            config["apiKey"] = Config.BINANCE_API_KEY
            config["secret"] = Config.BINANCE_SECRET
        self.exchange = ccxt.binance(config)

    # ------------------------------------------------------------------ #
    #  LOAD: Historical data from cache
    # ------------------------------------------------------------------ #
    def load_historical(self) -> pd.DataFrame:
        """Load cached historical data. Returns empty DataFrame if no cache."""
        if not os.path.exists(self.cache_path):
            logger.warning(f"No cached data for {self.pair} at {self.cache_path}")
            return pd.DataFrame()

        self.df = pd.read_csv(self.cache_path, index_col="timestamp", parse_dates=True)
        logger.info(f"Loaded {len(self.df)} rows from {self.cache_path} "
                     f"({self.df.index[0].date()} to {self.df.index[-1].date()})")
        return self.df

    # ------------------------------------------------------------------ #
    #  FETCH: Live data from Binance
    # ------------------------------------------------------------------ #
    def fetch_latest_candles(self, limit: int = 200) -> pd.DataFrame:
        """Fetch the most recent hourly candles from Binance Spot."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.spot_symbol, "1h", limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch live candles: {e}")
            return pd.DataFrame()

    def fetch_latest_funding(self) -> float:
        """Fetch the most recent funding rate."""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            resp = requests.get(url, params={"symbol": self.pair, "limit": 1}, timeout=10)
            data = resp.json()
            if data:
                return float(data[0]["fundingRate"])
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
        return 0.0001  # Default

    def fetch_fear_greed(self) -> float:
        """Fetch the latest Fear & Greed index value."""
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1&format=json", timeout=10
            )
            data = resp.json()
            return float(data["data"][0]["value"])
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
        return 50.0  # Default: neutral

    # ------------------------------------------------------------------ #
    #  UPDATE: Append live data to history (deduplicated)
    # ------------------------------------------------------------------ #
    def update(self) -> pd.DataFrame:
        """
        Fetch latest candles and append to historical data.
        Returns the full updated DataFrame.
        """
        live = self.fetch_latest_candles(limit=48)
        if live.empty:
            return self.df

        # Add funding rate and F&G to live candles
        funding = self.fetch_latest_funding()
        fg = self.fetch_fear_greed()
        live["funding_rate"] = funding
        live["fear_greed_value"] = fg

        if self.df.empty:
            self.df = live
        else:
            # Append and deduplicate by index (timestamp)
            combined = pd.concat([self.df, live])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            self.df = combined

        logger.info(f"Updated: {len(self.df)} rows, latest={self.df.index[-1]}")
        return self.df

    # ------------------------------------------------------------------ #
    #  SAVE: Persist to cache
    # ------------------------------------------------------------------ #
    def save(self):
        """Save current DataFrame to cache."""
        Config.ensure_dirs()
        self.df.to_csv(self.cache_path)
        logger.info(f"Saved {len(self.df)} rows to {self.cache_path}")

    # ------------------------------------------------------------------ #
    #  WINDOW: Get a feature-ready slice
    # ------------------------------------------------------------------ #
    def get_feature_window(self, lookback: int = 500) -> pd.DataFrame:
        """
        Return the last `lookback` rows of data for feature engineering.
        This ensures sufficient history for z-scores and rolling features.
        """
        if len(self.df) < lookback:
            return self.df.copy()
        return self.df.iloc[-lookback:].copy()

    def get_full_data(self) -> pd.DataFrame:
        """Return the full historical DataFrame."""
        return self.df.copy()
