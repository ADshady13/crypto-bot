"""
Feature Engineering — Shared between training and live inference.

This is the SINGLE SOURCE OF TRUTH for feature computation.
Both retrain.py and inference.py import from here to ensure
training and live features are computed identically.

Features (the "Shield" set — proven in walk-forward validation):
  - ROC (multi-timeframe: 4h, 12h, 24h, 72h)
  - Fear & Greed (normalized, z-score)
  - Funding Rate (z-score, deltas)
  - 200-EMA Distance
  - ADX (trend strength)
  - ATR (volatility, % of price, z-score)
  - Volume (z-score, delta, ratio)
  - Composite Sentiment Score
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("crypto_bot")

# The exact feature set used for ML inference — order matters for model input
SHIELD_FEATURES = [
    "roc", "roc_4h", "roc_24h", "roc_72h", "roc_zscore",
    "fear_greed_value", "fg_normalized", "fg_zscore",
    "funding_rate", "funding_zscore", "funding_delta_4h", "funding_delta_24h",
    "ema_200_dist", "adx",
    "atr", "atr_pct", "atr_pct_zscore",
    "volume_zscore", "volume_delta_4h", "volume_ratio_24h",
    "sentiment_score", "sentiment_raw",
]


class FeatureEngineer:
    """Compute all technical features for the Dual-Core strategy."""

    def __init__(self, params=None):
        self.params = params or {
            "zscore_period": 30,
            "funding_percentile_period": 90,
            "atr_period": 14,
            "ema_period": 50,
            "roc_period": 12,
            "fg_zscore_period": 30,
            "volume_zscore_period": 30,
        }

    # ================================================================== #
    #  Core statistical functions
    # ================================================================== #
    @staticmethod
    def zscore(series: pd.Series, period: int = 30) -> pd.Series:
        """Rolling z-score."""
        mu = series.rolling(window=period, min_periods=1).mean()
        sigma = series.rolling(window=period, min_periods=1).std().replace(0, 1)
        return (series - mu) / sigma

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def roc(series: pd.Series, period: int) -> pd.Series:
        """Rate of Change (% change over N periods)."""
        return series.pct_change(periods=period) * 100

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        hl = high - low
        hc = np.abs(high - close.shift())
        lc = np.abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index — trend strength."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_val = FeatureEngineer.atr(high, low, close, period).replace(0, 1)
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
        return dx.ewm(span=period, adjust=False).mean()

    # ================================================================== #
    #  Main feature pipeline
    # ================================================================== #
    def transform(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Compute ALL features from raw OHLCV + funding + fear_greed data.
        This is the single entry point for both training and live inference.

        Input columns required: Open, High, Low, Close, Volume, funding_rate, fear_greed_value
        Output: DataFrame with all Shield features added.
        """
        if verbose:
            logger.info("Engineering features...")

        # --- Funding ---
        if "funding_rate" in df.columns:
            df["funding_zscore"] = self.zscore(df["funding_rate"], self.params["zscore_period"])
            df["funding_delta_4h"] = df["funding_rate"].diff(4)
            df["funding_delta_24h"] = df["funding_rate"].diff(24)

        # --- Fear & Greed ---
        if "fear_greed_value" in df.columns:
            df["fg_normalized"] = (df["fear_greed_value"] - 50) / 50
            fg_zp = self.params.get("fg_zscore_period", 30) * 24
            df["fg_zscore"] = self.zscore(df["fear_greed_value"].astype(float), fg_zp)

        # --- Volatility ---
        df["atr"] = self.atr(df["High"], df["Low"], df["Close"], self.params["atr_period"])
        df["atr_pct"] = df["atr"] / df["Close"]
        df["atr_pct_zscore"] = self.zscore(df["atr_pct"], 30)

        # --- Trend ---
        ema_200 = self.ema(df["Close"], 200)
        df["ema_200"] = ema_200
        df["ema_200_dist"] = ((df["Close"] - ema_200) / ema_200) * 100
        df["adx"] = self.adx(df["High"], df["Low"], df["Close"], 14)

        # --- Momentum (multi-timeframe ROC) ---
        df["roc"] = self.roc(df["Close"], self.params.get("roc_period", 12))
        df["roc_4h"] = self.roc(df["Close"], 4)
        df["roc_24h"] = self.roc(df["Close"], 24)
        df["roc_72h"] = self.roc(df["Close"], 72)
        df["roc_zscore"] = self.zscore(df["roc"], self.params["zscore_period"])

        # --- Volume ---
        if "Volume" in df.columns:
            df["volume_zscore"] = self.zscore(df["Volume"], self.params["volume_zscore_period"])
            df["volume_delta_4h"] = df["Volume"].pct_change(4) * 100
            df["volume_ratio_24h"] = df["Volume"] / df["Volume"].rolling(24, min_periods=1).mean()

        # --- Composite Sentiment Score ---
        df = self._compute_sentiment(df)

        # Fill NaN
        df = df.fillna(0)

        if verbose:
            avail = [f for f in SHIELD_FEATURES if f in df.columns]
            logger.info(f"Features ready: {len(avail)}/{len(SHIELD_FEATURES)}")

        return df

    def _compute_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite sentiment score (0-1 normalized)."""
        funding_c = (-1 * df.get("funding_zscore", 0)) * 0.15
        fg_c = (-1 * df.get("fg_zscore", 0)) * 0.10
        roc_c = df.get("roc_zscore", 0) * 0.25

        # Volume during breakout
        if "volume_zscore" in df.columns:
            close = df["Close"]
            high_20 = close.rolling(20).max().shift(1)
            breakout_up = close > high_20
            vol_c = df["volume_zscore"].where(breakout_up, 0).fillna(0) * 0.20
        else:
            vol_c = 0

        raw = funding_c + fg_c + roc_c + vol_c
        if isinstance(raw, (int, float)):
            raw = pd.Series(0, index=df.index)

        r_min = raw.rolling(500, min_periods=50).min()
        r_max = raw.rolling(500, min_periods=50).max()
        r_range = (r_max - r_min).replace(0, 1)

        df["sentiment_score"] = ((raw - r_min) / r_range).clip(0, 1).fillna(0.5)
        df["sentiment_raw"] = raw
        return df

    # ================================================================== #
    #  Target creation (for training only)
    # ================================================================== #
    def create_targets(self, df: pd.DataFrame, bull_horizon=24, bull_thresh=1.0,
                       bear_horizon=24, bear_thresh=2.0) -> pd.DataFrame:
        """
        Create both Bull and Bear binary targets.
        - target:       1 if price rises >= bull_thresh% in bull_horizon hours
        - target_short: 1 if price drops >= bear_thresh% in bear_horizon hours
        """
        future_ret = df["Close"].shift(-bull_horizon) / df["Close"] - 1
        df["target_return"] = future_ret * 100
        df["target"] = (future_ret >= bull_thresh / 100).astype(int)
        df["target_short"] = (future_ret <= -bear_thresh / 100).astype(int)
        return df

    # ================================================================== #
    #  Extract feature matrix for ML
    # ================================================================== #
    @staticmethod
    def get_feature_matrix(df: pd.DataFrame) -> tuple:
        """
        Extract the feature matrix X and available feature names.
        Only includes features that exist in the DataFrame.
        """
        available = [f for f in SHIELD_FEATURES if f in df.columns]
        X = df[available].fillna(0).values
        return X, available
