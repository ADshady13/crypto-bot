"""
Inference — Hot-Reloading ModelLoader for Dual-Core XGBoost.

The ModelLoader monitors the model files on disk and automatically
reloads them if retrain.py has promoted a new Champion. This enables
zero-downtime model updates.

Mechanism:
  - Stores self.last_modified_time for each model file
  - Before every prediction, checks os.path.getmtime(model_path)
  - If timestamp changed → reload the Booster immediately
"""

import os
import logging
from typing import Optional

import numpy as np
from xgboost import XGBClassifier

from core.config import Config
from strategies.feature_engineering import FeatureEngineer, SHIELD_FEATURES

logger = logging.getLogger("crypto_bot")


class ModelLoader:
    """
    Hot-reloading Dual-Core model manager.

    Loads Bull and Bear XGBoost models per pair, automatically
    detects when retrain.py has updated them on disk, and reloads
    without requiring a bot restart.
    """

    def __init__(self, pair: str):
        self.pair = pair
        self.bull_path = os.path.join(Config.MODEL_DIR, f"xgb_bull_{pair}.json")
        self.bear_path = os.path.join(Config.MODEL_DIR, f"xgb_bear_{pair}.json")

        self.bull_model: Optional[XGBClassifier] = None
        self.bear_model: Optional[XGBClassifier] = None

        self.bull_mtime: float = 0.0
        self.bear_mtime: float = 0.0

        self.feature_engineer = FeatureEngineer()

        # Initial load
        self._load_bull()
        self._load_bear()

    @property
    def ready(self) -> bool:
        """True if both models are loaded and ready for inference."""
        return self.bull_model is not None and self.bear_model is not None

    # ------------------------------------------------------------------ #
    #  Model Loading
    # ------------------------------------------------------------------ #
    def _load_bull(self):
        """Load or reload the Bull model."""
        if not os.path.exists(self.bull_path):
            logger.warning(f"Bull model not found: {self.bull_path}")
            return

        try:
            model = XGBClassifier()
            model.load_model(self.bull_path)
            self.bull_model = model
            self.bull_mtime = os.path.getmtime(self.bull_path)
            logger.info(f"Bull model loaded: {self.bull_path} (mtime={self.bull_mtime:.0f})")
        except Exception as e:
            logger.error(f"Failed to load Bull model: {e}")

    def _load_bear(self):
        """Load or reload the Bear model."""
        if not os.path.exists(self.bear_path):
            logger.warning(f"Bear model not found: {self.bear_path}")
            return

        try:
            model = XGBClassifier()
            model.load_model(self.bear_path)
            self.bear_model = model
            self.bear_mtime = os.path.getmtime(self.bear_path)
            logger.info(f"Bear model loaded: {self.bear_path} (mtime={self.bear_mtime:.0f})")
        except Exception as e:
            logger.error(f"Failed to load Bear model: {e}")

    # ------------------------------------------------------------------ #
    #  Hot-Reload Check
    # ------------------------------------------------------------------ #
    def _check_reload(self):
        """
        Check if model files have been updated on disk.
        If retrain.py promoted a new Champion, the mtime will differ.
        """
        reloaded = False

        if os.path.exists(self.bull_path):
            current_mtime = os.path.getmtime(self.bull_path)
            if current_mtime != self.bull_mtime:
                logger.info("🔄 Bull model changed on disk — hot-reloading...")
                self._load_bull()
                reloaded = True

        if os.path.exists(self.bear_path):
            current_mtime = os.path.getmtime(self.bear_path)
            if current_mtime != self.bear_mtime:
                logger.info("🔄 Bear model changed on disk — hot-reloading...")
                self._load_bear()
                reloaded = True

        if reloaded:
            logger.info("✅ Models hot-reloaded successfully")

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #
    def predict(self, df_window) -> tuple:
        """
        Generate Bull and Bear probabilities for the latest data point.

        Args:
            df_window: DataFrame with at least 200 rows of OHLCV + indicators

        Returns:
            (bull_prob, bear_prob) — probabilities in [0, 1]
            Returns (0.5, 0.5) if models aren't ready (neutral → FLAT signal)
        """
        # Hot-reload check — runs before every prediction
        self._check_reload()

        if not self.ready:
            logger.warning("Models not ready — returning neutral (0.5, 0.5)")
            return 0.5, 0.5

        # Feature engineering
        df_feat = self.feature_engineer.transform(df_window.copy(), verbose=False)

        # Extract features for the LAST row only
        available = [f for f in SHIELD_FEATURES if f in df_feat.columns]
        X = df_feat[available].fillna(0).values[-1:, :]  # Shape: (1, n_features)

        try:
            bull_prob = float(self.bull_model.predict_proba(X)[:, 1][0])
            bear_prob = float(self.bear_model.predict_proba(X)[:, 1][0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5, 0.5

        logger.debug(f"Prediction: Bull={bull_prob:.3f} Bear={bear_prob:.3f}")
        return bull_prob, bear_prob

    def predict_batch(self, df) -> tuple:
        """
        Generate Bull and Bear probability arrays for the entire DataFrame.
        Used by retrain.py for backtesting.

        Returns:
            (bull_probs, bear_probs) — numpy arrays
        """
        self._check_reload()

        if not self.ready:
            n = len(df)
            return np.full(n, 0.5), np.full(n, 0.5)

        df_feat = self.feature_engineer.transform(df.copy(), verbose=False)
        available = [f for f in SHIELD_FEATURES if f in df_feat.columns]
        X = df_feat[available].fillna(0).values

        try:
            bull_probs = self.bull_model.predict_proba(X)[:, 1]
            bear_probs = self.bear_model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            n = len(df)
            return np.full(n, 0.5), np.full(n, 0.5)

        return bull_probs, bear_probs
