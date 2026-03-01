"""
Inference — Static ModelLoader for V3 Triple-Core XGBoost.

Models are loaded once on startup. Hot-reloading has been removed
per V3 architectural guidelines. If models need retraining, the bot
must be restarted manually.
"""

import os
import logging
from typing import Optional, Dict

import numpy as np
from xgboost import XGBClassifier

from core.config import Config
from strategies.feature_engineering import FeatureEngineer, SHIELD_FEATURES

logger = logging.getLogger("crypto_bot")


class ModelLoader:
    """
    Static Triple-Core model manager.
    Loads Bull, Bear, and Time XGBoost models per pair on initialization.
    """

    def __init__(self, pairs: list[str]):
        """
        Args:
            pairs: List of trading pairs (e.g. ["BTCUSDT", "ETHUSDT", ...])
        """
        self.pairs = pairs
        self.models: Dict[str, Dict[str, XGBClassifier]] = {p: {} for p in pairs}
        self.feature_engineer = FeatureEngineer()
        
        # Load all models for all pairs immediately
        self._load_all_models()

    def _load_all_models(self):
        """Load all three models (bull, bear, time) for every configured pair."""
        for pair in self.pairs:
            sym = pair.replace("USDT", "")
            
            for target in ["bull", "bear", "time"]:
                path = os.path.join(Config.MODEL_DIR, f"xgb_{target}_{sym}.json")
                if not os.path.exists(path):
                    logger.warning(f"Model not found: {path}")
                    continue
                
                try:
                    model = XGBClassifier()
                    model.load_model(path)
                    self.models[pair][target] = model
                except Exception as e:
                    logger.error(f"Failed to load {target} model for {pair}: {e}")
                    
            loaded_count = len(self.models[pair])
            if loaded_count == 3:
                logger.info(f"✅ All 3 models loaded for {pair}")
            else:
                logger.error(f"❌ Only {loaded_count}/3 models loaded for {pair}")


    def is_ready(self, pair: str) -> bool:
        """True if all 3 models are loaded and ready for inference for a given pair."""
        return len(self.models.get(pair, {})) == 3

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #
    def predict(self, df_window, pair: str, btc_features=None) -> tuple:
        """
        Generate Bull, Bear, and Time probabilities for the latest data point.

        Args:
            df_window: DataFrame with at least 200 rows of OHLCV + indicators
            pair: The symbol being predicted (e.g. "ETHUSDT")
            btc_features: Globally computed BTC features specifically passed in for the new Macro setup

        Returns:
            (bull_prob, bear_prob, time_prob) — probabilities in [0, 1]
            Returns (0.0, 0.0, 1.0) if models aren't ready (neutral → FLAT signal forced by chop)
        """
        if not self.is_ready(pair):
            logger.warning(f"Models not ready for {pair} — returning neutral (0, 0, 1)")
            return 0.0, 0.0, 1.0

        # Feature engineering
        df_feat = self.feature_engineer.transform(df_window.copy(), verbose=False)
        
        # If this isn't BTC, and we have BTC features provided, inject them as macros
        if pair != "BTCUSDT" and btc_features is not None:
             df_feat = self.feature_engineer.add_macro_features(df_feat, btc_features)

        # Extract features for the LAST row only
        available = [f for f in SHIELD_FEATURES if f in df_feat.columns]
        X = df_feat[available].fillna(0).values[-1:, :]  # Shape: (1, n_features)

        try:
            m_bull = self.models[pair]["bull"]
            m_bear = self.models[pair]["bear"]
            m_time = self.models[pair]["time"]
            
            bull_prob = float(m_bull.predict_proba(X)[:, 1][0])
            bear_prob = float(m_bear.predict_proba(X)[:, 1][0])
            time_prob = float(m_time.predict_proba(X)[:, 1][0])
        except Exception as e:
            logger.error(f"Prediction failed for {pair}: {e}")
            return 0.0, 0.0, 1.0

        logger.debug(f"[{pair}] Preds: Bull={bull_prob:.3f} Bear={bear_prob:.3f} Chop={time_prob:.3f}")
        return bull_prob, bear_prob, time_prob
