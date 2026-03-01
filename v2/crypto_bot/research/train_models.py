"""
Train Bull & Bear XGBoost models for the Dual-Core strategy.

Bull Model: Predicts P(price rises > +1% in 24h)  → saved as models/xgb_bull_{pair}.json
Bear Model: Predicts P(price drops > -2% in 24h)  → saved as models/xgb_bear_{pair}.json

Uses only the proven "Shield" features:
  ROC (multi-timeframe), Fear & Greed, Funding Rate, 200-EMA Dist, ATR, ADX
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================
#  CONFIG
# ======================================================================
PAIRS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}

# Proven "Shield" features — only those with good data coverage
SHIELD_FEATURES = [
    "roc", "roc_4h", "roc_24h", "roc_72h", "roc_zscore",
    "fear_greed_value", "fg_normalized", "fg_zscore",
    "funding_rate", "funding_zscore", "funding_delta_4h", "funding_delta_24h",
    "ema_200_dist", "adx",
    "atr", "atr_pct", "atr_pct_zscore",
    "volume_zscore", "volume_delta_4h", "volume_ratio_24h",
    "sentiment_score", "sentiment_raw",
]

TRAIN_RATIO = 0.70
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ======================================================================
#  DATA PREPARATION
# ======================================================================
def prepare_dataset(symbol):
    """Load cached data, engineer features, create both targets."""
    print(f"\n{'='*60}")
    print(f"  PREPARING: {symbol}")
    print(f"{'='*60}")

    fetcher = DerivativesDataFetcher(symbol)
    df = fetcher.fetch_all_data(use_cache=True, target_candles=10000)
    if df.empty or len(df) < 500:
        return None, None

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)
    df, all_ml_features = engineer.engineer_ml_features(df)

    # Create BOTH targets
    df = engineer.create_ml_target(df, horizon=24, threshold_pct=1.0)       # Long: +1% in 24h
    df = engineer.create_ml_target_short(df, horizon=24, threshold_pct=2.0) # Short: -2% in 24h

    # Filter to Shield features only
    available = [f for f in SHIELD_FEATURES if f in df.columns]
    print(f"  Shield features: {len(available)}/{len(SHIELD_FEATURES)}")

    # Drop rows without target
    df = df.dropna(subset=["target_return"])
    df["target"] = df["target"].astype(int)
    df["target_short"] = df["target_short"].astype(int)

    return df, available


# ======================================================================
#  TRAINING
# ======================================================================
def train_model(df, feature_names, target_col, model_name, pair_name):
    """Train XGBoost on the given target column and save model."""
    print(f"\n--- Training {model_name} for {pair_name} ---")

    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_names].values
    y_train = train[target_col].values
    X_test = test[feature_names].values
    y_test = test[target_col].values

    pos_rate_train = y_train.mean()
    pos_rate_test = y_test.mean()
    print(f"  Train: {len(train)} rows ({y_train.sum()}/{len(y_train)} positive = {pos_rate_train*100:.1f}%)")
    print(f"  Test:  {len(test)} rows ({y_test.sum()}/{len(y_test)} positive = {pos_rate_test*100:.1f}%)")

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
        min_child_weight=5,
        gamma=0.1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{pair_name}.json")
    model.save_model(model_path)
    print(f"  Saved: {model_path}")

    # Save feature names for later loading
    meta_path = os.path.join(MODELS_DIR, f"{model_name}_{pair_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"features": feature_names, "target": target_col, "pair": pair_name}, f)

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    print(f"  Top 5 features:")
    for feat, imp in feat_imp[:5]:
        bar = "█" * int(imp * 100)
        print(f"    {feat:<25} {imp:.4f} {bar}")

    return model, {
        "pair": pair_name,
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "train_pos_rate": pos_rate_train,
        "test_pos_rate": pos_rate_test,
        "train_size": len(train),
        "test_size": len(test),
    }


# ======================================================================
#  MAIN
# ======================================================================
def main():
    print("=" * 60)
    print("  DUAL-CORE MODEL TRAINING — Bull & Bear XGBoost")
    print("=" * 60)

    all_metrics = []

    for pair_name, symbol in PAIRS.items():
        df, features = prepare_dataset(symbol)
        if df is None:
            continue

        # Train Bull model (Long target: +1% in 24h)
        bull_model, bull_metrics = train_model(
            df, features, "target", "xgb_bull", pair_name
        )
        all_metrics.append(bull_metrics)

        # Train Bear model (Short target: -2% in 24h)
        bear_model, bear_metrics = train_model(
            df, features, "target_short", "xgb_bear", pair_name
        )
        all_metrics.append(bear_metrics)

    # Summary
    print(f"\n{'='*80}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Pair':<6} {'Model':<12} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TestPos%':>10}")
    print(f"  {'-'*55}")
    for m in all_metrics:
        print(f"  {m['pair']:<6} {m['model']:<12} {m['auc']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['test_pos_rate']*100:>9.1f}%")

    print(f"\n  Models saved to: {MODELS_DIR}/")
    print(f"  Files per pair: xgb_bull_{{PAIR}}.json, xgb_bear_{{PAIR}}.json")


if __name__ == "__main__":
    main()
