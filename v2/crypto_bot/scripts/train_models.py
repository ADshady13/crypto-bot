"""
Train & Tune 15 XGBoost Models — V2 Triple-Core System

For each of the 5 pairs (BTC, ETH, SOL, BNB, XRP), trains 3 models:
  1. Bull  (y_bull)  — predicts LONG TP hit before SL
  2. Bear  (y_bear)  — predicts SHORT TP hit before SL
  3. Time  (y_time)  — predicts chop / dead market

Split: Train=15000 | Test=2500 | OOT=2500 (strict time-series order)

HP tuning via Optuna (50 trials per model, optimizing AUC-PR).
Final evaluation on OOT set — untouched during tuning.
"""

import os
import sys
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, accuracy_score, f1_score
)
from datetime import datetime

# Suppress optuna logs and XGB warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from strategies.feature_engineering import FeatureEngineer, SHIELD_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")

# ── Config ──────────────────────────────────────────────────────────────
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TARGETS = {
    "bull": "target",
    "bear": "target_short",
    "time": "target_time",
}
TRAIN_SIZE = 15_000
TEST_SIZE = 2_500
OOT_SIZE = 2_500
WARMUP_ROWS = 200  # Skip rows with zero-filled features
N_OPTUNA_TRIALS = 500
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_and_prepare(pair: str, btc_features: pd.DataFrame) -> tuple:
    """Load data, compute features and targets, split into train/test/OOT."""
    path = os.path.join(DATA_DIR, f"{pair}_20k.csv")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    logger.info(f"Loaded {pair}: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.transform(df)
    df = fe.add_macro_features(df, btc_features)
    df = fe.create_targets(df)

    # Drop warmup zone (first 200 rows have zero-filled indicators)
    df = df.iloc[WARMUP_ROWS:]
    logger.info(f"  After warmup drop: {len(df)} rows")

    # Strict time-series split
    # Train: first 15000, Test: next 2500, OOT: last 2500
    train = df.iloc[:TRAIN_SIZE]
    test = df.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    oot = df.iloc[TRAIN_SIZE + TEST_SIZE:TRAIN_SIZE + TEST_SIZE + OOT_SIZE]

    logger.info(f"  Train: {len(train)} ({train.index[0].date()} → {train.index[-1].date()})")
    logger.info(f"  Test:  {len(test)} ({test.index[0].date()} → {test.index[-1].date()})")
    logger.info(f"  OOT:   {len(oot)} ({oot.index[0].date()} → {oot.index[-1].date()})")

    return train, test, oot


def get_xy(df: pd.DataFrame, target_col: str) -> tuple:
    """Extract feature matrix X and target vector y."""
    available = [f for f in SHIELD_FEATURES if f in df.columns]
    X = df[available].values
    y = df[target_col].values
    return X, y, available


def create_objective(X_train, y_train, X_test, y_test, target_name: str):
    """Create an Optuna objective function for HP tuning."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }

        # Handle class imbalance
        pos = y_train.sum()
        neg = len(y_train) - pos
        params["scale_pos_weight"] = neg / max(pos, 1)

        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Optimize for Average Precision (area under PR curve)
        y_proba = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, y_proba)

        return ap

    return objective


def train_final_model(X_train, y_train, X_test, y_test, best_params: dict) -> xgb.XGBClassifier:
    """Train the final model with the best hyperparameters."""
    pos = y_train.sum()
    neg = len(y_train) - pos
    best_params["scale_pos_weight"] = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model


def evaluate_model(model, X, y, set_name: str, target_name: str) -> dict:
    """Evaluate model on a dataset and return metrics."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = 0.0

    ap = average_precision_score(y, y_proba)

    # Hit rate at various thresholds
    hit_rates = {}
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds_at_thresh = (y_proba >= thresh).astype(int)
        n_signals = preds_at_thresh.sum()
        if n_signals > 0:
            hits = (preds_at_thresh & y).sum()
            hit_rate = hits / n_signals * 100
            hit_rates[f"P>{thresh:.2f}"] = f"{hit_rate:.1f}% ({hits}/{n_signals})"
        else:
            hit_rates[f"P>{thresh:.2f}"] = "0 signals"

    metrics = {
        "set": set_name,
        "target": target_name,
        "accuracy": round(acc * 100, 1),
        "f1": round(f1, 3),
        "auc_roc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "pos_rate": round(y.mean() * 100, 1),
        "hit_rates": hit_rates,
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    logger.info(f"  [{metrics['set']}] {metrics['target']} | "
                f"Acc={metrics['accuracy']}% | F1={metrics['f1']} | "
                f"AUC={metrics['auc_roc']} | AvgPrec={metrics['avg_precision']} | "
                f"Pos={metrics['pos_rate']}%")
    for thresh, hr in metrics["hit_rates"].items():
        logger.info(f"    {thresh}: {hr}")


def save_model(model, pair: str, target_name: str, best_params: dict,
               feature_names: list, test_metrics: dict, oot_metrics: dict):
    """Save model and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"xgb_{target_name}_{pair.replace('USDT','')}.json")
    model.save_model(model_path)

    # Save metadata
    meta = {
        "pair": pair,
        "target": target_name,
        "trained_at": datetime.now().isoformat(),
        "feature_count": len(feature_names),
        "features": feature_names,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "oot_metrics": oot_metrics,
        "data_split": {
            "train": TRAIN_SIZE,
            "test": TEST_SIZE,
            "oot": OOT_SIZE,
            "warmup_dropped": WARMUP_ROWS,
        },
        "target_config": {
            "horizon": 6,
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 1.0,
            "fee_pct": 0.001,
        },
    }

    meta_path = model_path.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"  💾 Saved: {os.path.basename(model_path)}")
    return model_path


def train_pair(pair: str, btc_features: pd.DataFrame) -> list:
    """Train all 3 models for a single pair."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  TRAINING: {pair}")
    logger.info(f"{'='*70}")

    train_df, test_df, oot_df = load_and_prepare(pair, btc_features)
    results = []

    for target_name, target_col in TARGETS.items():
        logger.info(f"\n  --- {pair} / {target_name.upper()} ---")

        X_train, y_train, feat_names = get_xy(train_df, target_col)
        X_test, y_test, _ = get_xy(test_df, target_col)
        X_oot, y_oot, _ = get_xy(oot_df, target_col)

        pos_rate = y_train.mean() * 100
        logger.info(f"  Train: {len(y_train)} samples, {pos_rate:.1f}% positive")

        # HP Tuning with Optuna
        logger.info(f"  🔍 Running {N_OPTUNA_TRIALS} Optuna trials...")
        t0 = time.time()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        objective = create_objective(X_train, y_train, X_test, y_test, target_name)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        elapsed = time.time() - t0
        logger.info(f"  ✅ Best trial: AvgPrec={study.best_value:.4f} ({elapsed:.0f}s)")
        logger.info(f"  Best params: {json.dumps(best_params, indent=2)}")

        # Train final model with best params
        model = train_final_model(X_train, y_train, X_test, y_test, best_params)

        # Evaluate on Test
        test_metrics = evaluate_model(model, X_test, y_test, "TEST", target_name)
        print_metrics(test_metrics)

        # Evaluate on OOT (untouched)
        oot_metrics = evaluate_model(model, X_oot, y_oot, "OOT", target_name)
        print_metrics(oot_metrics)

        # Feature importance (top 10)
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        logger.info(f"  Top 10 features:")
        for rank, idx in enumerate(top_idx, 1):
            logger.info(f"    {rank}. {feat_names[idx]:25s} importance={importances[idx]:.4f}")

        # Save
        save_model(model, pair, target_name, best_params, feat_names, test_metrics, oot_metrics)
        results.append({
            "pair": pair,
            "model": target_name,
            "test": test_metrics,
            "oot": oot_metrics,
        })

    return results


def print_summary(all_results: list):
    """Print a final summary table of all 15 models."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  FINAL SUMMARY — ALL 15 MODELS")
    logger.info(f"{'='*70}")

    header = f"{'Pair':<10} {'Model':<6} {'Test Acc':>8} {'Test AUC':>8} {'Test AP':>8} │ {'OOT Acc':>8} {'OOT AUC':>8} {'OOT AP':>8}"
    logger.info(header)
    logger.info("─" * len(header))

    for r in all_results:
        t = r["test"]
        o = r["oot"]
        logger.info(
            f"{r['pair']:<10} {r['model']:<6} "
            f"{t['accuracy']:>7.1f}% {t['auc_roc']:>8.4f} {t['avg_precision']:>8.4f} │ "
            f"{o['accuracy']:>7.1f}% {o['auc_roc']:>8.4f} {o['avg_precision']:>8.4f}"
        )


def main():
    logger.info("=" * 70)
    logger.info("  CryptoBot V2 — Model Training & HP Tuning")
    logger.info(f"  {len(PAIRS)} pairs × {len(TARGETS)} models = {len(PAIRS) * len(TARGETS)} models")
    logger.info(f"  Split: Train={TRAIN_SIZE} | Test={TEST_SIZE} | OOT={OOT_SIZE}")
    logger.info(f"  Features: {len(SHIELD_FEATURES)}")
    logger.info(f"  Optuna trials: {N_OPTUNA_TRIALS} per model")
    logger.info("=" * 70)

    all_results = []
    t_start = time.time()

    # Pre-compute BTC features once for macro indicators
    logger.info("Computing global BTC macro features...")
    btc_path = os.path.join(DATA_DIR, "BTCUSDT_20k.csv")
    btc_raw = pd.read_csv(btc_path, index_col="timestamp", parse_dates=True)
    fe_btc = FeatureEngineer()
    btc_features = fe_btc.transform(btc_raw)

    for pair in PAIRS:
        pair_results = train_pair(pair, btc_features)
        all_results.extend(pair_results)

    total_time = time.time() - t_start
    print_summary(all_results)
    logger.info(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"💾 Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
