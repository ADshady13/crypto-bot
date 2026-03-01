"""
ML Research — XGBoost Classifier for Derivatives Sentiment Trading.

Tasks:
  1. Load cached data → engineer base + ML features → create target labels.
  2. Train XGBoost with 70/30 time-series split (no shuffle).
  3. Analyze feature importance.
  4. Run ML-filtered backtest: sentiment_score > threshold AND xgb_prob > 0.65.
  5. Compare ML-filtered results vs deterministic-only results.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer
from core.signal_generator import BacktestSignalExecutor

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

ML_CONFIG = {
    "target_horizon": 24,       # 24h lookahead
    "target_threshold_pct": 1.0, # +1% = bullish
    "train_ratio": 0.70,
    "xgb_prob_threshold": 0.55,  # ML gate: trade only if P(bullish) > this
    "sentiment_threshold": 0.65, # Existing deterministic threshold
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ml")


# ======================================================================
#  DATA PREPARATION
# ======================================================================
def prepare_dataset(symbol, target_candles=10000):
    """Load cached data, engineer all features, create ML target."""
    print(f"\n{'='*60}")
    print(f"  PREPARING DATASET: {symbol}")
    print(f"{'='*60}")

    fetcher = DerivativesDataFetcher(symbol)
    df = fetcher.fetch_all_data(use_cache=True, target_candles=target_candles)

    if df.empty or len(df) < 500:
        print(f"  ERROR: Insufficient data ({len(df)} rows)")
        return None, None

    # Step 1: Base features
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)

    # Step 2: ML-specific features
    df, ml_feature_names = engineer.engineer_ml_features(df)

    # Step 3: Create target
    df = engineer.create_ml_target(
        df,
        horizon=ML_CONFIG["target_horizon"],
        threshold_pct=ML_CONFIG["target_threshold_pct"],
    )

    # Drop rows without target (last N rows)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    print(f"  Final dataset: {len(df)} rows, {len(ml_feature_names)} features")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    return df, ml_feature_names


# ======================================================================
#  MODEL TRAINING (Task 2)
# ======================================================================
def train_xgboost(df, feature_names, pair_name):
    """Train XGBoost on 70% train, evaluate on 30% test. Time-series split."""
    print(f"\n--- Training XGBoost for {pair_name} ---")

    # Time-series split (NO shuffling)
    split_idx = int(len(df) * ML_CONFIG["train_ratio"])
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_names].values
    y_train = train["target"].values
    X_test = test[feature_names].values
    y_test = test["target"].values

    print(f"  Train: {len(train)} rows ({y_train.sum()}/{len(y_train)} positive = {y_train.mean()*100:.1f}%)")
    print(f"  Test:  {len(test)} rows ({y_test.sum()}/{len(y_test)} positive = {y_test.mean()*100:.1f}%)")

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    print(f"\n  === ML METRICS ({pair_name}) ===")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  AUC:       {auc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Store predictions on test set
    test_copy = test.copy()
    test_copy["ml_prob"] = y_prob
    test_copy["ml_pred"] = y_pred

    ml_metrics = {
        "pair": pair_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "train_size": len(train),
        "test_size": len(test),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
    }

    return model, test_copy, ml_metrics


# ======================================================================
#  FEATURE IMPORTANCE (Task 2)
# ======================================================================
def analyze_feature_importance(model, feature_names, pair_name):
    """Generate and save feature importance analysis."""
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])

    print(f"\n  === FEATURE IMPORTANCE ({pair_name}) ===")
    print(f"  {'Feature':<30} {'Importance':>10}")
    print(f"  {'-'*42}")
    for feat, imp in feat_imp[:15]:
        bar = "█" * int(imp * 100)
        print(f"  {feat:<30} {imp:>10.4f}  {bar}")

    # Categorize features
    categories = {
        "Derivatives (Funding)": ["funding_rate", "funding_zscore", "funding_delta_4h", "funding_delta_24h"],
        "Derivatives (OI)": ["open_interest", "oi_pct_change", "oi_zscore", "oi_delta_4h", "oi_delta_24h"],
        "Positioning (L/S)": ["long_short_ratio", "long_short_ratio_zscore"],
        "Sentiment (F&G)": ["fear_greed_value", "fg_normalized", "fg_zscore"],
        "Volume": ["Volume", "volume_zscore", "volume_delta_4h", "volume_ratio_24h"],
        "Trend/Momentum": ["ema_200_dist", "adx", "roc", "roc_4h", "roc_24h", "roc_72h", "roc_zscore"],
        "Volatility": ["atr", "atr_pct", "atr_pct_zscore"],
        "Composite": ["sentiment_score", "sentiment_raw"],
    }

    cat_importance = {}
    for cat, feats in categories.items():
        total = sum(imp for f, imp in feat_imp if f in feats)
        cat_importance[cat] = total

    print(f"\n  === CATEGORY IMPORTANCE ({pair_name}) ===")
    for cat, imp in sorted(cat_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 100)
        print(f"  {cat:<25} {imp:>8.4f}  {bar}")

    # Save plot
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Feature Importance — {pair_name}", fontsize=14, fontweight="bold")

    # Top 15 features bar chart
    top_feats = feat_imp[:15]
    ax1.barh(
        [f[0] for f in reversed(top_feats)],
        [f[1] for f in reversed(top_feats)],
        color="#4FC3F7",
        edgecolor="#0288D1",
    )
    ax1.set_xlabel("Importance (Gain)")
    ax1.set_title("Top 15 Individual Features")

    # Category pie chart
    cat_labels = list(cat_importance.keys())
    cat_values = list(cat_importance.values())
    colors = ["#FF6B6B", "#FF8E53", "#FFC107", "#4CAF50", "#2196F3", "#9C27B0", "#607D8B", "#E91E63"]
    ax2.pie(cat_values, labels=cat_labels, autopct="%1.1f%%", colors=colors[:len(cat_labels)])
    ax2.set_title("Importance by Category")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"{pair_name}_feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {plot_path}")

    return feat_imp, cat_importance


# ======================================================================
#  ML-FILTERED BACKTEST (Task 3)
# ======================================================================
def run_ml_filtered_backtest(test_df, pair_name):
    """
    Run the deterministic backtest on the test set, but add an ML gate:
    - Trade only if sentiment_score > threshold AND ml_prob > xgb_threshold.
    Compare against the deterministic-only approach.
    """
    print(f"\n--- ML-Filtered Backtest: {pair_name} ---")

    initial_capital = 100000

    # === RUN 1: Deterministic Only ===
    det_params = {
        "sentiment_entry_threshold": ML_CONFIG["sentiment_threshold"],
        "sentiment_exit_threshold": 0.35,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 3.0,
    }
    det_executor = BacktestSignalExecutor(det_params)
    det_trades, det_equity = det_executor.run(test_df, initial_capital)
    det_metrics = _compute_backtest_metrics(det_trades, det_equity, test_df, initial_capital)
    det_metrics["mode"] = "Deterministic Only"
    print(f"\n  [Deterministic] Return: {det_metrics['return_pct']:+.2f}%, "
          f"Trades: {det_metrics['num_trades']}, MaxDD: {det_metrics['max_dd']:.2f}%")

    # === RUN 2: ML-Filtered (inject ml_prob as a gate) ===
    # We modify the test_df to mask sentiment_score where ml_prob is too low
    ml_test = test_df.copy()
    xgb_threshold = ML_CONFIG["xgb_prob_threshold"]

    # Suppress the sentiment score where ML says "don't trade"
    ml_mask = ml_test["ml_prob"] < xgb_threshold
    ml_test.loc[ml_mask, "sentiment_score"] = 0.0  # Force below entry threshold

    ml_executor = BacktestSignalExecutor(det_params)
    ml_trades, ml_equity = ml_executor.run(ml_test, initial_capital)
    ml_metrics = _compute_backtest_metrics(ml_trades, ml_equity, ml_test, initial_capital)
    ml_metrics["mode"] = f"ML-Filtered (P>{xgb_threshold})"
    print(f"  [ML-Filtered]   Return: {ml_metrics['return_pct']:+.2f}%, "
          f"Trades: {ml_metrics['num_trades']}, MaxDD: {ml_metrics['max_dd']:.2f}%")

    # === RUN 3: Buy & Hold ===
    bh_return = (test_df["Close"].iloc[-1] / test_df["Close"].iloc[0] - 1) * 100
    print(f"  [Buy & Hold]    Return: {bh_return:+.2f}%")

    return det_metrics, ml_metrics, bh_return


def _compute_backtest_metrics(trades_df, equity_df, df, initial_capital):
    """Compute standard backtest metrics."""
    if equity_df.empty:
        return {
            "return_pct": 0, "max_dd": 0, "num_trades": 0,
            "win_rate": 0, "profit_factor": 0, "sharpe": 0,
        }

    final_eq = equity_df["equity"].iloc[-1]
    ret = (final_eq / initial_capital - 1) * 100

    # Max drawdown
    cummax = equity_df["equity"].cummax()
    dd = (equity_df["equity"] - cummax) / cummax
    max_dd = dd.min() * 100

    # Trade stats
    n = len(trades_df)
    if n > 0 and "pnl" in trades_df.columns:
        wins = (trades_df["pnl"] > 0).sum()
        win_rate = wins / n * 100
        gross_p = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_l = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        pf = gross_p / gross_l if gross_l > 0 else 0
    else:
        wins = 0
        win_rate = 0
        pf = 0

    # Sharpe
    rets = equity_df["equity"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(24 * 365)) if len(rets) > 0 and rets.std() > 0 else 0

    return {
        "return_pct": ret,
        "final_equity": final_eq,
        "max_dd": max_dd,
        "num_trades": n,
        "winning_trades": wins,
        "win_rate": win_rate,
        "profit_factor": pf,
        "sharpe": sharpe,
    }


# ======================================================================
#  MAIN PIPELINE
# ======================================================================
def run_pair_research(pair_name, symbol):
    """Full ML research pipeline for a single pair."""
    # 1. Prepare data
    df, feature_names = prepare_dataset(symbol)
    if df is None:
        return None

    # 2. Train model
    model, test_df, ml_metrics = train_xgboost(df, feature_names, pair_name)

    # 3. Analyze feature importance
    feat_imp, cat_imp = analyze_feature_importance(model, feature_names, pair_name)

    # 4. Run ML-filtered backtest
    det_metrics, ml_filtered_metrics, bh_return = run_ml_filtered_backtest(test_df, pair_name)

    return {
        "pair": pair_name,
        "symbol": symbol,
        "ml_metrics": ml_metrics,
        "det_backtest": det_metrics,
        "ml_backtest": ml_filtered_metrics,
        "buy_hold_return": bh_return,
        "feature_importance": feat_imp[:10],
        "category_importance": cat_imp,
    }


def print_summary(all_results):
    """Print the grand comparison table."""
    print("\n" + "=" * 100)
    print("ML RESEARCH SUMMARY — XGBoost as Trade Filter")
    print("=" * 100)

    # ML Model Performance
    print(f"\n{'Pair':<6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TrainPos%':>10} {'TestPos%':>10}")
    print("-" * 60)
    for r in all_results:
        m = r["ml_metrics"]
        print(f"{m['pair']:<6} {m['auc']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} "
              f"{m['train_positive_rate']*100:>9.1f}% {m['test_positive_rate']*100:>9.1f}%")

    # Backtest Comparison
    print(f"\n{'='*100}")
    print(f"{'Pair':<6} {'Mode':<25} {'Return%':>10} {'MaxDD%':>8} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Sharpe':>7}")
    print("-" * 100)
    for r in all_results:
        for mode, metrics in [("Deterministic", r["det_backtest"]), ("ML-Filtered", r["ml_backtest"])]:
            print(f"{r['pair']:<6} {metrics['mode']:<25} {metrics['return_pct']:>+10.2f} "
                  f"{metrics['max_dd']:>8.2f} {metrics['num_trades']:>7} "
                  f"{metrics['win_rate']:>7.1f}% {metrics['profit_factor']:>6.2f} "
                  f"{metrics['sharpe']:>7.2f}")
        print(f"{r['pair']:<6} {'Buy & Hold':<25} {r['buy_hold_return']:>+10.2f}")
        print("-" * 100)

    # Averages
    if all_results:
        avg_det = np.mean([r["det_backtest"]["return_pct"] for r in all_results])
        avg_ml = np.mean([r["ml_backtest"]["return_pct"] for r in all_results])
        avg_bh = np.mean([r["buy_hold_return"] for r in all_results])
        avg_det_dd = np.mean([r["det_backtest"]["max_dd"] for r in all_results])
        avg_ml_dd = np.mean([r["ml_backtest"]["max_dd"] for r in all_results])

        print(f"\n{'AVERAGES':}")
        print(f"  Deterministic:  {avg_det:+.2f}% return, {avg_det_dd:.2f}% max DD")
        print(f"  ML-Filtered:    {avg_ml:+.2f}% return, {avg_ml_dd:.2f}% max DD")
        print(f"  Buy & Hold:     {avg_bh:+.2f}%")
        print(f"  ML Improvement: {avg_ml - avg_det:+.2f}% vs deterministic")

    # Top features across all pairs
    print(f"\n{'='*100}")
    print("TOP FEATURES (aggregated across pairs)")
    print("-" * 60)
    agg_imp = {}
    for r in all_results:
        for feat, imp in r["feature_importance"]:
            agg_imp[feat] = agg_imp.get(feat, 0) + imp
    for feat, imp in sorted(agg_imp.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * int(imp / len(all_results) * 100)
        print(f"  {feat:<30} {imp/len(all_results):>8.4f}  {bar}")

    print("=" * 100)


def save_results(all_results):
    """Save ML research results to disk."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Summary CSV
    rows = []
    for r in all_results:
        row = {
            "pair": r["pair"],
            "auc": r["ml_metrics"]["auc"],
            "precision": r["ml_metrics"]["precision"],
            "recall": r["ml_metrics"]["recall"],
            "f1": r["ml_metrics"]["f1"],
            "det_return": r["det_backtest"]["return_pct"],
            "det_max_dd": r["det_backtest"]["max_dd"],
            "det_trades": r["det_backtest"]["num_trades"],
            "det_win_rate": r["det_backtest"]["win_rate"],
            "ml_return": r["ml_backtest"]["return_pct"],
            "ml_max_dd": r["ml_backtest"]["max_dd"],
            "ml_trades": r["ml_backtest"]["num_trades"],
            "ml_win_rate": r["ml_backtest"]["win_rate"],
            "buy_hold": r["buy_hold_return"],
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ml_research_summary.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")


# ======================================================================
#  ENTRY POINT
# ======================================================================
def main():
    print("=" * 60)
    print("  XGBOOST ML RESEARCH — DERIVATIVES SENTIMENT STRATEGY")
    print("=" * 60)
    print(f"Config: {json.dumps(ML_CONFIG, indent=2)}")

    all_results = []
    for pair_name, symbol in PAIRS.items():
        try:
            result = run_pair_research(pair_name, symbol)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR for {pair_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
