"""
Walk-Forward Validation (WFO) — Dual-Core XGBoost Strategy

Purpose: Rule out overfitting by retraining fresh models on expanding windows
and stitching together truly out-of-sample predictions.

Logic:
  1. Start with initial_train = 90 days (2160 hourly candles)
  2. Train fresh Bull + Bear XGBoost on [0 : train_end]
  3. Generate signals on test window [train_end : train_end + step_days]
  4. Record PnL for that blind 30-day window
  5. Advance by step_days, repeat until data ends
  6. Stitch all test windows → true OOS equity curve

Also includes a "Reverse Polarity" stress test to check if the model
learned universal momentum or just "sell everything."
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")

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

INITIAL_TRAIN_DAYS = 90      # 90 days = 2160 hourly candles
STEP_DAYS = 30               # 30-day test windows
HOURS_PER_DAY = 24
INITIAL_TRAIN_H = INITIAL_TRAIN_DAYS * HOURS_PER_DAY
STEP_H = STEP_DAYS * HOURS_PER_DAY

INITIAL_CAPITAL = 100_000
FEE_PCT = 0.001
SLIPPAGE_PCT = 0.0005
SL_PCT = 0.03
FUNDING_INTERVAL = 8

# Dual-Core thresholds
BULL_ENTRY = 0.70
BEAR_ENTRY = 0.70
BULL_BLOCK = 0.30
BEAR_BLOCK = 0.30

# Shield features (proven set)
SHIELD_FEATURES = [
    "roc", "roc_4h", "roc_24h", "roc_72h", "roc_zscore",
    "fear_greed_value", "fg_normalized", "fg_zscore",
    "funding_rate", "funding_zscore", "funding_delta_4h", "funding_delta_24h",
    "ema_200_dist", "adx",
    "atr", "atr_pct", "atr_pct_zscore",
    "volume_zscore", "volume_delta_4h", "volume_ratio_24h",
    "sentiment_score", "sentiment_raw",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "wfo")


# ======================================================================
#  DATA PREPARATION
# ======================================================================
def prepare_full_dataset(symbol):
    """Load data, engineer ALL features + both targets. Return full df."""
    fetcher = DerivativesDataFetcher(symbol)
    df = fetcher.fetch_all_data(use_cache=True)
    if df.empty:
        return None, None

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)
    df, _ = engineer.engineer_ml_features(df)
    df = engineer.create_ml_target(df, horizon=24, threshold_pct=1.0)
    df = engineer.create_ml_target_short(df, horizon=24, threshold_pct=2.0)

    # Available shield features
    available = [f for f in SHIELD_FEATURES if f in df.columns]

    # Drop rows with NaN targets at tail
    df = df.dropna(subset=["target", "target_short"])
    df["target"] = df["target"].astype(int)
    df["target_short"] = df["target_short"].astype(int)
    df = df.fillna(0)

    return df, available


def train_dual_models(df_train, features):
    """Train fresh Bull + Bear XGBoost on the given training data."""
    X = df_train[features].values
    y_bull = df_train["target"].values
    y_bear = df_train["target_short"].values

    def train_one(X_train, y_train):
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, use_label_encoder=False,
            verbosity=0, min_child_weight=5, gamma=0.1,
        )
        model.fit(X_train, y_train, verbose=False)
        return model

    bull_model = train_one(X, y_bull)
    bear_model = train_one(X, y_bear)
    return bull_model, bear_model


# ======================================================================
#  PER-WINDOW SIMULATION
# ======================================================================
def simulate_window(df_window, bull_probs, bear_probs, capital_in):
    """
    Run Dual-Core simulation on a single test window.
    Returns final capital, list of trades, and per-hour equity.
    """
    capital = capital_in
    position = 0
    entry_price = 0.0
    trade_capital = 0.0
    trades = []
    equity = []

    for i in range(len(df_window)):
        row = df_window.iloc[i]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        funding_rate = row.get("funding_rate", 0)
        bp = bull_probs[i]
        sp = bear_probs[i]

        # Signal
        if bp > BULL_ENTRY and sp < BEAR_BLOCK:
            action = "long"
        elif sp > BEAR_ENTRY and bp < BULL_BLOCK:
            action = "short"
        else:
            action = "flat"

        # Funding
        if position != 0 and i % FUNDING_INTERVAL == 0:
            if position == 1:
                capital -= trade_capital * funding_rate
            elif position == -1:
                capital += trade_capital * funding_rate

        # Stop loss
        if position == 1 and low <= entry_price * (1 - SL_PCT):
            exit_price = entry_price * (1 - SL_PCT)
            pnl = trade_capital * (exit_price / entry_price - 1) - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append({"dir": "L", "pnl": pnl, "exit": "SL"})
            position = 0; trade_capital = 0

        elif position == -1 and high >= entry_price * (1 + SL_PCT):
            exit_price = entry_price * (1 + SL_PCT)
            pnl = trade_capital * (entry_price - exit_price) / entry_price - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append({"dir": "S", "pnl": pnl, "exit": "SL"})
            position = 0; trade_capital = 0

        # Signal exit
        if position == 1 and action != "long":
            pnl = trade_capital * (close / entry_price - 1) - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append({"dir": "L", "pnl": pnl, "exit": "SIG"})
            position = 0; trade_capital = 0

        elif position == -1 and action != "short":
            pnl = trade_capital * (entry_price - close) / entry_price - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append({"dir": "S", "pnl": pnl, "exit": "SIG"})
            position = 0; trade_capital = 0

        # Entry
        if position == 0 and action == "long":
            entry_cost = capital * (FEE_PCT + SLIPPAGE_PCT)
            trade_capital = capital - entry_cost
            capital = 0
            entry_price = close
            position = 1
        elif position == 0 and action == "short":
            entry_cost = capital * (FEE_PCT + SLIPPAGE_PCT)
            trade_capital = capital - entry_cost
            capital = 0
            entry_price = close
            position = -1

        # Equity
        if position == 1:
            eq = capital + trade_capital + trade_capital * (close / entry_price - 1)
        elif position == -1:
            eq = capital + trade_capital + trade_capital * (entry_price - close) / entry_price
        else:
            eq = capital
        equity.append(eq)

    # Close at end of window
    if position != 0:
        final_close = df_window.iloc[-1]["Close"]
        if position == 1:
            pnl = trade_capital * (final_close / entry_price - 1)
        else:
            pnl = trade_capital * (entry_price - final_close) / entry_price
        pnl -= trade_capital * (FEE_PCT + SLIPPAGE_PCT)
        capital += trade_capital + pnl
        trades.append({"dir": "L" if position == 1 else "S", "pnl": pnl, "exit": "EOW"})

    return capital, trades, equity


# ======================================================================
#  WALK-FORWARD ENGINE
# ======================================================================
def walk_forward(pair_name, symbol):
    """Run walk-forward validation for a single pair."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: {pair_name} ({symbol})")
    print(f"{'='*70}")

    df, features = prepare_full_dataset(symbol)
    if df is None:
        return None

    # Need at least initial_train + 1 step
    min_rows = INITIAL_TRAIN_H + STEP_H
    if len(df) < min_rows:
        print(f"  Not enough data: {len(df)} < {min_rows}")
        return None

    print(f"  Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Windows: train={INITIAL_TRAIN_DAYS}d, step={STEP_DAYS}d")
    print(f"  Features: {len(features)}")

    capital = INITIAL_CAPITAL
    all_equity = []
    all_trades = []
    window_results = []
    window_num = 0

    train_end = INITIAL_TRAIN_H

    while train_end + STEP_H <= len(df):
        test_start = train_end
        test_end = min(train_end + STEP_H, len(df))

        df_train = df.iloc[:train_end]
        df_test = df.iloc[test_start:test_end].reset_index(drop=False)

        # Train fresh models
        bull_model, bear_model = train_dual_models(df_train, features)

        # Predict on test window
        X_test = df_test[features].fillna(0).values
        bull_probs = bull_model.predict_proba(X_test)[:, 1]
        bear_probs = bear_model.predict_proba(X_test)[:, 1]

        # Simulate
        capital_out, trades, equity = simulate_window(df_test, bull_probs, bear_probs, capital)

        # Record
        window_ret = (capital_out / capital - 1) * 100
        n_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / n_trades * 100 if n_trades > 0 else 0
        long_trades = sum(1 for t in trades if t["dir"] == "L")
        short_trades = sum(1 for t in trades if t["dir"] == "S")
        long_pnl = sum(t["pnl"] for t in trades if t["dir"] == "L")
        short_pnl = sum(t["pnl"] for t in trades if t["dir"] == "S")

        window_results.append({
            "window": window_num,
            "train_end_date": str(df.index[train_end - 1].date()),
            "test_start": str(df.index[test_start].date()),
            "test_end": str(df.index[min(test_end - 1, len(df) - 1)].date()),
            "return_pct": window_ret,
            "capital_in": capital,
            "capital_out": capital_out,
            "trades": n_trades,
            "win_rate": win_rate,
            "longs": long_trades,
            "shorts": short_trades,
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
        })

        print(f"  W{window_num:02d} Train→{df.index[train_end-1].date()} "
              f"Test:{df.index[test_start].date()}→{df.index[min(test_end-1, len(df)-1)].date()} "
              f"Ret:{window_ret:>+7.2f}% Trades:{n_trades:>3} WR:{win_rate:>5.1f}% "
              f"L:{long_trades} S:{short_trades}")

        capital = capital_out
        all_equity.extend(equity)
        all_trades.extend(trades)
        window_num += 1
        train_end += STEP_H

    # Overall OOS metrics
    total_ret = (capital / INITIAL_CAPITAL - 1) * 100
    eq_series = pd.Series(all_equity)
    cummax = eq_series.cummax()
    dd = (eq_series - cummax) / cummax
    max_dd = dd.min() * 100

    rets = eq_series.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(24 * 365)) if len(rets) > 0 and rets.std() > 0 else 0

    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["pnl"] > 0)
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    gross_p = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    gross_l = abs(sum(t["pnl"] for t in all_trades if t["pnl"] < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")

    total_long_pnl = sum(t["pnl"] for t in all_trades if t["dir"] == "L")
    total_short_pnl = sum(t["pnl"] for t in all_trades if t["dir"] == "S")

    # Sharpe stability — first half vs second half of windows
    wdf = pd.DataFrame(window_results)
    mid = len(wdf) // 2
    first_half_ret = wdf.iloc[:mid]["return_pct"].mean() if mid > 0 else 0
    second_half_ret = wdf.iloc[mid:]["return_pct"].mean() if mid > 0 else 0

    # Buy & Hold
    bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[INITIAL_TRAIN_H] - 1) * 100

    result = {
        "pair": pair_name,
        "total_return": total_ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_trades": total_trades,
        "win_rate": total_wr,
        "profit_factor": pf,
        "windows": window_num,
        "first_half_avg_ret": first_half_ret,
        "second_half_avg_ret": second_half_ret,
        "bh_return": bh_ret,
        "window_details": wdf,
        "equity": eq_series,
        "long_pnl": total_long_pnl,
        "short_pnl": total_short_pnl,
    }

    return result


# ======================================================================
#  REVERSE POLARITY STRESS TEST
# ======================================================================
def reverse_polarity_test(pair_name, symbol):
    """
    Invert price data (Bull becomes Bear and vice versa).
    If the model learned universal momentum, it should still work.
    If it just learned "sell everything," it will fail.
    """
    print(f"\n  --- Reverse Polarity Test: {pair_name} ---")

    fetcher = DerivativesDataFetcher(symbol)
    df = fetcher.fetch_all_data(use_cache=True)
    if df.empty:
        return None

    # Invert: High/Low/Close become 1/Low, 1/High, 1/Close (preserving OHLC relationships)
    max_price = df["Close"].max() * 2  # Scale factor
    df_inv = df.copy()
    df_inv["Close"] = max_price - df["Close"]
    df_inv["Open"] = max_price - df["Open"]
    df_inv["High"] = max_price - df["Low"]   # Inverted: old low becomes new high
    df_inv["Low"] = max_price - df["High"]   # Inverted: old high becomes new low

    # Re-engineer features on inverted data
    engineer = FeatureEngineer()
    df_inv = engineer.engineer_features(df_inv)
    df_inv = engineer.compute_sentiment_score(df_inv)
    df_inv, _ = engineer.engineer_ml_features(df_inv)
    df_inv = engineer.create_ml_target(df_inv, horizon=24, threshold_pct=1.0)
    df_inv = engineer.create_ml_target_short(df_inv, horizon=24, threshold_pct=2.0)

    available = [f for f in SHIELD_FEATURES if f in df_inv.columns]
    df_inv = df_inv.dropna(subset=["target", "target_short"])
    df_inv["target"] = df_inv["target"].astype(int)
    df_inv["target_short"] = df_inv["target_short"].astype(int)
    df_inv = df_inv.fillna(0)

    # Run a simpler walk-forward: just train on first 70%, test on last 30%
    split = int(len(df_inv) * 0.70)
    df_train = df_inv.iloc[:split]
    df_test = df_inv.iloc[split:].reset_index(drop=False)

    bull_model, bear_model = train_dual_models(df_train, available)

    X_test = df_test[available].fillna(0).values
    bull_probs = bull_model.predict_proba(X_test)[:, 1]
    bear_probs = bear_model.predict_proba(X_test)[:, 1]

    capital_out, trades, equity = simulate_window(df_test, bull_probs, bear_probs, INITIAL_CAPITAL)
    ret = (capital_out / INITIAL_CAPITAL - 1) * 100

    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0

    eq_s = pd.Series(equity)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100

    print(f"    Inverted data: Return={ret:+.2f}%, Trades={n_trades}, WR={wr:.1f}%, MaxDD={max_dd:.2f}%")
    return {"return": ret, "trades": n_trades, "win_rate": wr, "max_dd": max_dd}


# ======================================================================
#  MAIN + REPORTING
# ======================================================================
def main():
    print("=" * 80)
    print("  WALK-FORWARD VALIDATION — Dual-Core XGBoost")
    print("=" * 80)
    print(f"  Initial Train: {INITIAL_TRAIN_DAYS} days | Step: {STEP_DAYS} days")
    print(f"  Thresholds: Bull>{BULL_ENTRY} Bear>{BEAR_ENTRY} Block<{BULL_BLOCK}")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    all_polarity = []

    for pair_name, symbol in PAIRS.items():
        result = walk_forward(pair_name, symbol)
        if result is None:
            continue
        all_results.append(result)

        # Save window details
        result["window_details"].to_csv(
            os.path.join(RESULTS_DIR, f"{pair_name}_windows.csv"), index=False
        )

        # Reverse polarity
        pol = reverse_polarity_test(pair_name, symbol)
        if pol:
            pol["pair"] = pair_name
            all_polarity.append(pol)

    if not all_results:
        print("\nNo results.")
        return

    # ===== GRAND REPORT =====
    IN_SAMPLE_AVG = 270.92  # Previous in-sample result

    print(f"\n{'='*100}")
    print(f"  WALK-FORWARD VALIDATION REPORT")
    print(f"{'='*100}")

    print(f"\n  {'Pair':<6} {'WF Return%':>11} {'MaxDD%':>8} {'Sharpe':>8} {'Trades':>7} "
          f"{'WR%':>6} {'PF':>6} {'Windows':>8} {'B&H%':>8}")
    print(f"  {'-'*80}")

    avgs = {"ret": [], "dd": [], "sharpe": [], "wr": [], "pf": []}

    for r in all_results:
        print(f"  {r['pair']:<6} {r['total_return']:>+11.2f} {r['max_dd']:>8.2f} "
              f"{r['sharpe']:>8.2f} {r['total_trades']:>7} {r['win_rate']:>6.1f} "
              f"{r['profit_factor']:>6.2f} {r['windows']:>8} {r['bh_return']:>+8.2f}")
        avgs["ret"].append(r["total_return"])
        avgs["dd"].append(r["max_dd"])
        avgs["sharpe"].append(r["sharpe"])
        avgs["wr"].append(r["win_rate"])
        avgs["pf"].append(r["profit_factor"])

    print(f"  {'-'*80}")
    print(f"  {'AVG':<6} {np.mean(avgs['ret']):>+11.2f} {np.mean(avgs['dd']):>8.2f} "
          f"{np.mean(avgs['sharpe']):>8.2f} {'':>7} {np.mean(avgs['wr']):>6.1f} "
          f"{np.mean(avgs['pf']):>6.2f}")

    # Direction breakdown
    print(f"\n  Direction Breakdown:")
    for r in all_results:
        print(f"    {r['pair']}: Long PnL=${r['long_pnl']:+,.0f} | Short PnL=${r['short_pnl']:+,.0f}")

    # Sharpe Stability
    print(f"\n  Sharpe Stability (1st Half vs 2nd Half of Windows):")
    for r in all_results:
        trend = "📈" if r["second_half_avg_ret"] >= r["first_half_avg_ret"] else "📉"
        print(f"    {r['pair']}: 1st={r['first_half_avg_ret']:+.2f}% 2nd={r['second_half_avg_ret']:+.2f}% {trend}")

    # Reality Gap
    avg_wf = np.mean(avgs["ret"])
    print(f"\n  === THE REALITY GAP ===")
    print(f"    In-Sample (Dual-Core):    +{IN_SAMPLE_AVG:.2f}%")
    print(f"    Walk-Forward (OOS avg):   {avg_wf:+.2f}%")
    gap = avg_wf / IN_SAMPLE_AVG * 100 if IN_SAMPLE_AVG != 0 else 0
    print(f"    Retention:                {gap:.1f}% of in-sample performance")

    if avg_wf > 80:
        print(f"    ✅ STRONG — retained >{80/IN_SAMPLE_AVG*100:.0f}% of in-sample performance")
    elif avg_wf > 0:
        print(f"    ⚠️  MODERATE — positive but significant degradation")
    else:
        print(f"    ❌ FAILURE — Walk-forward is NEGATIVE. Model is overfit.")

    # Reverse Polarity
    if all_polarity:
        print(f"\n  === REVERSE POLARITY STRESS TEST ===")
        print(f"  (Did the model learn universal momentum, or just 'sell everything'?)")
        for p in all_polarity:
            status = "✅ PASS" if p["return"] > 0 and p["win_rate"] > 50 else "❌ FAIL"
            print(f"    {p['pair']}: Return={p['return']:+.2f}% WR={p['win_rate']:.1f}% "
                  f"MaxDD={p['max_dd']:.2f}% → {status}")

        avg_pol_ret = np.mean([p["return"] for p in all_polarity])
        if avg_pol_ret > 0:
            print(f"    ✅ UNIVERSAL: Model works on inverted data (avg {avg_pol_ret:+.2f}%)")
        else:
            print(f"    ❌ REGIME-SPECIFIC: Model fails on inverted data (avg {avg_pol_ret:+.2f}%)")

    # ===== PASS/FAIL VERDICT =====
    avg_wr = np.mean(avgs["wr"])
    avg_dd = np.mean(avgs["dd"])

    print(f"\n{'='*100}")
    print(f"  FINAL VERDICT")
    print(f"{'='*100}")

    pass_wr = avg_wr > 55
    pass_dd = avg_dd > -25  # max_dd is negative, so > -25 means less than 25% DD
    pass_ret = avg_wf > 0

    print(f"    Win Rate:     {avg_wr:.1f}% {'✅ PASS (>55%)' if pass_wr else '❌ FAIL (<55%)'}")
    print(f"    Max Drawdown: {avg_dd:.2f}% {'✅ PASS (>-25%)' if pass_dd else '❌ FAIL (<-25%)'}")
    print(f"    OOS Return:   {avg_wf:+.2f}% {'✅ PASS (>0%)' if pass_ret else '❌ FAIL (<0%)'}")

    if pass_wr and pass_dd and pass_ret:
        print(f"\n  🟢 OVERALL: PASS — Strategy shows real edge in walk-forward validation.")
        print(f"     RECOMMENDATION: Proceed to paper trading.")
    elif pass_ret:
        print(f"\n  🟡 OVERALL: CONDITIONAL PASS — Positive returns but with concerns.")
        print(f"     RECOMMENDATION: Investigate drawdown/win rate before deploying.")
    else:
        print(f"\n  🔴 OVERALL: FAIL — Strategy does not survive walk-forward validation.")
        print(f"     RECOMMENDATION: DO NOT DEPLOY. Model is likely overfit.")

    print(f"{'='*100}")

    # Save summary
    summary = pd.DataFrame([{
        "pair": r["pair"],
        "wf_return": r["total_return"],
        "max_dd": r["max_dd"],
        "sharpe": r["sharpe"],
        "trades": r["total_trades"],
        "win_rate": r["win_rate"],
        "pf": r["profit_factor"],
        "windows": r["windows"],
        "bh_return": r["bh_return"],
    } for r in all_results])
    summary.to_csv(os.path.join(RESULTS_DIR, "wfo_summary.csv"), index=False)
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
