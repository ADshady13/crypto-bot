"""
Retrain Pipeline — Champion vs. Challenger Model Promotion.

This script runs INDEPENDENTLY of the main bot (via cron or manual invocation).
It trains a Challenger model and tests it against the current Champion on
the most recent 30 days of data. If the Challenger wins, it is atomically
promoted via os.replace().

Usage:
    python -m jobs.retrain --pair BTCUSDT
    python -m jobs.retrain --all

Logic:
  1. Data Merge: Load historical CSV + append any live trade CSVs (deduplicated).
  2. The Arena:
     a. Train Challenger on Data[0 : -30 days]
     b. Test Challenger on the last 30 days (OOS)
     c. Test Champion on the same 30 days
  3. The Gate:
     IF Challenger_Sharpe > Champion_Sharpe AND Challenger_WR > 55%:
       → Atomic os.replace() to promote Challenger
       → Log: "🏆 New Champion Promoted!"
     ELSE:
       → Log: "Champion retained."
"""

import os
import sys
import argparse
import logging
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.notification import Notifier
from strategies.feature_engineering import FeatureEngineer, SHIELD_FEATURES

logger = logging.getLogger("crypto_bot")

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
HOURS_PER_DAY = 24
TEST_HOURS = Config.RETRAIN_TEST_DAYS * HOURS_PER_DAY  # 30 days = 720 hours

# Simulation constants
FEE_PCT = Config.FEE_PCT
SLIPPAGE_PCT = Config.SLIPPAGE_PCT
SL_PCT = Config.SL_PCT
BULL_ENTRY = Config.BULL_ENTRY_PROB
BEAR_ENTRY = Config.BEAR_ENTRY_PROB
BULL_BLOCK = Config.BULL_BLOCK_PROB
BEAR_BLOCK = Config.BEAR_BLOCK_PROB


# ======================================================================
#  DATA
# ======================================================================
def load_data(pair: str) -> pd.DataFrame:
    """Load historical + live trade data, deduplicated."""
    hist_path = os.path.join(Config.DATA_DIR, f"{pair}_data.csv")
    live_path = os.path.join(Config.DATA_DIR, f"{pair}_live.csv")

    if not os.path.exists(hist_path):
        logger.error(f"Historical data not found: {hist_path}")
        return pd.DataFrame()

    df = pd.read_csv(hist_path, index_col="timestamp", parse_dates=True)
    logger.info(f"Loaded historical: {len(df)} rows")

    # Append live data if available
    if os.path.exists(live_path):
        live = pd.read_csv(live_path, index_col="timestamp", parse_dates=True)
        df = pd.concat([df, live])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        logger.info(f"Appended live data: {len(live)} rows → total {len(df)}")

    return df


# ======================================================================
#  TRAINING
# ======================================================================
def train_model(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """Train a fresh XGBoost classifier."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=42, use_label_encoder=False,
        verbosity=0, min_child_weight=5, gamma=0.1,
    )
    model.fit(X, y, verbose=False)
    return model


# ======================================================================
#  SIMULATION (lightweight — for scoring models on test period)
# ======================================================================
def simulate_test(df_test: pd.DataFrame, bull_probs: np.ndarray, bear_probs: np.ndarray) -> dict:
    """Run Dual-Core simulation on a test window. Returns metrics."""
    capital = 100_000
    position = 0       # +1 long, -1 short, 0 flat
    entry_price = 0.0
    trade_capital = 0.0
    trades = []
    equity = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        bp = bull_probs[i]
        sp = bear_probs[i]

        # Signal
        if bp > BULL_ENTRY and sp < BEAR_BLOCK:
            action = "long"
        elif sp > BEAR_ENTRY and bp < BULL_BLOCK:
            action = "short"
        else:
            action = "flat"

        # Stop loss
        if position == 1 and low <= entry_price * (1 - SL_PCT):
            exit_p = entry_price * (1 - SL_PCT)
            pnl = trade_capital * (exit_p / entry_price - 1) - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append(pnl)
            position = 0; trade_capital = 0

        elif position == -1 and high >= entry_price * (1 + SL_PCT):
            exit_p = entry_price * (1 + SL_PCT)
            pnl = trade_capital * (entry_price - exit_p) / entry_price - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append(pnl)
            position = 0; trade_capital = 0

        # Signal exit
        if position == 1 and action != "long":
            pnl = trade_capital * (close / entry_price - 1) - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append(pnl)
            position = 0; trade_capital = 0
        elif position == -1 and action != "short":
            pnl = trade_capital * (entry_price - close) / entry_price - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            capital += trade_capital + pnl
            trades.append(pnl)
            position = 0; trade_capital = 0

        # Entry
        if position == 0 and action == "long":
            tc = capital * 0.95
            trade_capital = tc - tc * (FEE_PCT + SLIPPAGE_PCT)
            capital -= tc
            entry_price = close
            position = 1
        elif position == 0 and action == "short":
            tc = capital * 0.95
            trade_capital = tc - tc * (FEE_PCT + SLIPPAGE_PCT)
            capital -= tc
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

    # Close at end
    if position != 0:
        fc = df_test.iloc[-1]["Close"]
        pnl_pct = (fc / entry_price - 1) if position == 1 else (entry_price - fc) / entry_price
        pnl = trade_capital * pnl_pct - trade_capital * (FEE_PCT + SLIPPAGE_PCT)
        capital += trade_capital + pnl
        trades.append(pnl)

    # Metrics
    eq_s = pd.Series(equity)
    total_ret = (capital / 100_000 - 1) * 100
    rets = eq_s.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(24 * 365)) if len(rets) > 0 and rets.std() > 0 else 0
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100

    n = len(trades)
    wins = sum(1 for t in trades if t > 0)
    wr = wins / n * 100 if n > 0 else 0

    return {
        "return_pct": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": n,
        "win_rate": wr,
    }


# ======================================================================
#  THE ARENA — Champion vs Challenger
# ======================================================================
def run_arena(pair: str, notifier: Notifier) -> bool:
    """
    Train Challenger, test against Champion on last 30 days.
    Returns True if Challenger was promoted.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  ARENA: {pair}")
    logger.info(f"{'='*60}")

    df = load_data(pair)
    if len(df) < TEST_HOURS + 2160:  # Need at least test + 90 days training
        logger.warning(f"Not enough data for {pair}: {len(df)} rows")
        return False

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.transform(df)
    df = fe.create_targets(df)

    # Drop NaN targets
    df = df.dropna(subset=["target", "target_short"])
    df["target"] = df["target"].astype(int)
    df["target_short"] = df["target_short"].astype(int)

    # Split: train on everything EXCEPT last 30 days, test on last 30 days
    test_start = len(df) - TEST_HOURS
    df_train = df.iloc[:test_start]
    df_test = df.iloc[test_start:]

    available = [f for f in SHIELD_FEATURES if f in df.columns]
    X_train = df_train[available].fillna(0).values
    X_test = df_test[available].fillna(0).values
    y_bull_train = df_train["target"].values
    y_bear_train = df_train["target_short"].values

    logger.info(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")
    logger.info(f"  Features: {len(available)}")

    # --- Train Challenger ---
    logger.info("  Training Challenger Bull...")
    challenger_bull = train_model(X_train, y_bull_train)
    logger.info("  Training Challenger Bear...")
    challenger_bear = train_model(X_train, y_bear_train)

    # --- Challenger Score ---
    c_bull_probs = challenger_bull.predict_proba(X_test)[:, 1]
    c_bear_probs = challenger_bear.predict_proba(X_test)[:, 1]
    challenger_score = simulate_test(df_test, c_bull_probs, c_bear_probs)
    logger.info(f"  Challenger: Sharpe={challenger_score['sharpe']:.2f} "
                f"WR={challenger_score['win_rate']:.1f}% "
                f"Ret={challenger_score['return_pct']:+.2f}%")

    # --- Champion Score ---
    pair_short = pair.replace("USDT", "")
    bull_path = os.path.join(Config.MODEL_DIR, f"xgb_bull_{pair_short}.json")
    bear_path = os.path.join(Config.MODEL_DIR, f"xgb_bear_{pair_short}.json")

    champion_score = None
    if os.path.exists(bull_path) and os.path.exists(bear_path):
        champion_bull = XGBClassifier()
        champion_bull.load_model(bull_path)
        champion_bear = XGBClassifier()
        champion_bear.load_model(bear_path)

        ch_bull_probs = champion_bull.predict_proba(X_test)[:, 1]
        ch_bear_probs = champion_bear.predict_proba(X_test)[:, 1]
        champion_score = simulate_test(df_test, ch_bull_probs, ch_bear_probs)
        logger.info(f"  Champion:   Sharpe={champion_score['sharpe']:.2f} "
                    f"WR={champion_score['win_rate']:.1f}% "
                    f"Ret={champion_score['return_pct']:+.2f}%")
    else:
        logger.info("  No existing Champion — Challenger auto-promotes if gate passes.")

    # --- THE GATE ---
    promoted = False
    gate_passed = (
        challenger_score["win_rate"] > Config.RETRAIN_MIN_WIN_RATE
        and challenger_score["sharpe"] > 0
    )

    if champion_score is not None:
        gate_passed = gate_passed and (
            challenger_score["sharpe"] > champion_score["sharpe"]
        )

    if gate_passed:
        # Save Challenger as temp, then atomic rename
        temp_bull = bull_path + ".challenger"
        temp_bear = bear_path + ".challenger"

        challenger_bull.save_model(temp_bull)
        challenger_bear.save_model(temp_bear)

        # Atomic rename — if the process crashes mid-swap, the old Champion survives
        os.replace(temp_bull, bull_path)
        os.replace(temp_bear, bear_path)

        promoted = True
        msg = (
            f"🏆 New Champion Promoted for {pair}!\n"
            f"Sharpe: {challenger_score['sharpe']:.2f} "
            f"(was {champion_score['sharpe']:.2f})\n" if champion_score else ""
            f"WR: {challenger_score['win_rate']:.1f}%\n"
            f"Return: {challenger_score['return_pct']:+.2f}%"
        )
        logger.info(msg)
        notifier.model_alert(msg)
    else:
        reason = []
        if challenger_score["win_rate"] <= Config.RETRAIN_MIN_WIN_RATE:
            reason.append(f"WR={challenger_score['win_rate']:.1f}% < {Config.RETRAIN_MIN_WIN_RATE}%")
        if champion_score and challenger_score["sharpe"] <= champion_score["sharpe"]:
            reason.append(f"Sharpe {challenger_score['sharpe']:.2f} <= {champion_score['sharpe']:.2f}")
        logger.info(f"  Champion retained. Reason: {', '.join(reason)}")

    return promoted


# ======================================================================
#  CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Champion vs Challenger Model Retraining")
    parser.add_argument("--pair", type=str, help="Single pair to retrain (e.g. BTCUSDT)")
    parser.add_argument("--all", action="store_true", help="Retrain all pairs")
    args = parser.parse_args()

    from core.logging_setup import setup_logging
    setup_logging("retrain")

    Config.ensure_dirs()
    notifier = Notifier()

    pairs = PAIRS if args.all else [args.pair or Config.DEFAULT_PAIR]

    logger.info(f"Starting retraining pipeline for: {pairs}")
    results = {}

    for pair in pairs:
        try:
            promoted = run_arena(pair, notifier)
            results[pair] = "PROMOTED" if promoted else "RETAINED"
        except Exception as e:
            logger.error(f"Retraining failed for {pair}: {e}", exc_info=True)
            results[pair] = f"ERROR: {e}"

    logger.info(f"\nRetraining complete: {results}")


if __name__ == "__main__":
    main()
