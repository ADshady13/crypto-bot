"""
Dual-Core ML Backtest — Sniper Entries via Bull + Bear XGBoost gates.

Strategy Logic:
  LONG:  Bull_Prob > 0.70 AND Bear_Prob < 0.30 → BUY (1x)
  SHORT: Bear_Prob > 0.70 AND Bull_Prob < 0.30 → SHORT (1x via Spot proxy)
  FLAT:  Otherwise → USDT

Compares:
  1. Raw Trend Strategy (200-EMA + ROC, no ML) — the -87% disaster
  2. ML Long-Only Shield (Bull P > 0.55)
  3. ML Dual-Core (Bull + Bear probability gates)

Uses existing cached data + saved XGBoost models.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

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

INITIAL_CAPITAL = 100_000
FEE_PCT = 0.001       # 0.1% taker fee
SLIPPAGE_PCT = 0.0005 # 0.05% slippage
SL_PCT = 0.03         # 3% stop loss
EMA_PERIOD = 200
ROC_PERIOD = 12
FUNDING_INTERVAL = 8

# ML Thresholds
BULL_ENTRY = 0.70      # Bull prob must be > this to go long
BEAR_ENTRY = 0.70      # Bear prob must be > this to go short
BULL_BLOCK_SHORT = 0.30 # Bull prob must be < this to allow short
BEAR_BLOCK_LONG = 0.30  # Bear prob must be < this to allow long

# Shield (long-only ML) threshold — from previous research
SHIELD_THRESH = 0.55

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ======================================================================
#  DATA + MODEL LOADING
# ======================================================================
def load_model(model_name, pair_name):
    """Load a saved XGBoost model and its metadata."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{pair_name}.json")
    meta_path = os.path.join(MODELS_DIR, f"{model_name}_{pair_name}_meta.json")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found: {model_path}")
        return None, None

    model = XGBClassifier()
    model.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta["features"]


def prepare_data(symbol):
    """Load cached data, engineer all features, return full DataFrame."""
    fetcher = DerivativesDataFetcher(symbol)
    df = fetcher.fetch_all_data(use_cache=True, target_candles=10000)
    if df.empty:
        return None

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)
    df, _ = engineer.engineer_ml_features(df)

    # Trend signals for raw strategy
    df["ema_200_raw"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["roc_raw"] = df["Close"].pct_change(periods=ROC_PERIOD) * 100
    df["signal_long_raw"] = (df["Close"] > df["ema_200_raw"]) & (df["roc_raw"] > 0)
    df["signal_short_raw"] = (df["Close"] < df["ema_200_raw"]) & (df["roc_raw"] < 0)

    df = df.iloc[EMA_PERIOD:].copy()
    df = df.fillna(0)

    return df


# ======================================================================
#  SIMULATION ENGINE
# ======================================================================
def simulate(df, signals, mode_label):
    """
    Run backtest given a signals DataFrame with columns:
      'action': 'long', 'short', or 'flat'

    Returns metrics dict.
    """
    capital = INITIAL_CAPITAL
    position = 0        # +1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_idx = 0
    trade_capital = 0.0

    trades = []
    equity = []
    sl_hits = 0

    rows = df

    for i in range(len(rows)):
        row = rows.iloc[i]
        ts = row["timestamp"]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        action = signals[i]  # 'long', 'short', or 'flat'
        funding_rate = row.get("funding_rate", 0)

        # ---- Funding (every 8h) ----
        if position != 0 and i % FUNDING_INTERVAL == 0:
            position_value = trade_capital
            if position == 1:
                capital -= position_value * funding_rate
            elif position == -1:
                capital += position_value * funding_rate

        # ---- Intra-candle Stop Loss ----
        if position == 1:
            sl_price = entry_price * (1 - SL_PCT)
            if low <= sl_price:
                exit_price = sl_price
                pnl_pct = (exit_price / entry_price) - 1
                exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
                net_pnl = trade_capital * pnl_pct - exit_cost
                capital += trade_capital + net_pnl
                trades.append({
                    "direction": "LONG", "pnl_pct": pnl_pct * 100,
                    "pnl_usd": net_pnl, "exit_reason": "SL",
                    "entry_time": rows.iloc[entry_idx]["timestamp"], "exit_time": ts,
                })
                sl_hits += 1
                position = 0; trade_capital = 0

        elif position == -1:
            sl_price = entry_price * (1 + SL_PCT)
            if high >= sl_price:
                exit_price = sl_price
                pnl_pct = (entry_price - exit_price) / entry_price
                exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
                net_pnl = trade_capital * pnl_pct - exit_cost
                capital += trade_capital + net_pnl
                trades.append({
                    "direction": "SHORT", "pnl_pct": pnl_pct * 100,
                    "pnl_usd": net_pnl, "exit_reason": "SL",
                    "entry_time": rows.iloc[entry_idx]["timestamp"], "exit_time": ts,
                })
                sl_hits += 1
                position = 0; trade_capital = 0

        # ---- Signal-based exit ----
        if position == 1 and action != "long":
            exit_price = close
            pnl_pct = (exit_price / entry_price) - 1
            exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            net_pnl = trade_capital * pnl_pct - exit_cost
            capital += trade_capital + net_pnl
            trades.append({
                "direction": "LONG", "pnl_pct": pnl_pct * 100,
                "pnl_usd": net_pnl, "exit_reason": "SIGNAL",
                "entry_time": rows.iloc[entry_idx]["timestamp"], "exit_time": ts,
            })
            position = 0; trade_capital = 0

        elif position == -1 and action != "short":
            exit_price = close
            pnl_pct = (entry_price - exit_price) / entry_price
            exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            net_pnl = trade_capital * pnl_pct - exit_cost
            capital += trade_capital + net_pnl
            trades.append({
                "direction": "SHORT", "pnl_pct": pnl_pct * 100,
                "pnl_usd": net_pnl, "exit_reason": "SIGNAL",
                "entry_time": rows.iloc[entry_idx]["timestamp"], "exit_time": ts,
            })
            position = 0; trade_capital = 0

        # ---- Entry ----
        if position == 0:
            if action == "long":
                entry_cost = capital * (FEE_PCT + SLIPPAGE_PCT)
                trade_capital = capital - entry_cost
                capital = 0
                entry_price = close
                entry_idx = i
                position = 1
            elif action == "short":
                entry_cost = capital * (FEE_PCT + SLIPPAGE_PCT)
                trade_capital = capital - entry_cost
                capital = 0
                entry_price = close
                entry_idx = i
                position = -1

        # ---- Track equity ----
        if position == 1:
            unrealized = trade_capital * (close / entry_price - 1)
            eq = capital + trade_capital + unrealized
        elif position == -1:
            unrealized = trade_capital * (entry_price - close) / entry_price
            eq = capital + trade_capital + unrealized
        else:
            eq = capital
        equity.append(eq)

    # Close open position at end
    if position != 0:
        final_close = rows.iloc[-1]["Close"]
        if position == 1:
            pnl_pct = (final_close / entry_price) - 1
        else:
            pnl_pct = (entry_price - final_close) / entry_price
        exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
        net_pnl = trade_capital * pnl_pct - exit_cost
        capital += trade_capital + net_pnl
        trades.append({
            "direction": "LONG" if position == 1 else "SHORT",
            "pnl_pct": pnl_pct * 100, "pnl_usd": net_pnl, "exit_reason": "EOD",
            "entry_time": rows.iloc[entry_idx]["timestamp"],
            "exit_time": rows.iloc[-1]["timestamp"],
        })

    # Compute metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_s = pd.Series(equity)

    final_eq = equity_s.iloc[-1] if len(equity_s) > 0 else INITIAL_CAPITAL
    ret = (final_eq / INITIAL_CAPITAL - 1) * 100

    cummax = equity_s.cummax()
    dd = (equity_s - cummax) / cummax
    max_dd = dd.min() * 100

    rets = equity_s.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(24 * 365)) if len(rets) > 0 and rets.std() > 0 else 0

    n = len(trades_df)
    if n > 0 and "pnl_usd" in trades_df.columns:
        wins = (trades_df["pnl_usd"] > 0).sum()
        win_rate = wins / n * 100
        gross_p = trades_df[trades_df["pnl_usd"] > 0]["pnl_usd"].sum()
        gross_l = abs(trades_df[trades_df["pnl_usd"] < 0]["pnl_usd"].sum())
        pf = gross_p / gross_l if gross_l > 0 else float("inf")

        longs = trades_df[trades_df["direction"] == "LONG"]
        shorts = trades_df[trades_df["direction"] == "SHORT"]
        long_pnl = longs["pnl_usd"].sum() if len(longs) > 0 else 0
        short_pnl = shorts["pnl_usd"].sum() if len(shorts) > 0 else 0
        long_count = len(longs)
        short_count = len(shorts)
    else:
        wins = win_rate = pf = long_pnl = short_pnl = 0
        long_count = short_count = 0

    return {
        "label": mode_label,
        "return_pct": ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_trades": n,
        "wins": wins,
        "win_rate": win_rate,
        "profit_factor": pf,
        "sl_hits": sl_hits,
        "long_count": long_count,
        "short_count": short_count,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }


# ======================================================================
#  SIGNAL GENERATORS
# ======================================================================
def generate_raw_signals(df):
    """Raw trend strategy: 200-EMA + ROC, long and short (the -87% disaster)."""
    signals = []
    for _, row in df.iterrows():
        if row["signal_long_raw"]:
            signals.append("long")
        elif row["signal_short_raw"]:
            signals.append("short")
        else:
            signals.append("flat")
    return signals


def generate_shield_signals(df, bull_probs):
    """ML Long-Only Shield: Only long when Bull P > threshold."""
    signals = []
    for i in range(len(df)):
        if bull_probs[i] > SHIELD_THRESH and df.iloc[i]["signal_long_raw"]:
            signals.append("long")
        else:
            signals.append("flat")
    return signals


def generate_dual_core_signals(bull_probs, bear_probs):
    """
    Dual-Core: Both models must agree with high conviction.
    Long:  Bull_P > 0.70 AND Bear_P < 0.30
    Short: Bear_P > 0.70 AND Bull_P < 0.30
    """
    signals = []
    for i in range(len(bull_probs)):
        bp = bull_probs[i]
        sp = bear_probs[i]

        if bp > BULL_ENTRY and sp < BEAR_BLOCK_LONG:
            signals.append("long")
        elif sp > BEAR_ENTRY and bp < BULL_BLOCK_SHORT:
            signals.append("short")
        else:
            signals.append("flat")
    return signals


# ======================================================================
#  MAIN
# ======================================================================
def main():
    print("=" * 80)
    print("  DUAL-CORE ML BACKTEST — Bull + Bear XGBoost Sniper Strategy")
    print("=" * 80)
    print(f"  Long:  Bull_P > {BULL_ENTRY} AND Bear_P < {BEAR_BLOCK_LONG}")
    print(f"  Short: Bear_P > {BEAR_ENTRY} AND Bull_P < {BULL_BLOCK_SHORT}")
    print(f"  Shield (Long-Only): Bull_P > {SHIELD_THRESH}")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f} | Fee: {FEE_PCT*100}% | SL: {SL_PCT*100}%")

    all_results = []

    for pair_name, symbol in PAIRS.items():
        print(f"\n{'~'*60}")
        print(f"  {pair_name} ({symbol})")
        print(f"{'~'*60}")

        # Load data
        df = prepare_data(symbol)
        if df is None:
            continue

        # Use only the TEST portion (last 30%) for backtesting — same as training split
        split_idx = int(len(df) * 0.70)
        df_test = df.iloc[split_idx:].copy().reset_index(drop=False)
        print(f"  Test data: {len(df_test)} rows ({df.index[split_idx]} to {df.index[-1]})")

        # Load models
        bull_model, bull_features = load_model("xgb_bull", pair_name)
        bear_model, bear_features = load_model("xgb_bear", pair_name)

        if bull_model is None or bear_model is None:
            print(f"  SKIPPING {pair_name}: Models not found. Run train_models.py first.")
            continue

        # Generate probabilities
        bull_features_avail = [f for f in bull_features if f in df_test.columns]
        bear_features_avail = [f for f in bear_features if f in df_test.columns]

        X_bull = df_test[bull_features_avail].fillna(0).values
        X_bear = df_test[bear_features_avail].fillna(0).values

        bull_probs = bull_model.predict_proba(X_bull)[:, 1]
        bear_probs = bear_model.predict_proba(X_bear)[:, 1]

        # Buy & Hold
        bh_return = (df_test["Close"].iloc[-1] / df_test["Close"].iloc[0] - 1) * 100

        # 1. Raw Strategy (Long + Short, no ML)
        raw_signals = generate_raw_signals(df_test)
        raw_metrics = simulate(df_test, raw_signals, "Raw (L+S)")

        # 2. ML Shield (Long-Only, Bull P > 0.55)
        shield_signals = generate_shield_signals(df_test, bull_probs)
        shield_metrics = simulate(df_test, shield_signals, "ML Shield (L)")

        # 3. Dual-Core (Bull + Bear gates)
        dual_signals = generate_dual_core_signals(bull_probs, bear_probs)
        dual_metrics = simulate(df_test, dual_signals, "Dual-Core")

        # Signal distribution
        dual_longs = dual_signals.count("long")
        dual_shorts = dual_signals.count("short")
        dual_flat = dual_signals.count("flat")
        total_h = len(dual_signals)
        print(f"\n  Dual-Core Signal Distribution:")
        print(f"    Long:  {dual_longs:>5} hours ({dual_longs/total_h*100:.1f}%)")
        print(f"    Short: {dual_shorts:>5} hours ({dual_shorts/total_h*100:.1f}%)")
        print(f"    Flat:  {dual_flat:>5} hours ({dual_flat/total_h*100:.1f}%)")

        # Print pair results
        print(f"\n  {'Metric':<25} {'Raw (L+S)':>12} {'ML Shield':>12} {'Dual-Core':>12}")
        print(f"  {'-'*62}")

        for label, key, fmt in [
            ("Return %", "return_pct", ".2f"),
            ("Max Drawdown %", "max_dd", ".2f"),
            ("Sharpe", "sharpe", ".2f"),
            ("Total Trades", "total_trades", "d"),
            ("Win Rate %", "win_rate", ".1f"),
            ("Profit Factor", "profit_factor", ".2f"),
            ("SL Hits", "sl_hits", "d"),
            ("Long Trades", "long_count", "d"),
            ("Short Trades", "short_count", "d"),
            ("Long PnL $", "long_pnl", ",.0f"),
            ("Short PnL $", "short_pnl", ",.0f"),
        ]:
            r = raw_metrics.get(key, 0)
            s = shield_metrics.get(key, 0)
            d = dual_metrics.get(key, 0)
            print(f"  {label:<25} {r:>12{fmt}} {s:>12{fmt}} {d:>12{fmt}}")

        print(f"  {'Buy & Hold %':<25} {bh_return:>12.2f}")

        all_results.append({
            "pair": pair_name,
            "raw": raw_metrics,
            "shield": shield_metrics,
            "dual": dual_metrics,
            "bh": bh_return,
        })

    # Grand Summary
    if not all_results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*100}")
    print(f"  GRAND SUMMARY — DUAL-CORE vs SHIELD vs RAW")
    print(f"{'='*100}")
    print(f"\n  {'Pair':<6} {'Raw Ret%':>10} {'Shield Ret%':>12} {'Dual Ret%':>10} "
          f"{'B&H%':>8} {'Raw Tr':>7} {'Shield Tr':>10} {'Dual Tr':>8} {'Dual WR%':>9}")
    print(f"  {'-'*90}")

    avgs = {"raw": [], "shield": [], "dual": [], "bh": []}

    for r in all_results:
        raw = r["raw"]
        sh = r["shield"]
        dc = r["dual"]
        bh = r["bh"]

        print(f"  {r['pair']:<6} {raw['return_pct']:>+10.2f} {sh['return_pct']:>+12.2f} "
              f"{dc['return_pct']:>+10.2f} {bh:>+8.2f} {raw['total_trades']:>7} "
              f"{sh['total_trades']:>10} {dc['total_trades']:>8} {dc['win_rate']:>9.1f}")

        avgs["raw"].append(raw["return_pct"])
        avgs["shield"].append(sh["return_pct"])
        avgs["dual"].append(dc["return_pct"])
        avgs["bh"].append(bh)

    print(f"  {'-'*90}")
    print(f"  {'AVG':<6} {np.mean(avgs['raw']):>+10.2f} {np.mean(avgs['shield']):>+12.2f} "
          f"{np.mean(avgs['dual']):>+10.2f} {np.mean(avgs['bh']):>+8.2f}")

    # Direction breakdown for dual-core
    print(f"\n  Dual-Core Direction Breakdown:")
    for r in all_results:
        dc = r["dual"]
        print(f"    {r['pair']}: Longs={dc['long_count']} (${dc['long_pnl']:+,.0f}) | "
              f"Shorts={dc['short_count']} (${dc['short_pnl']:+,.0f})")

    # Verdict
    avg_raw = np.mean(avgs["raw"])
    avg_sh = np.mean(avgs["shield"])
    avg_dc = np.mean(avgs["dual"])

    print(f"\n{'='*100}")
    print(f"  VERDICT:")
    print(f"    Raw Strategy:     {avg_raw:>+8.2f}%")
    print(f"    ML Shield (L):    {avg_sh:>+8.2f}%")
    print(f"    Dual-Core (L+S):  {avg_dc:>+8.2f}%")
    print(f"    Buy & Hold:       {np.mean(avgs['bh']):>+8.2f}%")

    if avg_dc > avg_sh:
        print(f"\n  ✅ Dual-Core BEATS Shield by {avg_dc - avg_sh:+.2f}%")
    else:
        print(f"\n  ❌ Dual-Core TRAILS Shield by {avg_dc - avg_sh:.2f}%")

    if avg_dc > avg_raw:
        print(f"  ✅ Dual-Core BEATS Raw by {avg_dc - avg_raw:+.2f}%")

    # Check if shorts contributed
    total_short_pnl = sum(r["dual"]["short_pnl"] for r in all_results)
    print(f"\n  Total Short PnL (Dual-Core): ${total_short_pnl:+,.0f}")
    if total_short_pnl > 0:
        print(f"  ✅ Shorts ADDED value this time!")
    else:
        print(f"  ❌ Shorts still bleeding. Signal quality needs work.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
