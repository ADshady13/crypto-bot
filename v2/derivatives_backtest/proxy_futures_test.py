"""
Proxy Futures Backtest — Long/Short (1x Leverage) using Spot data.

Simulates a futures strategy using existing spot OHLCV data as a proxy
for futures mark price. Tests whether adding short legs to a trend-following
strategy improves returns over a long-only baseline.

Strategy Logic (Trend + Momentum):
  Long:  Close > 200-EMA AND ROC(12) > 0
  Short: Close < 200-EMA AND ROC(12) < 0

Funding Logic:
  Applied every 8 hours (every 8th row in 1h data).
  Longs pay funding when rate > 0.
  Shorts earn funding when rate > 0.

Stop Loss:
  Intra-candle check: 3% adverse move triggers forced exit.
  Longs: Low <= Entry * 0.97  →  exit at Entry * 0.97
  Shorts: High >= Entry * 1.03 → exit at Entry * 1.03
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------
PAIRS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}

INITIAL_CAPITAL = 100_000
SL_PCT = 0.03          # 3% stop loss
FUNDING_INTERVAL = 8   # Apply funding every 8 hours
EMA_PERIOD = 200
ROC_PERIOD = 12         # 12-hour rate of change
FEE_PCT = 0.001         # 0.1% taker fee per side (entry + exit)
SLIPPAGE_PCT = 0.0005   # 0.05% slippage per side

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
#  DATA LOADING + FEATURE COMPUTATION
# ---------------------------------------------------------------------------
def load_pair(symbol):
    """Load cached CSV, compute EMA-200 and ROC, validate funding coverage."""
    path = os.path.join(DATA_DIR, f"{symbol}_data.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return None

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    df = df.sort_index()

    # Validate funding rate coverage
    if "funding_rate" not in df.columns:
        print(f"  WARNING: No funding_rate column in {symbol}")
        df["funding_rate"] = 0.0
    else:
        default_count = (df["funding_rate"] == 0.0001).sum()
        default_pct = default_count / len(df) * 100
        if default_pct > 20:
            print(f"  ⚠️  WARNING: {symbol} funding_rate is default (0.0001) for "
                  f"{default_count}/{len(df)} rows ({default_pct:.1f}%)")

    # Compute indicators
    df["ema_200"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["roc"] = df["Close"].pct_change(periods=ROC_PERIOD) * 100

    # Signal generation
    df["signal_long"] = (df["Close"] > df["ema_200"]) & (df["roc"] > 0)
    df["signal_short"] = (df["Close"] < df["ema_200"]) & (df["roc"] < 0)

    # Drop warm-up period (200 candles for EMA)
    df = df.iloc[EMA_PERIOD:].copy()
    df = df.fillna(0)

    return df


# ---------------------------------------------------------------------------
#  CORE SIMULATION ENGINE
# ---------------------------------------------------------------------------
def simulate(df, mode="long_short"):
    """
    Simulate trading with intra-candle stop loss.

    mode: "long_only" or "long_short"

    Returns dict with trades, equity curve, and statistics.
    """
    capital = INITIAL_CAPITAL
    position = 0        # +1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    entry_idx = 0
    trade_capital = 0.0  # Capital allocated to current trade

    trades = []
    equity = []
    funding_total = 0.0
    sl_hits = 0
    normal_exits = 0

    rows = df.reset_index()

    for i in range(len(rows)):
        row = rows.iloc[i]
        ts = row["timestamp"]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        funding_rate = row["funding_rate"]

        # ---- Funding application (every 8 hours) ----
        if position != 0 and i % FUNDING_INTERVAL == 0:
            # Funding cost = position_value * funding_rate
            position_value = trade_capital
            if position == 1:
                # Long pays when rate > 0
                cost = position_value * funding_rate
                capital -= cost
                funding_total -= cost
            elif position == -1:
                # Short earns when rate > 0
                income = position_value * funding_rate
                capital += income
                funding_total += income

        # ---- Intra-candle stop loss check (BEFORE signal logic) ----
        if position == 1:
            sl_price = entry_price * (1 - SL_PCT)
            if low <= sl_price:
                # Stop loss hit on long
                exit_price = sl_price
                pnl_pct = (exit_price / entry_price) - 1
                # Deduct exit fees
                exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
                raw_pnl = trade_capital * pnl_pct
                net_pnl = raw_pnl - exit_cost
                capital += trade_capital + net_pnl

                trades.append({
                    "entry_time": rows.iloc[entry_idx]["timestamp"],
                    "exit_time": ts,
                    "direction": "LONG",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct * 100,
                    "pnl_usd": net_pnl,
                    "exit_reason": "STOP_LOSS",
                    "duration_h": i - entry_idx,
                })
                sl_hits += 1
                position = 0
                entry_price = 0
                trade_capital = 0

        elif position == -1:
            sl_price = entry_price * (1 + SL_PCT)
            if high >= sl_price:
                # Stop loss hit on short
                exit_price = sl_price
                pnl_pct = (entry_price - exit_price) / entry_price
                exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
                raw_pnl = trade_capital * pnl_pct
                net_pnl = raw_pnl - exit_cost
                capital += trade_capital + net_pnl

                trades.append({
                    "entry_time": rows.iloc[entry_idx]["timestamp"],
                    "exit_time": ts,
                    "direction": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct * 100,
                    "pnl_usd": net_pnl,
                    "exit_reason": "STOP_LOSS",
                    "duration_h": i - entry_idx,
                })
                sl_hits += 1
                position = 0
                entry_price = 0
                trade_capital = 0

        # ---- Signal-based exit ----
        if position == 1 and not row["signal_long"]:
            # Exit long: trend reversed
            exit_price = close
            pnl_pct = (exit_price / entry_price) - 1
            exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            raw_pnl = trade_capital * pnl_pct
            net_pnl = raw_pnl - exit_cost
            capital += trade_capital + net_pnl

            trades.append({
                "entry_time": rows.iloc[entry_idx]["timestamp"],
                "exit_time": ts,
                "direction": "LONG",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct * 100,
                "pnl_usd": net_pnl,
                "exit_reason": "SIGNAL",
                "duration_h": i - entry_idx,
            })
            normal_exits += 1
            position = 0
            entry_price = 0
            trade_capital = 0

        elif position == -1 and not row["signal_short"]:
            # Exit short: trend reversed
            exit_price = close
            pnl_pct = (entry_price - exit_price) / entry_price
            exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
            raw_pnl = trade_capital * pnl_pct
            net_pnl = raw_pnl - exit_cost
            capital += trade_capital + net_pnl

            trades.append({
                "entry_time": rows.iloc[entry_idx]["timestamp"],
                "exit_time": ts,
                "direction": "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct * 100,
                "pnl_usd": net_pnl,
                "exit_reason": "SIGNAL",
                "duration_h": i - entry_idx,
            })
            normal_exits += 1
            position = 0
            entry_price = 0
            trade_capital = 0

        # ---- Entry logic (only when flat) ----
        if position == 0:
            if row["signal_long"]:
                # Enter long
                entry_cost = capital * (FEE_PCT + SLIPPAGE_PCT)
                trade_capital = capital - entry_cost
                capital = 0  # All-in (1x, no leverage)
                entry_price = close
                entry_idx = i
                position = 1

            elif mode == "long_short" and row["signal_short"]:
                # Enter short
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

        equity.append({"timestamp": ts, "equity": eq, "position": position})

    # Close any open position at end
    if position != 0:
        final_close = rows.iloc[-1]["Close"]
        if position == 1:
            pnl_pct = (final_close / entry_price) - 1
        else:
            pnl_pct = (entry_price - final_close) / entry_price
        exit_cost = trade_capital * (FEE_PCT + SLIPPAGE_PCT)
        raw_pnl = trade_capital * pnl_pct
        net_pnl = raw_pnl - exit_cost
        capital += trade_capital + net_pnl

        trades.append({
            "entry_time": rows.iloc[entry_idx]["timestamp"],
            "exit_time": rows.iloc[-1]["timestamp"],
            "direction": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": final_close,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": net_pnl,
            "exit_reason": "END_OF_DATA",
            "duration_h": len(rows) - 1 - entry_idx,
        })

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity) if equity else pd.DataFrame()

    return {
        "trades": trades_df,
        "equity": equity_df,
        "funding_total": funding_total,
        "sl_hits": sl_hits,
        "normal_exits": normal_exits,
    }


# ---------------------------------------------------------------------------
#  METRICS COMPUTATION
# ---------------------------------------------------------------------------
def compute_metrics(result, label):
    """Compute and return standard backtest metrics."""
    trades = result["trades"]
    equity = result["equity"]

    if equity.empty:
        return {"label": label, "return_pct": 0, "max_dd": 0, "trades": 0}

    final_eq = equity["equity"].iloc[-1]
    ret = (final_eq / INITIAL_CAPITAL - 1) * 100

    # Max drawdown
    cummax = equity["equity"].cummax()
    dd = (equity["equity"] - cummax) / cummax
    max_dd = dd.min() * 100

    # Sharpe
    rets = equity["equity"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(24 * 365)) if len(rets) > 0 and rets.std() > 0 else 0

    n = len(trades)
    if n > 0 and "pnl_usd" in trades.columns:
        wins = (trades["pnl_usd"] > 0).sum()
        win_rate = wins / n * 100
        gross_p = trades[trades["pnl_usd"] > 0]["pnl_usd"].sum()
        gross_l = abs(trades[trades["pnl_usd"] < 0]["pnl_usd"].sum())
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        avg_win = trades[trades["pnl_usd"] > 0]["pnl_usd"].mean() if wins > 0 else 0
        avg_loss = trades[trades["pnl_usd"] < 0]["pnl_usd"].mean() if (n - wins) > 0 else 0
    else:
        wins = 0
        win_rate = 0
        pf = 0
        avg_win = 0
        avg_loss = 0

    # Direction breakdown
    if n > 0 and "direction" in trades.columns:
        longs = trades[trades["direction"] == "LONG"]
        shorts = trades[trades["direction"] == "SHORT"]
        long_pnl = longs["pnl_usd"].sum() if len(longs) > 0 else 0
        short_pnl = shorts["pnl_usd"].sum() if len(shorts) > 0 else 0
        long_wins = (longs["pnl_usd"] > 0).sum() if len(longs) > 0 else 0
        short_wins = (shorts["pnl_usd"] > 0).sum() if len(shorts) > 0 else 0
    else:
        longs = pd.DataFrame()
        shorts = pd.DataFrame()
        long_pnl = short_pnl = 0
        long_wins = short_wins = 0

    return {
        "label": label,
        "return_pct": ret,
        "final_equity": final_eq,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_trades": n,
        "wins": wins,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "funding_pnl": result["funding_total"],
        "sl_hits": result["sl_hits"],
        "normal_exits": result["normal_exits"],
        "long_trades": len(longs),
        "long_pnl": long_pnl,
        "long_win_rate": (long_wins / len(longs) * 100) if len(longs) > 0 else 0,
        "short_trades": len(shorts),
        "short_pnl": short_pnl,
        "short_win_rate": (short_wins / len(shorts) * 100) if len(shorts) > 0 else 0,
    }


# ---------------------------------------------------------------------------
#  REPORTING
# ---------------------------------------------------------------------------
def print_pair_report(pair_name, long_only, long_short, bh_return):
    """Print detailed comparison for a single pair."""
    print(f"\n{'='*80}")
    print(f"  {pair_name} — PROXY FUTURES BACKTEST RESULTS")
    print(f"{'='*80}")

    print(f"\n  {'Metric':<30} {'Long Only':>15} {'Long + Short':>15}")
    print(f"  {'-'*60}")

    metrics = [
        ("Net Return %", "return_pct", ".2f"),
        ("Final Equity $", "final_equity", ",.0f"),
        ("Max Drawdown %", "max_dd", ".2f"),
        ("Sharpe Ratio", "sharpe", ".2f"),
        ("Total Trades", "total_trades", "d"),
        ("Win Rate %", "win_rate", ".1f"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Avg Win $", "avg_win", ",.0f"),
        ("Avg Loss $", "avg_loss", ",.0f"),
        ("SL Hits", "sl_hits", "d"),
        ("Signal Exits", "normal_exits", "d"),
        ("Funding PnL $", "funding_pnl", ",.2f"),
    ]

    for label, key, fmt in metrics:
        lo_val = long_only.get(key, 0)
        ls_val = long_short.get(key, 0)
        print(f"  {label:<30} {lo_val:>15{fmt}} {ls_val:>15{fmt}}")

    print(f"  {'-'*60}")
    print(f"  {'Buy & Hold Return %':<30} {bh_return:>15.2f}")

    # Direction breakdown for Long+Short
    if long_short["short_trades"] > 0:
        print(f"\n  --- Direction Breakdown (Long + Short mode) ---")
        print(f"  {'Long Trades:':<25} {long_short['long_trades']:>5}  "
              f"PnL: ${long_short['long_pnl']:>+12,.2f}  "
              f"Win Rate: {long_short['long_win_rate']:.1f}%")
        print(f"  {'Short Trades:':<25} {long_short['short_trades']:>5}  "
              f"PnL: ${long_short['short_pnl']:>+12,.2f}  "
              f"Win Rate: {long_short['short_win_rate']:.1f}%")
        
        short_helped = long_short["short_pnl"] > 0
        print(f"\n  🔑 Did shorts add profit? {'✅ YES' if short_helped else '❌ NO'}"
              f" (${long_short['short_pnl']:+,.2f})")


def print_summary(all_results):
    """Print the grand summary table."""
    print(f"\n{'='*100}")
    print(f"  GRAND SUMMARY — PROXY FUTURES BACKTEST (1x, Trend + Momentum)")
    print(f"{'='*100}")

    print(f"\n  {'Pair':<6} {'LO Ret%':>10} {'LS Ret%':>10} {'B&H%':>10} "
          f"{'LO Trades':>10} {'LS Trades':>10} {'Short PnL$':>12} {'Shorts Won?':>12}")
    print(f"  {'-'*85}")

    totals = {"lo_ret": [], "ls_ret": [], "bh": [], "short_pnl": []}

    for r in all_results:
        lo = r["long_only"]
        ls = r["long_short"]
        bh = r["bh_return"]
        short_ok = "✅" if ls["short_pnl"] > 0 else "❌"

        print(f"  {r['pair']:<6} {lo['return_pct']:>+10.2f} {ls['return_pct']:>+10.2f} {bh:>+10.2f} "
              f"{lo['total_trades']:>10} {ls['total_trades']:>10} "
              f"{ls['short_pnl']:>+12,.0f} {short_ok:>12}")

        totals["lo_ret"].append(lo["return_pct"])
        totals["ls_ret"].append(ls["return_pct"])
        totals["bh"].append(bh)
        totals["short_pnl"].append(ls["short_pnl"])

    print(f"  {'-'*85}")
    print(f"  {'AVG':<6} {np.mean(totals['lo_ret']):>+10.2f} {np.mean(totals['ls_ret']):>+10.2f} "
          f"{np.mean(totals['bh']):>+10.2f} {'':>10} {'':>10} "
          f"{np.mean(totals['short_pnl']):>+12,.0f}")

    print(f"\n  SL Hit Summary:")
    for r in all_results:
        ls = r["long_short"]
        total_exits = ls["sl_hits"] + ls["normal_exits"]
        sl_pct = ls["sl_hits"] / total_exits * 100 if total_exits > 0 else 0
        print(f"    {r['pair']}: {ls['sl_hits']} SL hits / {total_exits} total exits ({sl_pct:.0f}% stopped out)")

    # Final verdict
    avg_lo = np.mean(totals["lo_ret"])
    avg_ls = np.mean(totals["ls_ret"])
    avg_short_pnl = np.mean(totals["short_pnl"])

    print(f"\n{'='*100}")
    print(f"  VERDICT:")
    if avg_ls > avg_lo:
        print(f"  ✅ Long+Short ({avg_ls:+.2f}%) OUTPERFORMS Long-Only ({avg_lo:+.2f}%) by {avg_ls - avg_lo:+.2f}%")
    else:
        print(f"  ❌ Long+Short ({avg_ls:+.2f}%) UNDERPERFORMS Long-Only ({avg_lo:+.2f}%) by {avg_ls - avg_lo:.2f}%")

    if avg_short_pnl > 0:
        print(f"  ✅ Short legs ADDED value: avg ${avg_short_pnl:+,.0f} per pair")
    else:
        print(f"  ❌ Short legs BLED money: avg ${avg_short_pnl:+,.0f} per pair")
    print(f"{'='*100}")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  PROXY FUTURES BACKTEST — Long/Short (1x) via Spot Data")
    print("=" * 80)
    print(f"  Config: EMA={EMA_PERIOD}, ROC={ROC_PERIOD}h, SL={SL_PCT*100}%, "
          f"Fee={FEE_PCT*100}%, Slippage={SLIPPAGE_PCT*100}%")
    print(f"  Funding: Applied every {FUNDING_INTERVAL}h")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}")

    all_results = []

    for pair_name, symbol in PAIRS.items():
        print(f"\n{'~'*60}")
        print(f"  Loading {pair_name} ({symbol})...")

        df = load_pair(symbol)
        if df is None:
            continue

        print(f"  Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

        # Buy & Hold baseline
        bh_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

        # Run Long Only
        lo_result = simulate(df, mode="long_only")
        lo_metrics = compute_metrics(lo_result, "Long Only")

        # Run Long + Short
        ls_result = simulate(df, mode="long_short")
        ls_metrics = compute_metrics(ls_result, "Long + Short")

        print_pair_report(pair_name, lo_metrics, ls_metrics, bh_return)

        all_results.append({
            "pair": pair_name,
            "long_only": lo_metrics,
            "long_short": ls_metrics,
            "bh_return": bh_return,
        })

    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
