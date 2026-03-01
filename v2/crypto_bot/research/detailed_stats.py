import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer
from core.signal_generator import BacktestSignalExecutor


def run_detailed_backtest(symbol, pair_name, params):
    """Run detailed backtest with full stats"""
    print(f"\n{'=' * 70}")
    print(f"DETAILED BACKTEST: {pair_name} ({symbol})")
    print("=" * 70)

    # Load cached data
    cache_path = f"derivatives_backtest/data/{symbol}_data.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="timestamp", parse_dates=True)
    else:
        print(f"  No cache found")
        return None

    print(f"  Data: {len(df)} rows")

    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)

    # 70/30 split
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"  Date range: {test_df.index[0]} to {test_df.index[-1]}")

    # Count signals
    signals = (
        (test_df["sentiment_score"] > params["sentiment_entry_threshold"])
        & (test_df["funding_zscore"] < params["funding_zscore_entry"])
        & (test_df["long_short_ratio_zscore"] < params["lsr_zscore_entry"])
        & (test_df["price_above_ema"] == True)
    )
    total_signals = signals.sum()
    print(f"\n  Signal Analysis:")
    print(f"    Total entry signals: {total_signals}")

    # Run backtest on test data
    bt = BacktestSignalExecutor(params)
    trades, equity = bt.run(test_df, initial_capital=100000)

    executed_trades = len(trades)
    print(f"    Executed trades: {executed_trades}")
    print(f"    Signal execution rate: {executed_trades / total_signals * 100:.1f}%")

    if len(trades) == 0:
        print(f"\n  No trades executed!")
        return None

    # Detailed trade stats
    print(f"\n  Trade Statistics:")
    print(f"    Total trades: {len(trades)}")

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    print(f"    Winning trades: {len(wins)}")
    print(f"    Losing trades: {len(losses)}")
    print(f"    Win rate: {len(wins) / len(trades) * 100:.1f}%")

    # PnL stats
    total_pnl = trades["pnl"].sum()
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0

    print(f"\n  PnL Statistics:")
    print(f"    Total PnL: ${total_pnl:,.2f}")
    print(f"    Gross Profit: ${gross_profit:,.2f}")
    print(f"    Gross Loss: ${gross_loss:,.2f}")
    print(
        f"    Profit Factor: {gross_profit / gross_loss if gross_loss > 0 else 0:.2f}"
    )

    if len(wins) > 0:
        print(f"    Avg Win: ${wins['pnl'].mean():,.2f}")
        print(f"    Max Win: ${wins['pnl'].max():,.2f}")
    if len(losses) > 0:
        print(f"    Avg Loss: ${losses['pnl'].mean():,.2f}")
        print(f"    Max Loss: ${losses['pnl'].min():,.2f}")

    # Return stats
    returns_pct = trades["pnl_pct"].dropna()
    if len(returns_pct) > 0:
        print(f"\n  Return Statistics:")
        print(f"    Avg Return: {returns_pct.mean():.2f}%")
        print(f"    Max Return: {returns_pct.max():.2f}%")
        print(f"    Min Return: {returns_pct.min():.2f}%")

    # R multiple stats
    r_mults = trades["return_r"].dropna()
    if len(r_mults) > 0:
        print(f"\n  Risk/R Multiple Stats:")
        print(f"    Avg R: {r_mults.mean():.2f}")
        print(f"    Max R: {r_mults.max():.2f}")
        print(f"    Min R: {r_mults.min():.2f}")

    # Trade duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations = []
        for _, trade in trades.iterrows():
            if trade["entry_time"] and trade["exit_time"]:
                dur = (trade["exit_time"] - trade["entry_time"]).total_seconds() / 3600
                durations.append(dur)
        if durations:
            print(f"\n  Trade Duration:")
            print(f"    Avg duration: {np.mean(durations):.1f} hours")
            print(f"    Max duration: {max(durations):.1f} hours")
            print(f"    Min duration: {min(durations):.1f} hours")

    # Exit reasons
    print(f"\n  Exit Reasons:")
    exit_reasons = trades["exit_reason"].value_counts()
    for reason, count in exit_reasons.items():
        print(f"    {reason}: {count}")

    # Equity curve
    initial = 100000
    final = equity["equity"].iloc[-1]
    ret_pct = (final - initial) / initial * 100

    print(f"\n  Equity Performance:")
    print(f"    Initial: ${initial:,.2f}")
    print(f"    Final: ${final:,.2f}")
    print(f"    Return: {ret_pct:.2f}%")

    # Drawdown
    cummax = equity["equity"].cummax()
    dd = (equity["equity"] - cummax) / cummax * 100
    max_dd = dd.min()
    print(f"    Max Drawdown: {max_dd:.2f}%")

    # Sample trades
    print(f"\n  Sample Trades (first 5):")
    print(
        f"    {'Entry Time':<20} {'Entry':<12} {'Exit':<12} {'PnL':<12} {'R':<8} {'Reason'}"
    )
    print(f"    {'-' * 80}")
    for i, row in trades.head(5).iterrows():
        et = str(row.get("entry_time", ""))[:19] if row.get("entry_time") else "N/A"
        ep = f"${row.get('entry_price', 0):,.0f}" if row.get("entry_price") else "N/A"
        ex = str(row.get("exit_time", ""))[:19] if row.get("exit_time") else "N/A"
        pnl = f"${row.get('pnl', 0):,.0f}" if row.get("pnl") else "N/A"
        r = f"{row.get('return_r', 0):.2f}" if row.get("return_r") else "N/A"
        reason = row.get("exit_reason", "N/A")[:15]
        print(f"    {et:<20} {ep:<12} {ex:<12} {pnl:<12} {r:<8} {reason}")

    return {
        "pair": pair_name,
        "symbol": symbol,
        "params": params,
        "test_rows": len(test_df),
        "total_signals": total_signals,
        "executed_trades": executed_trades,
        "win_rate": len(wins) / len(trades) * 100,
        "total_pnl": total_pnl,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "max_win": wins["pnl"].max() if len(wins) > 0 else 0,
        "max_loss": losses["pnl"].min() if len(losses) > 0 else 0,
        "return_pct": ret_pct,
        "max_drawdown": max_dd,
        "trades": trades,
    }


if __name__ == "__main__":
    # Winning pairs with tuned params
    pairs = {
        "SOL": (
            "SOLUSDT",
            {
                "sentiment_entry_threshold": 0.6,
                "sentiment_exit_threshold": 0.2,
                "funding_zscore_entry": -1.0,
                "lsr_zscore_entry": -0.5,
                "atr_sl_multiplier": 1.0,
                "atr_tp_multiplier": 3.0,
            },
        ),
        "XRP": (
            "XRPUSDT",
            {
                "sentiment_entry_threshold": 0.7,
                "sentiment_exit_threshold": 0.2,
                "funding_zscore_entry": -0.5,
                "lsr_zscore_entry": 0.0,
                "atr_sl_multiplier": 1.0,
                "atr_tp_multiplier": 3.0,
            },
        ),
        "BNB": (
            "BNBUSDT",
            {
                "sentiment_entry_threshold": 0.5,
                "sentiment_exit_threshold": 0.2,
                "funding_zscore_entry": -0.5,
                "lsr_zscore_entry": 0.0,
                "atr_sl_multiplier": 1.5,
                "atr_tp_multiplier": 3.0,
            },
        ),
    }

    results = []
    for pair_name, (symbol, params) in pairs.items():
        result = run_detailed_backtest(symbol, pair_name, params)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - WINNING PAIRS")
    print("=" * 70)
    print(
        f"{'Pair':<8} {'Signals':<10} {'Trades':<8} {'Win%':<8} {'PnL':<12} {'Return%':<10} {'MaxDD%':<10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['pair']:<8} {r['total_signals']:<10} {r['executed_trades']:<8} {r['win_rate']:<7.1f}% ${r['total_pnl']:<10,.0f} {r['return_pct']:<9.2f}% {r['max_drawdown']:<9.2f}%"
        )
    print("=" * 70)
