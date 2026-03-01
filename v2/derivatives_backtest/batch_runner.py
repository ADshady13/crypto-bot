"""
Batch Runner — Run backtests across top 5 pairs with parameter tuning (70/30 split).
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer
from core.signal_generator import BacktestSignalExecutor


class BatchBacktester:
    def __init__(self):
        self.pairs = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "BNB": "BNBUSDT",
            "XRP": "XRPUSDT",
        }
        self.results = {}

    def load_or_fetch(self, symbol, target_candles=10000):
        """Load cached data or fetch fresh."""
        fetcher = DerivativesDataFetcher(symbol)
        return fetcher.fetch_all_data(use_cache=True, target_candles=target_candles)

    def tune_params(self, df, verbose=False):
        """Grid search for best params on training data."""
        param_grid = {
            "sentiment_entry_threshold": [0.6, 0.65, 0.7, 0.75],
            "sentiment_exit_threshold": [0.3, 0.35, 0.4],
            "atr_sl_multiplier": [1.0, 1.5, 2.0],
            "atr_tp_multiplier": [2.0, 3.0, 4.0],
        }

        best_score = -np.inf
        best_params = None
        total = 1
        for v in param_grid.values():
            total *= len(v)

        if verbose:
            print(f"    Testing {total} parameter combinations...")

        count = 0
        for sent_entry in param_grid["sentiment_entry_threshold"]:
            for sent_exit in param_grid["sentiment_exit_threshold"]:
                for atr_sl in param_grid["atr_sl_multiplier"]:
                    for atr_tp in param_grid["atr_tp_multiplier"]:
                        count += 1
                        params = {
                            "sentiment_entry_threshold": sent_entry,
                            "sentiment_exit_threshold": sent_exit,
                            "atr_sl_multiplier": atr_sl,
                            "atr_tp_multiplier": atr_tp,
                        }

                        try:
                            bt = BacktestSignalExecutor(params)
                            trades, equity = bt.run(
                                df, initial_capital=100000
                            )

                            if len(trades) >= 3:
                                # Use Sharpe ratio on equity curve as objective
                                returns = equity["equity"].pct_change().dropna()
                                if len(returns) > 0 and returns.std() > 0:
                                    sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
                                    if sharpe > best_score:
                                        best_score = sharpe
                                        best_params = params.copy()
                        except Exception:
                            continue

        if verbose and best_params:
            print(f"    Best train Sharpe: {best_score:.2f}")

        return best_params, best_score

    def run_pair(self, pair_name, symbol, target_candles=10000):
        """Run full backtest for a single pair with train/test split."""
        print(f"\n{'='*60}")
        print(f"  PAIR: {pair_name} ({symbol})")
        print(f"{'='*60}")

        # Load data
        df = self.load_or_fetch(symbol, target_candles)
        if df is None or df.empty or len(df) < 500:
            print(f"  Skipping {pair_name} — insufficient data ({len(df) if df is not None else 0} rows)")
            return None

        # Engineer features
        print("  Engineering features...")
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)
        df = engineer.compute_sentiment_score(df)

        # 70/30 split
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        # Tune on training data
        print("  Tuning parameters on training set...")
        best_params, train_sharpe = self.tune_params(train_df, verbose=True)

        if best_params is None:
            best_params = {
                "sentiment_entry_threshold": 0.65,
                "sentiment_exit_threshold": 0.35,
                "atr_sl_multiplier": 1.5,
                "atr_tp_multiplier": 3.0,
            }
            train_sharpe = 0
            print("    No profitable params found, using defaults")

        print(f"  Best params: {best_params}")

        # Test on holdout
        print("  Testing on holdout set...")
        bt = BacktestSignalExecutor(best_params)
        test_trades, test_equity = bt.run(test_df, initial_capital=100000)

        # Compute test metrics
        test_return = 0
        test_sharpe = 0
        test_win_rate = 0
        test_max_dd = 0
        test_profit_factor = 0

        if not test_equity.empty and len(test_trades) > 0:
            test_return = (
                (test_equity["equity"].iloc[-1] - 100000) / 100000 * 100
            )
            rets = test_equity["equity"].pct_change().dropna()
            test_sharpe = (
                rets.mean() / rets.std() * np.sqrt(24 * 365)
                if rets.std() > 0
                else 0
            )

            if "pnl" in test_trades.columns:
                wins = (test_trades["pnl"] > 0).sum()
                test_win_rate = wins / len(test_trades) * 100
                gross_p = test_trades[test_trades["pnl"] > 0]["pnl"].sum()
                gross_l = abs(test_trades[test_trades["pnl"] < 0]["pnl"].sum())
                test_profit_factor = gross_p / gross_l if gross_l > 0 else 0

            test_max_dd = (
                (test_equity["equity"] - test_equity["equity"].cummax())
                / test_equity["equity"].cummax()
            ).min() * 100

        # Buy & Hold for comparison
        bh_return = (
            (test_df["Close"].iloc[-1] - test_df["Close"].iloc[0])
            / test_df["Close"].iloc[0]
            * 100
            if len(test_df) > 0
            else 0
        )

        result = {
            "pair": pair_name,
            "symbol": symbol,
            "total_candles": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "best_params": best_params,
            "train_sharpe": train_sharpe,
            "test_return_pct": test_return,
            "test_sharpe": test_sharpe,
            "test_trades": len(test_trades),
            "test_win_rate": test_win_rate,
            "test_max_dd": test_max_dd,
            "test_profit_factor": test_profit_factor,
            "buy_hold_return": bh_return,
            "test_equity": test_equity,
        }

        print(f"\n  TEST RESULTS ({pair_name}):")
        print(f"    Return:        {test_return:>+10.2f}%  (Buy&Hold: {bh_return:>+.2f}%)")
        print(f"    Sharpe:        {test_sharpe:>10.2f}")
        print(f"    Trades:        {len(test_trades):>10}")
        print(f"    Win Rate:      {test_win_rate:>10.1f}%")
        print(f"    Profit Factor: {test_profit_factor:>10.2f}")
        print(f"    Max Drawdown:  {test_max_dd:>10.2f}%")

        return result

    def run_all(self, target_candles=10000):
        """Run backtests for all pairs."""
        all_results = []

        for pair_name, symbol in self.pairs.items():
            try:
                result = self.run_pair(pair_name, symbol, target_candles)
                if result:
                    all_results.append(result)
                    self.results[pair_name] = result
            except Exception as e:
                print(f"  Error for {pair_name}: {e}")
                import traceback
                traceback.print_exc()

        return all_results

    def print_summary(self, results):
        """Print summary table."""
        print("\n" + "=" * 90)
        print("BATCH BACKTEST SUMMARY — 10K CANDLES, 70/30 SPLIT, TUNED PARAMS")
        print("=" * 90)
        print(
            f"{'Pair':<6} {'Return%':>10} {'B&H%':>10} {'Sharpe':>8} "
            f"{'Trades':>7} {'WinRate':>8} {'PF':>8} {'MaxDD%':>8}"
        )
        print("-" * 90)

        for r in results:
            print(
                f"{r['pair']:<6} {r['test_return_pct']:>+10.2f} "
                f"{r['buy_hold_return']:>+10.2f} "
                f"{r['test_sharpe']:>8.2f} "
                f"{r['test_trades']:>7} "
                f"{r['test_win_rate']:>7.1f}% "
                f"{r['test_profit_factor']:>8.2f} "
                f"{r['test_max_dd']:>8.2f}"
            )

        # Averages
        if results:
            print("-" * 90)
            avg = lambda k: np.mean([r[k] for r in results])
            print(
                f"{'AVG':<6} {avg('test_return_pct'):>+10.2f} "
                f"{avg('buy_hold_return'):>+10.2f} "
                f"{avg('test_sharpe'):>8.2f} "
                f"{avg('test_trades'):>7.0f} "
                f"{avg('test_win_rate'):>7.1f}% "
                f"{avg('test_profit_factor'):>8.2f} "
                f"{avg('test_max_dd'):>8.2f}"
            )
        print("=" * 90)

    def save_results(self, results):
        """Save results to CSV."""
        save_path = "derivatives_backtest/results"
        os.makedirs(save_path, exist_ok=True)

        # Summary table
        summary = [
            {k: v for k, v in r.items() if k not in ["test_equity", "best_params"]}
            for r in results
        ]
        pd.DataFrame(summary).to_csv(f"{save_path}/batch_summary.csv", index=False)

        # Per-pair params
        params_data = [
            {"pair": r["pair"], **r["best_params"]}
            for r in results
            if r.get("best_params")
        ]
        pd.DataFrame(params_data).to_csv(f"{save_path}/tuned_params.csv", index=False)

        # Per-pair equity curves
        for r in results:
            if "test_equity" in r and not r["test_equity"].empty:
                r["test_equity"].to_csv(f"{save_path}/{r['pair']}_equity.csv")

        print(f"\nResults saved to {save_path}/")


def main():
    batch = BatchBacktester()
    results = batch.run_all(target_candles=10000)

    if results:
        batch.print_summary(results)
        batch.save_results(results)
    else:
        print("No results.")


if __name__ == "__main__":
    main()
