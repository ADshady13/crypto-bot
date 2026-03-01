import pandas as pd
import numpy as np
from itertools import product
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import DerivativesBacktester


class Optimizer:
    def __init__(self, param_grid=None):
        self.param_grid = param_grid or {
            "sentiment_entry_threshold": [0.6, 0.65, 0.7, 0.75, 0.8],
            "sentiment_exit_threshold": [0.3, 0.35, 0.4, 0.45],
            "funding_zscore_entry": [-1.0, -1.5, -2.0],
            "lsr_zscore_entry": [-0.5, -1.0, -1.5],
            "atr_sl_multiplier": [1.0, 1.5, 2.0],
            "zscore_period": [20, 30, 40],
            "ema_period": [30, 50, 70],
        }

        self.results = []

    def grid_search(
        self, symbol="BTC/USDT:USDT", limit=10000, metric="sharpe_ratio", verbose=True
    ):
        """
        Run grid search over parameter combinations.

        Args:
            symbol: Trading symbol
            limit: Number of candles
            metric: Optimization metric (sharpe_ratio, total_return, win_rate)
        """

        # Generate all parameter combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(product(*values))

        total_combinations = len(combinations)

        if verbose:
            print("=" * 60)
            print("GRID SEARCH OPTIMIZATION")
            print("=" * 60)
            print(f"Total combinations: {total_combinations}")
            print(f"Optimizing for: {metric}")
            print("-" * 60)

        best_score = -np.inf
        best_params = None
        best_results = None

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            if verbose:
                print(f"\n[{i + 1}/{total_combinations}] Testing: {params}")

            try:
                # Run backtest with these params
                bt = DerivativesBacktester(params)
                results, _, _, _ = bt.run(symbol=symbol, limit=limit, verbose=False)

                # Get optimization metric
                score = results.get(metric, 0)

                # Store results
                result_entry = {
                    "params": params,
                    metric: score,
                    "total_return_pct": results["total_return_pct"],
                    "sharpe_ratio": results["sharpe_ratio"],
                    "win_rate": results["win_rate"],
                    "max_drawdown": results["max_drawdown"],
                    "num_trades": results["num_trades"],
                }
                self.results.append(result_entry)

                if verbose:
                    print(
                        f"  {metric}: {score:.2f}, Return: {results['total_return_pct']:.2f}%, Trades: {results['num_trades']}"
                    )

                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_results = results

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

        if verbose:
            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"Best {metric}: {best_score:.2f}")
            print(f"Best params: {best_params}")

        return best_params, best_results

    def get_top_n(self, n=10, metric="sharpe_ratio"):
        """Get top N parameter combinations"""
        if not self.results:
            return []

        df = pd.DataFrame(self.results)
        df_sorted = df.sort_values(metric, ascending=False)
        return df_sorted.head(n)

    def save_results(
        self, filepath="derivatives_backtest/results/optimization_results.csv"
    ):
        """Save optimization results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filepath, index=False)
            print(f"Optimization results saved to: {filepath}")


def main():
    # Quick optimization with reduced grid for testing
    param_grid = {
        "sentiment_entry_threshold": [0.6, 0.7, 0.8],
        "sentiment_exit_threshold": [0.3, 0.4],
        "funding_zscore_entry": [-1.5],
        "lsr_zscore_entry": [-1.0],
        "atr_sl_multiplier": [1.5],
        "zscore_period": [30],
        "ema_period": [50],
    }

    optimizer = Optimizer(param_grid)
    best_params, best_results = optimizer.grid_search(
        symbol="BTC/USDT:USDT",
        limit=5000,  # Reduced for faster testing
        metric="sharpe_ratio",
    )

    # Save results
    optimizer.save_results()

    print("\nTop 5 configurations:")
    print(optimizer.get_top_n(5))


if __name__ == "__main__":
    main()
