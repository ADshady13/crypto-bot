"""
Backtester — Full pipeline: fetch → features → signals → metrics → plot.
Runs on 10K hourly candles for a single pair.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_fetcher import DerivativesDataFetcher
from core.feature_engineering import FeatureEngineer
from core.signal_generator import BacktestSignalExecutor


class DerivativesBacktester:
    def __init__(self, params=None):
        self.params = params or {
            # Feature params
            "zscore_period": 30,
            "funding_percentile_period": 90,
            "atr_period": 14,
            "ema_period": 50,
            "volume_zscore_period": 30,
            "roc_period": 12,
            "fg_zscore_period": 30,
            # Signal params (primary triggers)
            "sentiment_entry_threshold": 0.65,
            "sentiment_exit_threshold": 0.35,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 3.0,
            # Capital
            "initial_capital": 100000,
        }
        self.results = {}

    def run(self, symbol="BTCUSDT", target_candles=10000, start_time_ms=None, verbose=True):
        """Run full backtest pipeline."""

        if verbose:
            print("=" * 60)
            print("DERIVATIVES SENTIMENT BACKTEST")
            print("=" * 60)
            print(f"Symbol: {symbol}")
            print(f"Target candles: {target_candles}")
            print(f"Capital: ${self.params['initial_capital']:,.2f}")
            print("-" * 60)

        # Step 1: Fetch data
        if verbose:
            print("\n[1/4] Fetching data...")
        fetcher = DerivativesDataFetcher(symbol=symbol)
        # Assuming fetch_all_data might need modification to accept target_start_time if we want precise alignment
        # For now, let's just use what we have in data_fetcher
        df = fetcher.fetch_all_data(target_candles=target_candles)

        if df.empty or len(df) < 100:
            print(f"ERROR: Insufficient data ({len(df)} rows)")
            return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Step 2: Feature engineering
        if verbose:
            print("\n[2/4] Engineering features...")
        feature_params = {
            k: self.params[k]
            for k in [
                "zscore_period",
                "funding_percentile_period",
                "atr_period",
                "ema_period",
                "volume_zscore_period",
                "roc_period",
                "fg_zscore_period",
            ]
            if k in self.params
        }
        engineer = FeatureEngineer(feature_params)
        df = engineer.engineer_features(df)
        df = engineer.compute_sentiment_score(df)

        # Step 3: Run backtest
        if verbose:
            print("\n[3/4] Running backtest...")
        signal_params = {
            k: self.params[k]
            for k in [
                "sentiment_entry_threshold",
                "sentiment_exit_threshold",
                "atr_sl_multiplier",
                "atr_tp_multiplier",
            ]
            if k in self.params
        }
        executor = BacktestSignalExecutor(signal_params)
        trades_df, equity_df = executor.run(df, self.params["initial_capital"])

        # Step 4: Calculate metrics
        if verbose:
            print("\n[4/4] Calculating metrics...")
        self.results = self._calculate_metrics(df, trades_df, equity_df)

        if verbose:
            self._print_results()

        return self.results, df, trades_df, equity_df

    def _calculate_metrics(self, df, trades_df, equity_df):
        """Calculate comprehensive backtest metrics."""

        initial_capital = self.params["initial_capital"]

        if equity_df.empty:
            return {"initial_capital": initial_capital, "final_equity": initial_capital}

        final_equity = equity_df["equity"].iloc[-1]

        # Time span
        start_date = df.index[0]
        end_date = df.index[-1]
        days = (end_date - start_date).days
        years = days / 365 if days > 0 else 1

        # Returns
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        cagr = (
            ((final_equity / initial_capital) ** (1 / years) - 1) * 100
            if years > 0 and final_equity > 0
            else 0
        )

        # Trade metrics
        num_trades = len(trades_df)
        winning = 0
        losing = 0
        gross_profit = 0
        gross_loss = 0
        avg_r = 0

        if num_trades > 0 and "pnl" in trades_df.columns:
            winning = (trades_df["pnl"] > 0).sum()
            losing = (trades_df["pnl"] <= 0).sum()
            gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
            gross_loss_val = trades_df[trades_df["pnl"] < 0]["pnl"].sum()
            gross_loss = abs(gross_loss_val) if gross_loss_val != 0 else 1
            avg_r = (
                trades_df["return_r"].mean()
                if "return_r" in trades_df.columns
                else 0
            )

        win_rate = (winning / num_trades * 100) if num_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Risk metrics
        returns = equity_df["equity"].pct_change().dropna()
        sharpe = (
            (returns.mean() / returns.std() * np.sqrt(24 * 365))
            if len(returns) > 0 and returns.std() > 0
            else 0
        )

        # Max drawdown
        cummax = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Buy & Hold
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        buy_hold_return = ((end_price - start_price) / start_price) * 100

        return {
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "candles": len(df),
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "buy_hold_return": buy_hold_return,
            "num_trades": num_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_r": avg_r,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
        }

    def _print_results(self):
        """Print formatted results."""
        r = self.results

        start = r.get("start_date")
        end = r.get("end_date")
        start_str = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
        end_str = end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period:          {start_str} to {end_str} ({r.get('days', 0)} days)")
        print(f"Candles:         {r.get('candles', 0)}")
        print("-" * 60)
        print(f"Initial Capital: ${r['initial_capital']:>15,.2f}")
        print(f"Final Equity:    ${r['final_equity']:>15,.2f}")
        print(f"Total Return:    {r['total_return_pct']:>14.2f}%")
        print(f"CAGR:            {r['cagr']:>14.2f}%")
        print(f"Buy & Hold:      {r['buy_hold_return']:>14.2f}%")
        print("-" * 60)
        print(f"# Trades:        {r['num_trades']:>15}")
        print(f"Win Rate:        {r['win_rate']:>14.2f}%")
        print(f"Profit Factor:   {r['profit_factor']:>14.2f}")
        print(f"Avg R Multiple:  {r['avg_r']:>14.2f}")
        print("-" * 60)
        print(f"Sharpe Ratio:    {r['sharpe_ratio']:>14.2f}")
        print(f"Max Drawdown:    {r['max_drawdown']:>14.2f}%")
        print("=" * 60)

    def save_results(self, save_path="derivatives_backtest/results"):
        """Save trades, equity curve, and summary to CSV."""
        os.makedirs(save_path, exist_ok=True)
        r = self.results

        # This method expects run() to have been called
        # The caller should pass trades_df and equity_df if needed
        summary = {k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}
        summary_file = os.path.join(save_path, "summary.csv")
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")


def main():
    bt = DerivativesBacktester()
    results, df, trades, equity = bt.run(symbol="BTCUSDT", target_candles=10000)

    if not equity.empty:
        bt.save_results()
        # Save equity curve
        equity.to_csv("derivatives_backtest/results/equity_curve.csv")
        if not trades.empty:
            trades.to_csv("derivatives_backtest/results/trades.csv", index=False)
        print("\nResults saved.")


if __name__ == "__main__":
    main()
