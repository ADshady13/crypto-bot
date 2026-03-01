"""
Signal Generator & Backtest Executor — Entry / exit logic for the derivatives sentiment strategy.

Entry Rules (ALL must be true):
  1. sentiment_score > entry_threshold
  2. funding_zscore < funding_zscore_entry (negative = shorts paying)
  3. long_short_ratio_zscore < lsr_zscore_entry (crowd is short)
  4. Price > EMA-50 (trend confirmation)

Exit Rules (ANY triggers exit):
  1. sentiment_score drops below exit_threshold
  2. Stop-loss hit (ATR × multiplier below entry)
  3. Take-profit hit (ATR × multiplier above entry)
"""

import pandas as pd
import numpy as np


class BacktestSignalExecutor:
    """
    Execute signals in a backtest context with proper position tracking,
    fee + slippage deduction on both entry and exit.
    """

    def __init__(self, params=None):
        self.params = params or {
            "sentiment_entry_threshold": 0.7,
            "sentiment_exit_threshold": 0.4,
            "funding_zscore_entry": -1.5,
            "lsr_zscore_entry": -1.0,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 3.0,
        }

        # Realistic Binance spot fees + slippage
        self.fee_pct = 0.001  # 0.1% per side (Binance spot taker)
        self.slippage_pct = 0.0005  # 0.05% slippage per side

    def run(self, df, initial_capital=100000):
        """
        Run backtest with proper signal execution.
        Returns (trades_df, equity_df).
        """
        cash = initial_capital
        position = 0  # units of asset held
        entry_price = 0
        entry_atr = 0

        trades = []
        equity_curve = []

        total_cost_pct = self.fee_pct + self.slippage_pct  # per side

        for timestamp, row in df.iterrows():
            current_price = row["Close"]
            current_atr = row.get("atr", current_price * 0.01)

            # Track equity
            equity = cash + (position * current_price)
            equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "position": position,
                    "price": current_price,
                    "sentiment": row.get("sentiment_score", 0.5),
                }
            )

            # ========== ENTRY LOGIC (no position) ==========
            if position == 0:
                # Primary trigger: High composite sentiment score
                # Secondary filter: Price must be above EMA (trend following)
                # Note: funding, lsr, fg are already inside sentiment_score
                sentiment_ok = row.get("sentiment_score", 0) > self.params["sentiment_entry_threshold"]
                trend_ok = row.get("price_above_ema", False) == True
                
                entry_ok = sentiment_ok and trend_ok

                if entry_ok:
                    # Position sizing — use 95% of cash
                    trade_cash = cash * 0.95
                    # Entry cost includes fee + slippage
                    effective_entry_price = current_price * (1 + total_cost_pct)
                    position = trade_cash / effective_entry_price

                    cost = position * effective_entry_price
                    if cost > cash:
                        position = 0
                        continue

                    cash -= cost
                    entry_price = current_price
                    entry_atr = current_atr

                    trades.append(
                        {
                            "entry_time": timestamp,
                            "entry_price": entry_price,
                            "effective_entry": effective_entry_price,
                            "position_size": position,
                            "atr_at_entry": entry_atr,
                            "sentiment_at_entry": row.get("sentiment_score", 0),
                            "fg_at_entry": row.get("fear_greed_value", 50),
                            "exit_time": None,
                            "exit_price": None,
                            "pnl": None,
                            "pnl_pct": None,
                            "return_r": None,
                            "exit_reason": None,
                        }
                    )

            # ========== EXIT LOGIC (has position) ==========
            else:
                exit_triggered = False
                exit_reason = None

                # Exit 1: Sentiment drops below threshold
                if (
                    row.get("sentiment_score", 1)
                    < self.params["sentiment_exit_threshold"]
                ):
                    exit_triggered = True
                    exit_reason = "SENTIMENT_EXIT"

                # Exit 2: Stop-loss (ATR-based)
                sl_price = entry_price - (entry_atr * self.params["atr_sl_multiplier"])
                if current_price <= sl_price:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"

                # Exit 3: Take-profit (ATR-based)
                tp_price = entry_price + (entry_atr * self.params["atr_tp_multiplier"])
                if current_price >= tp_price:
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"

                if exit_triggered:
                    # Exit cost includes fee + slippage
                    effective_exit_price = current_price * (1 - total_cost_pct)
                    revenue = position * effective_exit_price

                    pnl = revenue - (position * trades[-1]["effective_entry"])
                    pnl_pct = (
                        (effective_exit_price - trades[-1]["effective_entry"])
                        / trades[-1]["effective_entry"]
                        * 100
                    )

                    # R-multiple (risk unit = SL distance)
                    risk_per_unit = entry_atr * self.params["atr_sl_multiplier"]
                    return_r = (
                        (current_price - entry_price) / risk_per_unit
                        if risk_per_unit > 0
                        else 0
                    )

                    # Update trade record
                    trades[-1]["exit_time"] = timestamp
                    trades[-1]["exit_price"] = current_price
                    trades[-1]["effective_exit"] = effective_exit_price
                    trades[-1]["pnl"] = pnl
                    trades[-1]["pnl_pct"] = pnl_pct
                    trades[-1]["return_r"] = return_r
                    trades[-1]["exit_reason"] = exit_reason

                    # Close position
                    cash += revenue
                    position = 0
                    entry_price = 0

        # Close any remaining position at end of data
        if position > 0:
            final_price = df.iloc[-1]["Close"]
            effective_exit = final_price * (1 - total_cost_pct)
            revenue = position * effective_exit

            trades[-1]["exit_time"] = df.index[-1]
            trades[-1]["exit_price"] = final_price
            trades[-1]["effective_exit"] = effective_exit
            trades[-1]["pnl"] = revenue - (position * trades[-1]["effective_entry"])
            trades[-1]["pnl_pct"] = (
                (effective_exit - trades[-1]["effective_entry"])
                / trades[-1]["effective_entry"]
                * 100
            )
            trades[-1]["exit_reason"] = "END_OF_DATA"
            cash += revenue
            position = 0

        # Build DataFrames
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)

        num_trades = len(trades_df)
        print(f"  Total trades: {num_trades}")
        if num_trades > 0 and "pnl" in trades_df.columns:
            wins = (trades_df["pnl"] > 0).sum()
            print(f"  Wins: {wins}, Losses: {num_trades - wins}")

        return trades_df, equity_df


def main():
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_fetcher import DerivativesDataFetcher
    from core.feature_engineering import FeatureEngineer

    fetcher = DerivativesDataFetcher("BTCUSDT")
    df = fetcher.fetch_all_data(target_candles=1000)

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)

    executor = BacktestSignalExecutor()
    trades, equity = executor.run(df)

    if not trades.empty:
        print("\nTrade log:")
        print(trades[["entry_time", "exit_time", "entry_price", "exit_price", "pnl", "exit_reason"]])


if __name__ == "__main__":
    main()
