import pandas as pd
import numpy as np


class RiskManager:
    def __init__(self, params=None):
        self.params = params or {
            "risk_per_trade_pct": 0.01,  # 1% risk per trade
            "fee_pct": 0.0004,  # 0.04% per side (0.08% total round trip)
            "slippage_pct": 0.0002,  # 0.02% slippage per trade
            "max_positions": 1,
        }

    def calculate_position_size(self, capital, entry_price, stop_distance):
        """
        Calculate position size based on risk amount.

        Risk = Position Size * Stop Distance
        Position Size = Risk / Stop Distance

        Returns: position_size (in units of asset)
        """
        risk_amount = capital * self.params["risk_per_trade_pct"]

        # Position in units
        position_size = risk_amount / stop_distance

        # Convert to quantity
        quantity = position_size / entry_price

        return quantity

    def apply_fees_and_slippage(self, entry_price, exit_price, quantity, side="long"):
        """
        Apply fees and slippage to entry and exit prices.

        Entry: Price * (1 - slippage) - fee
        Exit: Price * (1 - slippage) - fee
        """
        slippage = self.params["slippage_pct"]
        fee = self.params["fee_pct"]

        if side == "long":
            adjusted_entry = entry_price * (1 - slippage) * (1 + fee)
            adjusted_exit = exit_price * (1 - slippage) * (1 - fee)
        else:
            adjusted_entry = entry_price * (1 + slippage) * (1 + fee)
            adjusted_exit = exit_price * (1 + slippage) * (1 - fee)

        return adjusted_entry, adjusted_exit

    def check_daily_loss_limit(self, daily_pnl, capital, max_loss_pct=0.05):
        """Check if daily loss exceeds limit"""
        loss_pct = daily_pnl / capital
        return loss_pct < -max_loss_pct

    def check_max_drawdown(self, equity_curve, max_dd_pct=0.15):
        """Check if max drawdown exceeds limit"""
        if len(equity_curve) < 2:
            return False

        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()

        return max_dd < -max_dd_pct

    def validate_trade(self, trade, capital):
        """
        Validate trade parameters before execution.
        Returns: (is_valid, reason)
        """
        if trade["position_size"] <= 0:
            return False, "Invalid position size"

        # Check position value doesn't exceed capital
        position_value = trade["position_size"] * trade["entry_price"]
        if position_value > capital * 1.1:  # Allow 10% leverage
            return False, "Position exceeds capital"

        # Check stop distance is reasonable (not too tight)
        if trade.get("stop_distance", 0) < 0:
            return False, "Invalid stop distance"

        return True, "OK"


class PortfolioManager:
    """Manage overall portfolio risk and position tracking"""

    def __init__(self, initial_capital, params=None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.closed_trades = []
        self.risk_manager = RiskManager(params)

        # Track daily PnL
        self.daily_pnl = {}

    def open_position(self, symbol, entry_price, position_size, stop_distance, atr):
        """Open a new position"""
        if len(self.positions) >= self.risk_manager.params["max_positions"]:
            return None, "Max positions reached"

        # Calculate position size
        size = self.risk_manager.calculate_position_size(
            self.current_capital, entry_price, stop_distance
        )

        # Adjust for fees
        adjusted_entry, _ = self.risk_manager.apply_fees_and_slippage(
            entry_price, entry_price, size
        )

        position = {
            "symbol": symbol,
            "entry_price": adjusted_entry,
            "size": size,
            "entry_time": None,  # Set by caller
            "stop_loss": adjusted_entry - stop_distance,
            "take_profit": adjusted_entry + (atr * 3),
            "atr": atr,
        }

        self.positions[symbol] = position

        # Deduct cost from capital
        self.current_capital -= adjusted_entry * size

        return position, "OK"

    def close_position(self, symbol, exit_price):
        """Close an existing position"""
        if symbol not in self.positions:
            return None, "No position to close"

        position = self.positions[symbol]

        # Adjust for fees
        _, adjusted_exit = self.risk_manager.apply_fees_and_slippage(
            position["entry_price"], exit_price, position["size"]
        )

        # Calculate PnL
        pnl = (adjusted_exit - position["entry_price"]) * position["size"]

        # Return capital + PnL
        self.current_capital += adjusted_exit * position["size"]

        # Record closed trade
        trade = position.copy()
        trade["exit_price"] = adjusted_exit
        trade["exit_time"] = None  # Set by caller
        trade["pnl"] = pnl
        trade["pnl_pct"] = pnl / (position["entry_price"] * position["size"]) * 100

        self.closed_trades.append(trade)
        del self.positions[symbol]

        return trade, "OK"

    def get_equity(self, current_prices):
        """Calculate current portfolio equity"""
        position_value = sum(
            p["size"] * current_prices.get(p["symbol"], p["entry_price"])
            for p in self.positions.values()
        )
        return self.current_capital + position_value

    def get_stats(self):
        """Get portfolio statistics"""
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
            }

        trades_df = pd.DataFrame(self.closed_trades)

        wins = (trades_df["pnl"] > 0).sum()
        total = len(trades_df)

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": total - wins,
            "win_rate": wins / total * 100 if total > 0 else 0,
            "total_pnl": trades_df["pnl"].sum(),
            "avg_pnl": trades_df["pnl"].mean(),
        }


def main():
    rm = RiskManager()

    # Test position sizing
    capital = 100000
    entry = 50000
    stop_distance = 1000  # ATR * 1.5

    size = rm.calculate_position_size(capital, entry, stop_distance)
    print(f"Position size: {size} BTC")
    print(f"Position value: ${size * entry:,.2f}")
    print(f"Risk amount: ${capital * 0.01:,.2f}")


if __name__ == "__main__":
    main()
