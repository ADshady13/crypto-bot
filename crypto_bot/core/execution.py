"""
Execution Engine — 1x Leverage Order Safety + Position Management.

This module is the ONLY place where trades are executed.
Safety constraints are hard-coded and cannot be overridden:
  - LEVERAGE = 1x (always)
  - STOP LOSS = 3% (always checked)
  - Position sizing = 95% of available capital (5% buffer for fees)

Supports paper mode (simulated fills) and live mode (Binance API).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import ccxt

from core.config import Config

logger = logging.getLogger("crypto_bot")


@dataclass
class Position:
    """Represents an open position."""
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    size_usd: float         # Capital allocated (before fees)
    entry_time: float       # Unix timestamp
    pair: str
    sl_price: float = 0.0   # Stop loss price

    def __post_init__(self):
        if self.direction == "LONG":
            self.sl_price = self.entry_price * (1 - Config.SL_PCT)
        elif self.direction == "SHORT":
            self.sl_price = self.entry_price * (1 + Config.SL_PCT)


@dataclass
class TradeResult:
    """Result of an executed trade."""
    direction: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str        # "SIGNAL", "SL", "MANUAL"
    pair: str
    duration_h: float = 0.0


class ExecutionEngine:
    """
    Manages position entry/exit with hard-coded safety constraints.

    INVARIANTS:
      - Leverage is ALWAYS 1x (hard-coded, not configurable)
      - Stop loss is ALWAYS 3% (hard-coded)
      - Max one position per pair at a time
    """

    def __init__(self, pair: str, mode: str = "paper"):
        self.pair = pair
        self.mode = mode
        self.position: Optional[Position] = None
        self.capital: float = 0.0
        self.trade_history: list[TradeResult] = []

        # Initialize exchange for live mode
        if mode == "live":
            self.exchange = ccxt.binance({
                "apiKey": Config.BINANCE_API_KEY,
                "secret": Config.BINANCE_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            # HARD-CODED: Set leverage to 1x immediately
            try:
                self.exchange.set_leverage(1, self.pair)
                logger.info(f"Leverage set to 1x for {self.pair}")
            except Exception as e:
                logger.warning(f"Could not set leverage (may not be futures): {e}")
        else:
            self.exchange = None
            logger.info(f"Paper mode: simulated execution for {self.pair}")

    def set_capital(self, capital: float):
        """Set available capital for trading."""
        self.capital = capital
        logger.info(f"Capital set: ${capital:,.2f}")

    # ------------------------------------------------------------------ #
    #  SIGNAL EVALUATION
    # ------------------------------------------------------------------ #
    def evaluate_signal(self, bull_prob: float, bear_prob: float) -> str:
        """
        Evaluate ML probabilities and return action.

        Returns: "LONG", "SHORT", or "FLAT"
        """
        if bull_prob > Config.BULL_ENTRY_PROB and bear_prob < Config.BEAR_BLOCK_PROB:
            return "LONG"
        elif bear_prob > Config.BEAR_ENTRY_PROB and bull_prob < Config.BULL_BLOCK_PROB:
            return "SHORT"
        return "FLAT"

    # ------------------------------------------------------------------ #
    #  CHECK STOP LOSS
    # ------------------------------------------------------------------ #
    def check_stop_loss(self, current_price: float) -> Optional[TradeResult]:
        """Check if current price triggers a stop loss."""
        if self.position is None:
            return None

        triggered = False
        if self.position.direction == "LONG" and current_price <= self.position.sl_price:
            triggered = True
        elif self.position.direction == "SHORT" and current_price >= self.position.sl_price:
            triggered = True

        if triggered:
            logger.warning(
                f"🛑 STOP LOSS HIT: {self.position.direction} {self.pair} "
                f"@ {current_price:.2f} (SL={self.position.sl_price:.2f})"
            )
            return self._close_position(self.position.sl_price, "SL")
        return None

    # ------------------------------------------------------------------ #
    #  ENTRY
    # ------------------------------------------------------------------ #
    def open_position(self, direction: str, current_price: float) -> Optional[Position]:
        """
        Open a new position. Applies fees and slippage.

        Returns the Position if opened, None if rejected.
        """
        if self.position is not None:
            logger.info(f"Already in {self.position.direction} position, skipping entry")
            return None

        if self.capital <= 0:
            logger.warning("No capital available for entry")
            return None

        # Position sizing: 95% of capital (5% buffer for fees)
        trade_size = self.capital * 0.95
        entry_cost = trade_size * (Config.FEE_PCT + Config.SLIPPAGE_PCT)
        net_size = trade_size - entry_cost

        if self.mode == "live":
            result = self._execute_live_order(direction, net_size, current_price)
            if result is None:
                return None

        self.position = Position(
            direction=direction,
            entry_price=current_price,
            size_usd=net_size,
            entry_time=time.time(),
            pair=self.pair,
        )
        self.capital -= trade_size

        logger.info(
            f"{'🟢' if direction == 'LONG' else '🔴'} OPENED {direction} {self.pair} "
            f"@ ${current_price:,.2f} | Size: ${net_size:,.2f} | "
            f"SL: ${self.position.sl_price:,.2f}"
        )
        return self.position

    # ------------------------------------------------------------------ #
    #  EXIT
    # ------------------------------------------------------------------ #
    def close_position(self, current_price: float, reason: str = "SIGNAL") -> Optional[TradeResult]:
        """Close the current position at the given price."""
        if self.position is None:
            return None
        return self._close_position(current_price, reason)

    def _close_position(self, exit_price: float, reason: str) -> TradeResult:
        """Internal: close position and record the trade."""
        pos = self.position
        exit_cost = pos.size_usd * (Config.FEE_PCT + Config.SLIPPAGE_PCT)

        if pos.direction == "LONG":
            pnl_pct = (exit_price / pos.entry_price) - 1
        else:  # SHORT
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usd = pos.size_usd * pnl_pct - exit_cost
        duration_h = (time.time() - pos.entry_time) / 3600

        if self.mode == "live":
            self._execute_live_close(pos.direction, pos.size_usd, exit_price)

        self.capital += pos.size_usd + pnl_usd

        result = TradeResult(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct * 100,
            exit_reason=reason,
            pair=self.pair,
            duration_h=duration_h,
        )
        self.trade_history.append(result)
        self.position = None

        emoji = "💰" if pnl_usd > 0 else "💸"
        logger.info(
            f"{emoji} CLOSED {result.direction} {self.pair} @ ${exit_price:,.2f} | "
            f"PnL: ${pnl_usd:+,.2f} ({pnl_pct*100:+.2f}%) | Reason: {reason} | "
            f"Duration: {duration_h:.1f}h"
        )
        return result

    # ------------------------------------------------------------------ #
    #  LIVE EXECUTION (Binance Futures API)
    # ------------------------------------------------------------------ #
    def _execute_live_order(self, direction: str, size_usd: float, price: float) -> Optional[dict]:
        """Execute a real order on Binance Futures."""
        try:
            side = "buy" if direction == "LONG" else "sell"
            # Calculate quantity from USD size
            amount = size_usd / price

            order = self.exchange.create_market_order(
                self.pair, side, amount,
                params={"positionSide": "BOTH"}
            )
            logger.info(f"LIVE ORDER executed: {order['id']} {side} {amount:.6f} {self.pair}")
            return order
        except Exception as e:
            logger.error(f"LIVE ORDER FAILED: {e}")
            return None

    def _execute_live_close(self, direction: str, size_usd: float, price: float):
        """Close a real position on Binance Futures."""
        try:
            side = "sell" if direction == "LONG" else "buy"
            amount = size_usd / price

            order = self.exchange.create_market_order(
                self.pair, side, amount,
                params={"positionSide": "BOTH", "reduceOnly": True}
            )
            logger.info(f"LIVE CLOSE executed: {order['id']}")
        except Exception as e:
            logger.error(f"LIVE CLOSE FAILED: {e}")

    # ------------------------------------------------------------------ #
    #  STATS
    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict:
        """Return summary statistics for the session."""
        if not self.trade_history:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                    "total_pnl": 0.0, "profit_factor": 0.0, "capital": self.capital}

        trades = self.trade_history
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))

        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl": sum(t.pnl_usd for t in trades),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "capital": self.capital,
        }
