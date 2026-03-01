"""
Execution Engine — V3 Dynamic Position Management & Trailing Stops.

This module is the ONLY place where trades are executed.
It has been heavily upgraded for V3 to support:
  - Isolated Portfolios (each bot tracks its own 10,000 INR wallet).
  - Trailing Chandelier Exits (tracking Max Favorable Excursion).
  - Consolidated Telegram Notifications.
  - State Recovery (recovering from a JSON file in Paper mode).
"""

import logging
import time
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

import ccxt
import pandas as pd

from core.config import Config

logger = logging.getLogger("crypto_bot")


@dataclass
class Position:
    """Represents an open position with dynamic trailing state."""
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    size_usd: float         # Capital allocated (before fees)
    entry_time: float       # Unix timestamp
    pair: str
    atr_at_entry: float
    sl_price: float         # Current stop loss (starts at Hard SL, moves via trailing)
    mfe_price: float        # Max Favorable Excursion (highest high for long, lowest low for short)
    active_trailing: bool = False
    hold_hours: int = 0


@dataclass
class TradeResult:
    """Result of an executed trade (Logged to CSV)."""
    pair: str
    direction: str
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    fee_usd: float
    exit_reason: str        # "HARD_SL", "TRAIL_SL", "TIMEOUT", "SIGNAL"
    duration_h: float
    mfe_price: float

    def to_csv_dict(self) -> dict:
        d = asdict(self)
        d["entry_time_str"] = pd.to_datetime(d["entry_time"], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        d["exit_time_str"] = pd.to_datetime(d["exit_time"], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        return d


class BotInstance:
    """
    Manages an isolated wallet and position state for a single Pair.
    Contains isolated Kelly Criterion sizing and trailing logic.
    """

    def __init__(self, pair: str, mode: str, initial_capital: float = 10000.0):
        self.pair = pair
        self.mode = mode
        self.capital = initial_capital
        self.position: Optional[Position] = None
        
        # Load specific pair thresholds
        thresh = Config.THRESHOLDS.get(pair, Config.THRESHOLDS["BTCUSDT"])
        self.bull_thresh = thresh["bull"]
        self.bear_thresh = thresh["bear"]
        self.time_thresh = thresh["time"]

    def calculate_position_size(self, probability: float, threshold: float) -> float:
        """Kelly-inspired dynamic allocation directly to the current isolated wallet size."""
        if probability < threshold:
            return 0.0
            
        # Scale from 50% base up to 100% allocation
        score = min(1.0, (probability - threshold) / (0.95 - threshold + 0.001))
        size_pct = 0.50 + (score * 0.50)
        
        return self.capital * size_pct

    def get_state_dict(self) -> dict:
        """Serialize current position for recovery save."""
        return {
            "capital": self.capital,
            "position": asdict(self.position) if self.position else None
        }

    def restore_state(self, state: dict):
        """Restore from saved state."""
        if "capital" in state:
            self.capital = state["capital"]
        if "position" in state and state["position"]:
            self.position = Position(**state["position"])

    def process_tick_updates(self, current_price: float, high: float, low: float) -> Optional[TradeResult]:
        """
        Process trailing stops, timeouts, and stop losses. 
        Returns TradeResult if closed, else None.
        """
        if not self.position:
            return None
            
        pos = self.position
        pos.hold_hours += 1
        
        exit_price = None
        reason = None
        
        # --- Trailing Stop Logic ---
        if pos.direction == "LONG":
            # Update MFE
            if high > pos.mfe_price:
                pos.mfe_price = high
                
            profit_distance = pos.mfe_price - pos.entry_price
            if not pos.active_trailing and profit_distance >= (Config.TRAILING_ACTIVATION * pos.atr_at_entry):
                pos.active_trailing = True
                
            if pos.active_trailing:
                trail_level = pos.mfe_price - (Config.TRAILING_PULLBACK * pos.atr_at_entry)
                if trail_level > pos.sl_price:
                    pos.sl_price = trail_level
                    
            if low <= pos.sl_price:
                exit_price = pos.sl_price * (1 - Config.SLIPPAGE_PCT)
                reason = "TRAIL_SL" if pos.active_trailing else "HARD_SL"
            elif pos.hold_hours >= Config.MAX_HOLD_HOURS:
                exit_price = current_price * (1 - Config.SLIPPAGE_PCT)
                reason = "TIMEOUT"
                
        elif pos.direction == "SHORT":
            if low < pos.mfe_price:
                pos.mfe_price = low
                
            profit_distance = pos.entry_price - pos.mfe_price
            if not pos.active_trailing and profit_distance >= (Config.TRAILING_ACTIVATION * pos.atr_at_entry):
                pos.active_trailing = True
                
            if pos.active_trailing:
                trail_level = pos.mfe_price + (Config.TRAILING_PULLBACK * pos.atr_at_entry)
                if trail_level < pos.sl_price:
                    pos.sl_price = trail_level
                    
            if high >= pos.sl_price:
                exit_price = pos.sl_price * (1 + Config.SLIPPAGE_PCT)
                reason = "TRAIL_SL" if pos.active_trailing else "HARD_SL"
            elif pos.hold_hours >= Config.MAX_HOLD_HOURS:
                exit_price = current_price * (1 + Config.SLIPPAGE_PCT)
                reason = "TIMEOUT"
                
        # --- Has Triggered Exit ---
        if exit_price is not None:
            return self.close_position(exit_price, reason)
            
        return None

    def evaluate_signal(self, current_price: float, atr: float, bull_prob: float, bear_prob: float, time_prob: float) -> Optional[str]:
        """
        Evaluate ML probabilities and return action strings or execute them immediately.
        Returns the action string ("LONG", "SHORT", "CLOSE", "FLAT") for logging.
        """
        # 1. Check Chop Filter
        if time_prob >= self.time_thresh:
            if self.position:
                return "HOLD" # We don't close randomly on chop, we just hold and wait
            return "FLAT"

        is_long = bull_prob >= self.bull_thresh
        is_short = bear_prob >= self.bear_thresh
        
        # Conflict Resolution
        if is_long and is_short:
            if bull_prob >= bear_prob:
                is_short = False
            else:
                is_long = False
                
        # 2. Reversals
        if self.position:
            if (self.position.direction == "LONG" and is_short) or \
               (self.position.direction == "SHORT" and is_long):
                return "REVERSAL"
            else:
                return "HOLD"
                
        # 3. New Entry
        if is_long or is_short:
            direction = "LONG" if is_long else "SHORT"
            prob_score = bull_prob if is_long else bear_prob
            thresh_score = self.bull_thresh if is_long else self.bear_thresh
            
            size_usd = self.calculate_position_size(prob_score, thresh_score)
            
            if size_usd > 100: # Min trade size
                self._open_position(direction, current_price, size_usd, atr)
                return direction
                
        return "FLAT"


    def _open_position(self, direction: str, current_price: float, size_usd: float, atr: float):
        """Simulates internal state update for position open."""
        # Simulated Slippage
        entry_price = current_price * (1 + Config.SLIPPAGE_PCT if direction == "LONG" else 1 - Config.SLIPPAGE_PCT)
        
        sl_price = entry_price - (Config.SL_ATR_MULT * atr) if direction == "LONG" else entry_price + (Config.SL_ATR_MULT * atr)
        
        self.position = Position(
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
            entry_time=time.time(),
            pair=self.pair,
            atr_at_entry=atr,
            sl_price=sl_price,
            mfe_price=entry_price
        )
        logger.info(f"[{self.pair}] OPENED {direction} @ ${entry_price:,.2f} | Size: ${size_usd:,.2f}")

    def close_position(self, exit_price: float, reason: str) -> TradeResult:
        """Closes internal position and applies PnL physically to the isolated wallet."""
        pos = self.position
        
        # Apply fees via simulated math directly
        total_fee = pos.size_usd * (Config.FEE_PCT * 2) # Assume entry + exit fee
        
        if pos.direction == "LONG":
            raw_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            raw_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            
        raw_pnl_usd = pos.size_usd * raw_pnl_pct
        net_pnl = raw_pnl_usd - total_fee
        
        # Add PnL to isolated wallet
        self.capital += net_pnl
        
        duration_h = pos.hold_hours
        
        res = TradeResult(
            pair=self.pair,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            pnl_usd=net_pnl,
            pnl_pct=(net_pnl / pos.size_usd) * 100,
            fee_usd=total_fee,
            exit_reason=reason,
            duration_h=duration_h,
            mfe_price=pos.mfe_price
        )
        
        self.position = None
        logger.info(f"[{self.pair}] CLOSED {pos.direction} @ ${exit_price:,.2f} | PnL: ${net_pnl:+,.2f} ({reason})")
        return res


class PortfolioManager:
    """
    Overarching manager that holds the 5 isolated BotInstances.
    Is responsible for state recovery, trade logging, and generating the Telegram summary.
    """

    def __init__(self, pairs: list[str], mode: str = "paper"):
        self.mode = mode
        self.pairs = pairs
        self.bots: Dict[str, BotInstance] = {p: BotInstance(p, mode) for p in pairs}
        
        self.state_file = os.path.join(Config.LOG_DIR, f"active_state_{mode}.json")
        self.trade_log_file = os.path.join(Config.LOG_DIR, "trades.csv")
        
        self._recover_state()
        
    def _recover_state(self):
        """Recover Paper/Live position states on restart."""
        if self.mode == "paper":
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    try:
                        states = json.load(f)
                        for pair, state in states.items():
                            if pair in self.bots:
                                self.bots[pair].restore_state(state)
                        logger.info("Successfully recovered Paper state from JSON.")
                    except Exception as e:
                        logger.error(f"Failed to parse state file: {e}")
        else:
            # LIVE MODE: In a true live deployment, we would connect CCXT here to recreate state.
            pass

    def save_state(self):
        """Save current Paper simulation state to JSON."""
        if self.mode == "paper":
            states = {p: b.get_state_dict() for p, b in self.bots.items()}
            with open(self.state_file, "w") as f:
                json.dump(states, f, indent=4)

    def log_trade(self, result: TradeResult):
        """Append closed trade to the permanent CSV ledger."""
        df = pd.DataFrame([result.to_csv_dict()])
        
        write_header = not os.path.exists(self.trade_log_file)
        df.to_csv(self.trade_log_file, mode="a", index=False, header=write_header)

    def generate_telegram_summary(self) -> str:
        """Create the V3 single compiled Telegram alert message."""
        total_pnl = 0.0
        portfolio_val = 0.0
        
        msg_lines = []
        
        for p in self.pairs:
            bot = self.bots[p]
            cap_diff = bot.capital - 10000.0
            total_pnl += cap_diff
            portfolio_val += bot.capital
            
            pnl_pct = (cap_diff / 10000.0) * 100
            
            # Format the output logic
            if bot.position:
                direction = bot.position.direction
                emoji = "🟢" if direction == "LONG" else "🔴"
                
                # Calculate active PnL approximation loosely
                # (For real time string formatting only, not actual settlement)
                msg_lines.append(f"{emoji} {p}: HOLDING {direction} (Entry: ${bot.position.entry_price:,.2f})")
                msg_lines.append(f"   ↳ Trail SL: ${bot.position.sl_price:,.2f}")
                msg_lines.append(f"   ↳ Wallet: ₹{bot.capital:,.0f} [Net PnL: {pnl_pct:+.2f}%]")
            else:
                msg_lines.append(f"⚪ {p}: FLAT")
                msg_lines.append(f"   ↳ Wallet: ₹{bot.capital:,.0f} [Net PnL: {pnl_pct:+.2f}%]")
                
            msg_lines.append("") # Margin
            
        tax_impact = total_pnl * 0.70 if total_pnl > 0 else total_pnl
        
        header = f"📊 CryptoBot V2 - Hourly Update\n"
        header += f"💰 Global Portfolio: ₹{portfolio_val:,.0f} (+₹{total_pnl:+,.0f} Gross)\n"
        header += f"💸 Est Net Profit (after 30% tax): ₹{tax_impact:+,.0f}\n\n"
        
        return header + "\n".join(msg_lines)
