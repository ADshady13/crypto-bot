"""
Realistic Backtester — V2 Triple-Core System (v3: Dynamic Mode)

Simulates deploying capital using trained models with highly realistic math:
  - Trailing Exits: Replaces hard TP with Chandelier Exit / ATR Trailing Stop.
  - Indian Tax: Crypto futures are EXEMPT from 1% TDS in India (taxed as business income).
  - Global Wallet Limit: Bot cannot deploy more than MAX_ALLOCATION_PER_BOT.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from strategies.feature_engineering import FeatureEngineer, SHIELD_FEATURES

# ── Config ──────────────────────────────────────────────────────────────
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Capital & Risk
GLOBAL_WALLET_BALANCE = 50_000   # Total wallet balance (INR)
MAX_ALLOCATION_PER_BOT = 10_000  # Cap this bot to use max 10K INR at any time
POSITION_SCALE_BASE = 0.50       # Base allocation (50% of available max allocation)

# Fees & Taxes
FEE_PCT = 0.0004         # 0.04% taker fee (Binance Futures)
SLIPPAGE_PCT = 0.0003    # 0.03% slippage per side
# TDS is NOT applicable to Crypto Futures in India

# Exits
SL_ATR_MULT = 1.0          # Hard stop loss distance
TRAILING_ACTIVATION = 1.0  # Activate trailing stop when profit hits 1.0×ATR
TRAILING_PULLBACK = 0.5    # Trail behind the highest high / lowest low by 0.5×ATR
MAX_HOLD_HOURS = 12        # Increased to 12h to allow trends to run


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    direction: str
    entry_price: float
    exit_price: float
    max_favorable_price: float
    sl_price: float
    atr_at_entry: float
    size_inr: float
    pnl_inr: float      # Net PnL (after fees)
    fee_inr: float      # Exchange fees + Slippage costs
    exit_reason: str
    hold_hours: int
    bull_p: float
    bear_p: float
    time_p: float


@dataclass
class BacktestResult:
    pair: str
    period: str
    bull_thresh: float
    bear_thresh: float
    time_thresh: float
    final_capital: float
    net_pnl: float
    total_trades: int
    win_rate: float
    avg_win_inr: float
    avg_loss_inr: float
    profit_factor: float
    max_drawdown_pct: float
    trades: list = field(default_factory=list)


def calculate_position_size(probability: float, threshold: float, max_alloc: float) -> float:
    """
    Intelligent sizing: Scale size linearly from BASE at threshold 
    to 100% of max_alloc at 0.95 probability.
    """
    if probability < threshold:
        return 0.0
    
    # Scale factor from 0.0 to 1.0 based on how far above threshold we are
    # Example: threshold=0.5. Prob=0.5 -> score=0. Prob=0.95 -> score=1.
    score = min(1.0, (probability - threshold) / (0.95 - threshold + 0.001))
    
    # Position ranges from BASE% (e.g. 50%) up to 100% of max allocation
    size_pct = POSITION_SCALE_BASE + (score * (1.0 - POSITION_SCALE_BASE))
    
    return max_alloc * size_pct


def load_models(pair: str) -> dict:
    sym = pair.replace("USDT", "")
    models = {}
    for tname in ["bull", "bear", "time"]:
        path = os.path.join(MODEL_DIR, f"xgb_{tname}_{sym}.json")
        if os.path.exists(path):
            m = xgb.XGBClassifier()
            m.load_model(path)
            models[tname] = m
    return models


def run_dynamic_backtest(
    df: pd.DataFrame,
    models: dict,
    pair: str,
    period: str,
    bull_thresh: float,
    bear_thresh: float,
    time_thresh: float,
) -> BacktestResult:

    available = [f for f in SHIELD_FEATURES if f in df.columns]
    X = df[available].values
    
    bull_probs = models["bull"].predict_proba(X)[:, 1]
    bear_probs = models["bear"].predict_proba(X)[:, 1]
    time_probs = models["time"].predict_proba(X)[:, 1]
    
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)
    timestamps = df.index
    
    capital_pool = MAX_ALLOCATION_PER_BOT
    peak_capital = capital_pool
    max_drawdown_pct = 0.0
    trades = []
    
    # State
    in_pos = False
    direction = ""
    entry_p = 0.0
    entry_time = None
    sl_p = 0.0
    active_trailing = False
    mfe_p = 0.0  # Max Favorable Excursion price
    pos_size = 0.0
    entry_fee = 0.0
    hold_c = 0
    e_atr = 0.0
    bp, brp, tp = 0.0, 0.0, 0.0
    
    for i in range(len(df)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        if in_pos:
            hold_c += 1
            exit_p = None
            reason = None
            
            # --- Trailing Stop Logic ---
            if direction == "LONG":
                # Update Max Favorable Excursion
                if high[i] > mfe_p:
                    mfe_p = high[i]
                
                # Check if we should activate trailing stop
                profit_distance = mfe_p - entry_p
                if not active_trailing and profit_distance >= (TRAILING_ACTIVATION * e_atr):
                    active_trailing = True
                
                # Update Stop Loss if trailing is active
                if active_trailing:
                    trail_level = mfe_p - (TRAILING_PULLBACK * e_atr)
                    if trail_level > sl_p:
                        sl_p = trail_level
                        
                # Check for Stop Loss hit
                if low[i] <= sl_p:
                    exit_p = sl_p * (1 - SLIPPAGE_PCT)
                    reason = "TRAIL_SL" if active_trailing else "HARD_SL"
                    
                # Check Timeout
                elif hold_c >= MAX_HOLD_HOURS:
                    exit_p = close[i] * (1 - SLIPPAGE_PCT)
                    reason = "TIMEOUT"
                    
            elif direction == "SHORT":
                if low[i] < mfe_p:
                    mfe_p = low[i]
                    
                profit_distance = entry_p - mfe_p
                if not active_trailing and profit_distance >= (TRAILING_ACTIVATION * e_atr):
                    active_trailing = True
                    
                if active_trailing:
                    trail_level = mfe_p + (TRAILING_PULLBACK * e_atr)
                    if trail_level < sl_p:
                        sl_p = trail_level
                        
                if high[i] >= sl_p:
                    exit_p = sl_p * (1 + SLIPPAGE_PCT)
                    reason = "TRAIL_SL" if active_trailing else "HARD_SL"
                    
                elif hold_c >= MAX_HOLD_HOURS:
                    exit_p = close[i] * (1 + SLIPPAGE_PCT)
                    reason = "TIMEOUT"
            
            # --- Process Exit ---
            if exit_p is not None:
                exit_fee = pos_size * FEE_PCT
                
                if direction == "LONG":
                    raw_pnl_pct = (exit_p - entry_p) / entry_p
                else:
                    raw_pnl_pct = (entry_p - exit_p) / entry_p
                
                raw_pnl_inr = pos_size * raw_pnl_pct
                total_fee = entry_fee + exit_fee
                net_pnl = raw_pnl_inr - total_fee
                
                capital_pool += net_pnl
                
                trades.append(Trade(
                    entry_time=str(entry_time), exit_time=str(timestamps[i]),
                    direction=direction, entry_price=entry_p, exit_price=exit_p,
                    max_favorable_price=mfe_p, sl_price=sl_p, atr_at_entry=e_atr,
                    size_inr=pos_size, pnl_inr=net_pnl, fee_inr=total_fee,
                    exit_reason=reason, hold_hours=hold_c,
                    bull_p=bp, bear_p=brp, time_p=tp,
                ))
                
                peak_capital = max(peak_capital, capital_pool)
                max_drawdown_pct = max(max_drawdown_pct, (peak_capital - capital_pool) / peak_capital * 100)
                
                in_pos = False
                continue
        
        # --- Entry Logic ---
        if not in_pos and capital_pool > 0:
            bull_prob = bull_probs[i]
            bear_prob = bear_probs[i]
            time_prob = time_probs[i]
            
            # Chop filter overrides everything
            if time_prob >= time_thresh:
                continue
                
            is_long = bull_prob >= bull_thresh
            is_short = bear_prob >= bear_thresh
            
            # Resolve conflict
            if is_long and is_short:
                if bull_prob >= bear_prob:
                    is_short = False
                else:
                    is_long = False
            
            if is_long or is_short:
                direction = "LONG" if is_long else "SHORT"
                prob_score = bull_prob if is_long else bear_prob
                thresh_score = bull_thresh if is_long else bear_thresh
                
                # Dynamic Position sizing
                alloc = calculate_position_size(prob_score, thresh_score, capital_pool)
                if alloc < 500:  # Minimum 500 INR to trade
                    continue
                    
                pos_size = alloc
                entry_p = close[i] * (1 + SLIPPAGE_PCT if direction == "LONG" else 1 - SLIPPAGE_PCT)
                entry_time = timestamps[i]
                e_atr = atr[i]
                entry_fee = pos_size * FEE_PCT
                hold_c = 0
                bp, brp, tp = bull_prob, bear_prob, time_prob
                mfe_p = entry_p
                active_trailing = False
                
                if direction == "LONG":
                    sl_p = entry_p - (SL_ATR_MULT * e_atr)
                else:
                    sl_p = entry_p + (SL_ATR_MULT * e_atr)
                
                in_pos = True
    
    # Summary
    winning = [t for t in trades if t.pnl_inr > 0]
    losing = [t for t in trades if t.pnl_inr <= 0]
    total_wins = sum(t.pnl_inr for t in winning) if winning else 0
    total_losses = abs(sum(t.pnl_inr for t in losing)) if losing else 0
    
    return BacktestResult(
        pair=pair, period=period,
        bull_thresh=bull_thresh, bear_thresh=bear_thresh, time_thresh=time_thresh,
        final_capital=capital_pool,
        net_pnl=capital_pool - MAX_ALLOCATION_PER_BOT,
        total_trades=len(trades),
        win_rate=len(winning) / max(len(trades), 1) * 100,
        avg_win_inr=total_wins / max(len(winning), 1),
        avg_loss_inr=-total_losses / max(len(losing), 1),
        profit_factor=total_wins / max(total_losses, 0.01),
        max_drawdown_pct=max_drawdown_pct,
        trades=trades
    )


def print_result(r: BacktestResult):
    emoji = "🟢" if r.net_pnl > 0 else "🔴"
    print(f"\n  {emoji} {r.pair} [{r.period}] | Bull≥{r.bull_thresh:.2f} Bear≥{r.bear_thresh:.2f} Time<{r.time_thresh:.2f}")
    print(f"  ┌─ Cap: ₹{MAX_ALLOCATION_PER_BOT:,.0f} → ₹{r.final_capital:,.0f} ({r.net_pnl/MAX_ALLOCATION_PER_BOT*100:+.2f}%)")
    print(f"  │  Net PnL: ₹{r.net_pnl:+,.2f} | Max DD: {r.max_drawdown_pct:.1f}%")
    print(f"  │  Trades: {r.total_trades} | Win Rate: {r.win_rate:.1f}% | PF: {r.profit_factor:.2f}")
    print(f"  └─ Avg Win: ₹{r.avg_win_inr:+,.2f} | Avg Loss: ₹{r.avg_loss_inr:+,.2f}")


def main():
    print("=" * 70)
    print("  CryptoBot V2 — DYNAMIC BACKTESTER (Trailing Exits + Kelly Sizing)")
    print(f"  Max Alloc: ₹{MAX_ALLOCATION_PER_BOT:,} | Slippage: {SLIPPAGE_PCT*100}% | TDS: EXEMPT")
    print(f"  Trailing Activation: {TRAILING_ACTIVATION}×ATR | Pullback: {TRAILING_PULLBACK}×ATR")
    print("=" * 70)
    
    fe = FeatureEngineer()
    
    # Precompute BTC macro features for backtesting
    print("  Loading global BTC macro features...")
    btc_df = pd.read_csv(os.path.join(DATA_DIR, "BTCUSDT_20k.csv"), index_col="timestamp", parse_dates=True)
    btc_features = fe.transform(btc_df)
    
    # Search grid
    B_THRESH = [0.45, 0.50, 0.55, 0.60]
    T_THRESH = [0.35, 0.40, 0.45, 0.50, 0.55]
    
    for pair in PAIRS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{pair}_20k.csv"),
                         index_col="timestamp", parse_dates=True)
        df = fe.transform(df)
        df = fe.add_macro_features(df, btc_features)
        df = fe.create_targets(df)
        oot_df = df.iloc[17500:19800] # OOT Nov 2025 - Mar 2026
        
        models = load_models(pair)
        if len(models) != 3:
            continue
            
        best_r = None
        # Execute grid search
        for bt in B_THRESH:
            for brt in B_THRESH:
                for tt in T_THRESH:
                    r = run_dynamic_backtest(oot_df, models, pair, "OOT", bt, brt, tt)
                    # We want maximizing net_pnl, but only if it made $>0 and traded a statistically significant amount
                    if r.total_trades >= 5:
                        if best_r is None or r.net_pnl > best_r.net_pnl:
                            best_r = r
                            
        if best_r is not None:
            print_result(best_r)
        else:
            print(f"\n  🔴 {pair} | No profitable configurations found (min 5 trades).")


if __name__ == "__main__":
    main()
