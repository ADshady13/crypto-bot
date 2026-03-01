"""
Feature Engineering — Shared between training and live inference.

This is the SINGLE SOURCE OF TRUTH for feature computation.
Both retrain.py and inference.py import from here to ensure
training and live features are computed identically.

V2 Features (expanded):
  Original "Shield" set:
    - ROC (multi-timeframe: 4h, 12h, 24h, 72h)
    - Fear & Greed (normalized, z-score)
    - Funding Rate (z-score, deltas)
    - 200-EMA Distance
    - ADX (trend strength)
    - ATR (volatility, % of price, z-score)
    - Volume (z-score, delta, ratio)
    - Composite Sentiment Score

  NEW in V2:
    - Bollinger Bands (width, %B)
    - RSI (multi-timeframe: 14, 7, 21 + delta)
    - Candlestick Patterns (doji, hammer, engulfing, etc.)
"""

import pandas as pd
import numpy as np
import logging

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

logger = logging.getLogger("crypto_bot")

# The exact feature set used for ML inference — order matters for model input
SHIELD_FEATURES = [
    # --- Momentum (5) ---
    "roc", "roc_4h", "roc_24h", "roc_72h", "roc_zscore",

    # --- Sentiment (2) — removed fg_normalized (r=0.997 with raw) ---
    "fear_greed_value", "fg_zscore",

    # --- Derivatives (4) ---
    "funding_rate", "funding_zscore", "funding_delta_4h", "funding_delta_24h",

    # --- Trend (2) ---
    "ema_200_dist", "adx",

    # --- Volatility (2) — removed raw atr (price-dependent range) ---
    "atr_pct", "atr_pct_zscore",

    # --- Volume (2) — removed volume_ratio_24h (r=0.942 with zscore) ---
    "volume_zscore", "volume_delta_4h",

    # --- Composite (1) — removed sentiment_raw (r=0.950 with normalized) ---
    "sentiment_score",

    # === V2 FEATURES ===

    # --- Bollinger Bands — 20h (3) ---
    "bb_width_20",    # Band width: (upper - lower) / mid — 20h volatility squeeze
    "bb_pct_20",      # %B: position within 20h bands (0=lower, 1=upper)
    "bb_zscore_20",   # Z-score of bb_width_20 — squeeze/expansion extremes

    # --- Bollinger Bands — 50h (3) ---
    "bb_width_50",    # Band width over 50h — medium-term volatility
    "bb_pct_50",      # %B: position within 50h bands
    "bb_zscore_50",   # Z-score of bb_width_50

    # --- RSI (4) — removed rsi_21 (r=0.978 with rsi_14) ---
    "rsi_14",         # Standard RSI(14h)
    "rsi_7",          # Fast RSI(7h) — more responsive
    "rsi_delta",      # RSI(7) - RSI(21) — momentum divergence (still computed internally)
    "rsi_zscore",     # Z-score of RSI(14) — RSI extremes vs recent history

    # --- Candlestick Patterns (6) ---
    "cdl_doji",       # Doji: indecision candle (tiny body, long wicks)
    "cdl_hammer",     # Hammer: bullish reversal (long lower wick, small body at top)
    "cdl_invhammer",  # Inverted Hammer: bearish reversal (long upper wick)
    "cdl_engulf_bull", # Bullish Engulfing: bull candle fully engulfs prior bear candle
    "cdl_engulf_bear", # Bearish Engulfing: bear candle fully engulfs prior bull candle
    "cdl_body_ratio",  # Body / Total Range — candle conviction (1 = full body, 0 = doji)

    # === V3 MACRO FEATURES (5) ===
    "btc_roc",           # BTC Momentum
    "btc_rsi_14",        # BTC RSI
    "btc_volume_zscore", # BTC Volume Spikes
    "btc_atr_pct",       # BTC Volatility
    "btc_ema_200_dist",  # BTC Trend
]


class FeatureEngineer:
    """Compute all technical features for the Dual-Core strategy."""

    def __init__(self, params=None):
        self.params = params or {
            "zscore_period": 30,
            "funding_percentile_period": 90,
            "atr_period": 14,
            "ema_period": 50,
            "roc_period": 12,
            "fg_zscore_period": 30,
            "volume_zscore_period": 30,
            "bb_period": 20,
            "bb_std": 2,
        }

    # ================================================================== #
    #  Core statistical functions
    # ================================================================== #
    @staticmethod
    def zscore(series: pd.Series, period: int = 30) -> pd.Series:
        """Rolling z-score."""
        mu = series.rolling(window=period, min_periods=1).mean()
        sigma = series.rolling(window=period, min_periods=1).std().replace(0, 1)
        return (series - mu) / sigma

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def roc(series: pd.Series, period: int) -> pd.Series:
        """Rate of Change (% change over N periods)."""
        return series.pct_change(periods=period) * 100

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        hl = high - low
        hc = np.abs(high - close.shift())
        lc = np.abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index — trend strength."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_val = FeatureEngineer.atr(high, low, close, period).replace(0, 1)
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
        return dx.ewm(span=period, adjust=False).mean()

    # ================================================================== #
    #  Main feature pipeline
    # ================================================================== #
    def transform(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Compute ALL features from raw OHLCV + funding + fear_greed data.
        This is the single entry point for both training and live inference.

        Input columns required: Open, High, Low, Close, Volume, funding_rate, fear_greed_value
        Output: DataFrame with all Shield features added.
        """
        if verbose:
            logger.info("Engineering features...")

        # --- Funding ---
        if "funding_rate" in df.columns:
            df["funding_zscore"] = self.zscore(df["funding_rate"], self.params["zscore_period"])
            df["funding_delta_4h"] = df["funding_rate"].diff(4)
            df["funding_delta_24h"] = df["funding_rate"].diff(24)

        # --- Fear & Greed ---
        if "fear_greed_value" in df.columns:
            df["fg_normalized"] = (df["fear_greed_value"] - 50) / 50
            fg_zp = self.params.get("fg_zscore_period", 30) * 24
            df["fg_zscore"] = self.zscore(df["fear_greed_value"].astype(float), fg_zp)

        # --- Volatility ---
        df["atr"] = self.atr(df["High"], df["Low"], df["Close"], self.params["atr_period"])
        df["atr_pct"] = df["atr"] / df["Close"]
        df["atr_pct_zscore"] = self.zscore(df["atr_pct"], 30)

        # --- Trend ---
        ema_200 = self.ema(df["Close"], 200)
        df["ema_200"] = ema_200
        df["ema_200_dist"] = ((df["Close"] - ema_200) / ema_200) * 100
        df["adx"] = self.adx(df["High"], df["Low"], df["Close"], 14)

        # --- Momentum (multi-timeframe ROC) ---
        df["roc"] = self.roc(df["Close"], self.params.get("roc_period", 12))
        df["roc_4h"] = self.roc(df["Close"], 4)
        df["roc_24h"] = self.roc(df["Close"], 24)
        df["roc_72h"] = self.roc(df["Close"], 72)
        df["roc_zscore"] = self.zscore(df["roc"], self.params["zscore_period"])

        # --- Volume ---
        if "Volume" in df.columns:
            df["volume_zscore"] = self.zscore(df["Volume"], self.params["volume_zscore_period"])
            df["volume_delta_4h"] = df["Volume"].pct_change(4) * 100
            df["volume_ratio_24h"] = df["Volume"] / df["Volume"].rolling(24, min_periods=1).mean()

        # --- Composite Sentiment Score ---
        df = self._compute_sentiment(df)

        # === NEW V2 FEATURES ===

        # --- Bollinger Bands (using ta library) ---
        df = self._compute_bollinger_bands(df)

        # --- RSI (multi-timeframe) ---
        df = self._compute_rsi(df)

        # --- Candlestick Patterns ---
        df = self._compute_candle_patterns(df)

        # Fill NaN
        df = df.fillna(0)

        if verbose:
            avail = [f for f in SHIELD_FEATURES if f in df.columns]
            logger.info(f"Features ready: {len(avail)}/{len(SHIELD_FEATURES)}")

        return df

    # ================================================================== #
    #  V1 Composite features
    # ================================================================== #
    def _compute_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite sentiment score (0-1 normalized)."""
        funding_c = (-1 * df.get("funding_zscore", 0)) * 0.15
        fg_c = (-1 * df.get("fg_zscore", 0)) * 0.10
        roc_c = df.get("roc_zscore", 0) * 0.25

        # Volume during breakout
        if "volume_zscore" in df.columns:
            close = df["Close"]
            high_20 = close.rolling(20).max().shift(1)
            breakout_up = close > high_20
            vol_c = df["volume_zscore"].where(breakout_up, 0).fillna(0) * 0.20
        else:
            vol_c = 0

        raw = funding_c + fg_c + roc_c + vol_c
        if isinstance(raw, (int, float)):
            raw = pd.Series(0, index=df.index)

        r_min = raw.rolling(500, min_periods=50).min()
        r_max = raw.rolling(500, min_periods=50).max()
        r_range = (r_max - r_min).replace(0, 1)

        df["sentiment_score"] = ((raw - r_min) / r_range).clip(0, 1).fillna(0.5)
        df["sentiment_raw"] = raw
        return df

    # ================================================================== #
    #  V2 NEW: Bollinger Bands
    # ================================================================== #
    def _compute_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe Bollinger Band width and %B using ta library."""
        bb_std = self.params.get("bb_std", 2)

        for period in [20, 50]:
            bb = BollingerBands(
                close=df["Close"],
                window=period,
                window_dev=bb_std,
                fillna=False,
            )

            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            mid = bb.bollinger_mavg()

            # BB Width: (upper - lower) / mid — volatility squeeze/expansion
            band_range = upper - lower
            df[f"bb_width_{period}"] = band_range / mid.replace(0, 1)

            # %B: position within bands (0=lower, 1=upper, >1=above, <0=below)
            df[f"bb_pct_{period}"] = (df["Close"] - lower) / band_range.replace(0, 1)

            # Z-score of BB width — squeeze/expansion extremes
            df[f"bb_zscore_{period}"] = self.zscore(df[f"bb_width_{period}"], 30)

        return df

    # ================================================================== #
    #  V2 NEW: RSI (multi-timeframe)
    # ================================================================== #
    def _compute_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe RSI + delta + z-score."""
        close = df["Close"]

        # RSI at 3 timeframes
        df["rsi_14"] = RSIIndicator(close=close, window=14, fillna=False).rsi()
        df["rsi_7"] = RSIIndicator(close=close, window=7, fillna=False).rsi()
        df["rsi_21"] = RSIIndicator(close=close, window=21, fillna=False).rsi()

        # RSI Delta: fast vs slow — momentum divergence
        # Positive = momentum accelerating, Negative = momentum decelerating
        df["rsi_delta"] = df["rsi_7"] - df["rsi_21"]

        # RSI Z-score: detects when RSI(14) is extreme relative to recent history
        df["rsi_zscore"] = self.zscore(df["rsi_14"], 30)

        return df

    # ================================================================== #
    #  V2 NEW: Candlestick Patterns
    # ================================================================== #
    def _compute_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common candlestick patterns from OHLC data.
        Values are 0 (no pattern) or 1 (pattern detected).
        Pure-math implementation, no external TA-Lib dependency.
        """
        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        c = df["Close"]

        body = (c - o).abs()
        total_range = (h - l).replace(0, 1e-10)  # Avoid div by zero
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

        # Body ratio: body / total range (1 = full body marubozu, 0 = doji)
        df["cdl_body_ratio"] = body / total_range

        # --- Doji ---
        # Body < 10% of total range
        df["cdl_doji"] = (body / total_range < 0.10).astype(int)

        # --- Hammer (bullish reversal) ---
        # Small body at top, long lower wick (≥ 2× body), small upper wick
        is_hammer = (
            (lower_wick >= 2 * body) &
            (upper_wick < body) &
            (body / total_range > 0.10) &
            (body / total_range < 0.40)
        )
        df["cdl_hammer"] = is_hammer.astype(int)

        # --- Inverted Hammer (bearish signal at top, bullish at bottom) ---
        # Small body at bottom, long upper wick (≥ 2× body), small lower wick
        is_invhammer = (
            (upper_wick >= 2 * body) &
            (lower_wick < body) &
            (body / total_range > 0.10) &
            (body / total_range < 0.40)
        )
        df["cdl_invhammer"] = is_invhammer.astype(int)

        # --- Bullish Engulfing ---
        # Current candle is bullish (c > o), prior was bearish (c_prev < o_prev),
        # and current body engulfs prior body
        prev_o = o.shift(1)
        prev_c = c.shift(1)
        is_bullish = c > o
        prev_bearish = prev_c < prev_o

        df["cdl_engulf_bull"] = (
            is_bullish &
            prev_bearish &
            (o <= prev_c) &    # Current open at or below prior close
            (c >= prev_o)       # Current close at or above prior open
        ).astype(int)

        # --- Bearish Engulfing ---
        # Current candle is bearish (c < o), prior was bullish (c_prev > o_prev),
        # and current body engulfs prior body
        is_bearish = c < o
        prev_bullish = prev_c > prev_o

        df["cdl_engulf_bear"] = (
            is_bearish &
            prev_bullish &
            (o >= prev_c) &    # Current open at or above prior close
            (c <= prev_o)       # Current close at or below prior open
        ).astype(int)

        return df

    # ================================================================== #
    #  V3 NEW: Macro BTC Features
    # ================================================================== #
    def add_macro_features(self, df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge macro BTC features into the current DataFrame safely.
        btc_df must already have its own features computed.
        """
        features_to_extract = ["roc", "rsi_14", "volume_zscore", "atr_pct", "ema_200_dist"]
        
        # We only want to join the overlapping timestamps safely
        macro_subset = btc_df[features_to_extract].copy()
        
        # Rename columns to add 'btc_' prefix
        macro_subset.rename(columns={f: f"btc_{f}" for f in features_to_extract}, inplace=True)
        
        # Left join on the timestamp index - zero time leakage because timestamps align perfectly
        df = df.join(macro_subset, how="left")
        
        # If any missing (due to mismatches or start differences), forward-fill then fillna(0)
        for f in features_to_extract:
            df[f"btc_{f}"] = df[f"btc_{f}"].ffill().fillna(0)
            
        return df

    # ================================================================== #
    #  Target creation (for training only) — Triple-Barrier Method
    # ================================================================== #
    def create_targets(self, df: pd.DataFrame, horizon: int = 6,
                       tp_atr_mult: float = 1.5, sl_atr_mult: float = 1.0,
                       fee_pct: float = 0.001) -> pd.DataFrame:
        """
        Create triple-barrier target variables using ATR-based dynamic barriers.

        Three targets are produced:
          - target      (y_bull): 1 if LONG TP hit before SL within horizon
          - target_short (y_bear): 1 if SHORT TP hit before SL within horizon
          - target_time  (y_time): 1 if NO barrier hit within horizon (chop market)

        Args:
            df:           DataFrame with OHLCV + 'atr' column (must be computed first)
            horizon:      Max look-forward in hourly candles (default: 6h)
            tp_atr_mult:  ATR multiplier for take-profit barrier (default: 1.5)
            sl_atr_mult:  ATR multiplier for stop-loss barrier (default: 1.0)
            fee_pct:      Total round-trip fee+slippage as decimal (default: 0.001 = 0.10%)
                          Binance Futures L0: maker 0.02% + taker 0.04%
                          + slippage buffer → rounded to 0.10% total

        Barrier math:
          Bull TP = P₀ + (tp_atr_mult × ATR) + (P₀ × fee_pct)
          Bull SL = P₀ - (sl_atr_mult × ATR)
          Bear TP = P₀ - (tp_atr_mult × ATR) - (P₀ × fee_pct)
          Bear SL = P₀ + (sl_atr_mult × ATR)
        """
        # Ensure ATR is computed
        if "atr" not in df.columns:
            df["atr"] = self.atr(df["High"], df["Low"], df["Close"], self.params["atr_period"])

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        atr_vals = df["atr"].values
        n = len(df)

        y_bull = np.zeros(n, dtype=int)
        y_bear = np.zeros(n, dtype=int)
        y_time = np.zeros(n, dtype=int)

        for i in range(n):
            p0 = close[i]
            a = atr_vals[i]

            if np.isnan(a) or a == 0 or np.isnan(p0):
                # Can't compute barriers — leave as 0
                continue

            # Don't label rows too close to the end (not enough future data)
            if i + horizon >= n:
                continue

            # ── Bull barriers ──
            bull_tp = p0 + (tp_atr_mult * a) + (p0 * fee_pct)
            bull_sl = p0 - (sl_atr_mult * a)

            # ── Bear barriers ──
            bear_tp = p0 - (tp_atr_mult * a) - (p0 * fee_pct)
            bear_sl = p0 + (sl_atr_mult * a)

            bull_hit = False
            bear_hit = False
            any_barrier_hit = False

            # Walk forward candle-by-candle
            for j in range(i + 1, min(i + horizon + 1, n)):
                h_j = high[j]
                l_j = low[j]

                # ── Bull first-touch ──
                if not bull_hit:
                    # Check SL first (conservative: if both hit in same candle, SL wins)
                    if l_j <= bull_sl:
                        bull_hit = True  # SL hit → y_bull stays 0
                        any_barrier_hit = True
                    elif h_j >= bull_tp:
                        bull_hit = True
                        y_bull[i] = 1  # TP hit first!
                        any_barrier_hit = True

                # ── Bear first-touch ──
                if not bear_hit:
                    # Check SL first (conservative: if both hit in same candle, SL wins)
                    if h_j >= bear_sl:
                        bear_hit = True  # SL hit → y_bear stays 0
                        any_barrier_hit = True
                    elif l_j <= bear_tp:
                        bear_hit = True
                        y_bear[i] = 1  # TP hit first!
                        any_barrier_hit = True

                # Both barriers resolved — stop early
                if bull_hit and bear_hit:
                    break

            # ── Time/Chop target ──
            # y_time = 1 if NEITHER bull NOR bear barrier was hit (total chop)
            if not any_barrier_hit:
                y_time[i] = 1

        df["target"] = y_bull
        df["target_short"] = y_bear
        df["target_time"] = y_time

        # Also store the raw future return for analysis
        future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
        df["target_return"] = future_ret * 100

        logger.info(
            f"Triple-barrier targets created (horizon={horizon}h, "
            f"TP={tp_atr_mult}×ATR, SL={sl_atr_mult}×ATR, fees={fee_pct*100:.2f}%): "
            f"y_bull={y_bull.sum()} ({y_bull.mean()*100:.1f}%) | "
            f"y_bear={y_bear.sum()} ({y_bear.mean()*100:.1f}%) | "
            f"y_time={y_time.sum()} ({y_time.mean()*100:.1f}%)"
        )

        return df

    # ================================================================== #
    #  Extract feature matrix for ML
    # ================================================================== #
    @staticmethod
    def get_feature_matrix(df: pd.DataFrame) -> tuple:
        """
        Extract the feature matrix X and available feature names.
        Only includes features that exist in the DataFrame.
        """
        available = [f for f in SHIELD_FEATURES if f in df.columns]
        X = df[available].fillna(0).values
        return X, available
