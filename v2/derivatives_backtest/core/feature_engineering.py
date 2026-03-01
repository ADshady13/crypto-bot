"""
Feature Engineering — Compute all features and composite sentiment score.

Composite Score Components (designed to predict next-candle direction):
  1. Funding Rate Z-score (inverted)     — 25%  (negative funding = shorts paying = contrarian long)
  2. Long/Short Ratio Z-score (inverted) — 20%  (low ratio = more shorts = contrarian long)
  3. Open Interest change Z-score        — 15%  (rising OI during flat = accumulation)
  4. Volume Z-score                      — 10%  (high volume confirms breakout direction)
  5. Fear & Greed Z-score (inverted)     — 20%  (extreme fear = contrarian buy)
  6. Price momentum (ROC)                — 10%  (recent price direction confirmation)
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, params=None):
        self.params = params or {
            "zscore_period": 30,
            "funding_percentile_period": 90,
            "atr_period": 14,
            "ema_period": 50,
            "volume_zscore_period": 30,
            "roc_period": 12,  # Rate of change (12h lookback)
            "fg_zscore_period": 30,  # Fear & Greed z-score window (in days mapped to hours)
        }

    # ------------------------------------------------------------------ #
    #  Core statistical functions
    # ------------------------------------------------------------------ #
    def calculate_zscore(self, series, period=30):
        """Rolling z-score."""
        rolling_mean = series.rolling(window=period, min_periods=1).mean()
        rolling_std = series.rolling(window=period, min_periods=1).std()
        return (series - rolling_mean) / rolling_std.replace(0, 1)

    def calculate_percentile(self, series, period=90):
        """Rolling percentile rank (0-100)."""

        def percentile_rank(x):
            if len(x) < 2:
                return 50
            return (pd.Series(x) < x[-1]).sum() / len(x) * 100

        return series.rolling(window=period, min_periods=1).apply(
            percentile_rank, raw=True
        )

    def calculate_atr(self, high, low, close, period=14):
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    def calculate_ema(self, series, period=50):
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_roc(self, series, period=12):
        """Rate of Change — % price change over N periods."""
        return series.pct_change(periods=period) * 100

    # ------------------------------------------------------------------ #
    #  Regime detection
    # ------------------------------------------------------------------ #
    def is_price_flat(self, df, threshold=0.02):
        """Detect low-volatility / ranging regime."""
        atr = self.calculate_atr(df["High"], df["Low"], df["Close"])
        return (atr / df["Close"]) < threshold

    def is_breakout(self, df, lookback=20):
        """Detect price breakout above/below recent range."""
        close = df["Close"]
        high_lookback = close.rolling(lookback).max()
        low_lookback = close.rolling(lookback).min()
        breakout_up = close > high_lookback.shift(1)
        breakout_down = close < low_lookback.shift(1)
        return breakout_up, breakout_down

    # ------------------------------------------------------------------ #
    #  Main feature pipeline
    # ------------------------------------------------------------------ #
    def engineer_features(self, df):
        """Compute all features from raw merged data."""
        print("Engineering features...")

        # === Funding features ===
        if "funding_rate" in df.columns:
            print("  Computing funding features...")
            df["funding_zscore"] = self.calculate_zscore(
                df["funding_rate"], self.params["zscore_period"]
            )
            df["funding_percentile"] = self.calculate_percentile(
                df["funding_rate"], self.params["funding_percentile_period"]
            )

        # === Open Interest features ===
        if "open_interest" in df.columns:
            print("  Computing OI features...")
            df["oi_pct_change"] = df["open_interest"].pct_change()
            df["oi_zscore"] = self.calculate_zscore(
                df["open_interest"], self.params["zscore_period"]
            )

        # === Positioning features ===
        if "long_short_ratio" in df.columns:
            print("  Computing positioning features...")
            df["long_short_ratio_zscore"] = self.calculate_zscore(
                df["long_short_ratio"], self.params["zscore_period"]
            )

        # === Volume features ===
        if "Volume" in df.columns:
            print("  Computing volume features...")
            df["volume_zscore"] = self.calculate_zscore(
                df["Volume"], self.params["volume_zscore_period"]
            )

        # === Fear & Greed features ===
        if "fear_greed_value" in df.columns:
            print("  Computing Fear & Greed features...")
            # F&G is 0-100.  Normalize to -1 to 1 range.
            df["fg_normalized"] = (df["fear_greed_value"] - 50) / 50
            # Z-score of F&G over rolling window (daily data mapped to hourly)
            fg_zscore_period = self.params.get("fg_zscore_period", 30) * 24  # days→hours
            df["fg_zscore"] = self.calculate_zscore(
                df["fear_greed_value"].astype(float), fg_zscore_period
            )

        # === Volatility & trend features ===
        print("  Computing volatility & trend features...")
        df["atr"] = self.calculate_atr(
            df["High"], df["Low"], df["Close"], self.params["atr_period"]
        )
        df["atr_pct"] = df["atr"] / df["Close"]

        df["ema_50"] = self.calculate_ema(df["Close"], self.params["ema_period"])
        df["price_above_ema"] = df["Close"] > df["ema_50"]

        # === Price momentum ===
        print("  Computing momentum features...")
        df["roc"] = self.calculate_roc(df["Close"], self.params.get("roc_period", 12))
        df["roc_zscore"] = self.calculate_zscore(df["roc"], self.params["zscore_period"])

        # === Regime detection ===
        print("  Computing regime features...")
        df["price_flat"] = self.is_price_flat(df)
        breakout_up, breakout_down = self.is_breakout(df)
        df["breakout_up"] = breakout_up
        df["breakout_down"] = breakout_down

        # Fill NaN
        df = df.fillna(0)

        print(f"  Total features: {len(df.columns)}")
        return df

    # ------------------------------------------------------------------ #
    #  ML-Specific features (Task 1)
    # ------------------------------------------------------------------ #
    def calculate_adx(self, high, low, close, period=14):
        """Average Directional Index — trend strength indicator."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = self.calculate_atr(high, low, close, period)
        atr_safe = atr.replace(0, 1)

        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_safe)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_safe)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)) * 100
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    def engineer_ml_features(self, df):
        """
        Generate ML-specific features on top of the base features.
        Call AFTER engineer_features() and compute_sentiment_score().
        """
        print("Engineering ML features...")

        # --- Derivatives Momentum: 4h and 24h deltas ---
        if "funding_rate" in df.columns:
            df["funding_delta_4h"] = df["funding_rate"].diff(4)
            df["funding_delta_24h"] = df["funding_rate"].diff(24)
            print("  Added funding deltas (4h, 24h)")

        if "open_interest" in df.columns:
            df["oi_delta_4h"] = df["open_interest"].pct_change(4) * 100
            df["oi_delta_24h"] = df["open_interest"].pct_change(24) * 100
            print("  Added OI deltas (4h, 24h)")

        # --- Trend Context ---
        # 200-EMA distance (% above/below)
        ema_200 = self.calculate_ema(df["Close"], 200)
        df["ema_200"] = ema_200
        df["ema_200_dist"] = ((df["Close"] - ema_200) / ema_200) * 100
        print("  Added 200-EMA distance")

        # ADX (trend strength)
        df["adx"] = self.calculate_adx(df["High"], df["Low"], df["Close"], 14)
        print(f"  Added ADX (mean: {df['adx'].mean():.1f})")

        # --- Volume momentum ---
        if "Volume" in df.columns:
            df["volume_delta_4h"] = df["Volume"].pct_change(4) * 100
            df["volume_ratio_24h"] = df["Volume"] / df["Volume"].rolling(24, min_periods=1).mean()

        # --- ROC at multiple timeframes ---
        df["roc_4h"] = self.calculate_roc(df["Close"], 4)
        df["roc_24h"] = self.calculate_roc(df["Close"], 24)
        df["roc_72h"] = self.calculate_roc(df["Close"], 72)

        # --- Volatility regime ---
        df["atr_pct_zscore"] = self.calculate_zscore(df["atr_pct"], 30) if "atr_pct" in df.columns else 0

        # Fill NaN
        df = df.fillna(0)

        ml_features = [
            "funding_rate", "funding_zscore", "funding_delta_4h", "funding_delta_24h",
            "open_interest", "oi_pct_change", "oi_zscore", "oi_delta_4h", "oi_delta_24h",
            "long_short_ratio", "long_short_ratio_zscore",
            "fear_greed_value", "fg_normalized", "fg_zscore",
            "Volume", "volume_zscore", "volume_delta_4h", "volume_ratio_24h",
            "atr", "atr_pct", "atr_pct_zscore",
            "ema_200_dist", "adx",
            "roc", "roc_4h", "roc_24h", "roc_72h", "roc_zscore",
            "sentiment_score", "sentiment_raw",
        ]
        # Only keep features that actually exist
        available = [f for f in ml_features if f in df.columns]
        print(f"  ML features available: {len(available)}")
        return df, available

    def create_ml_target(self, df, horizon=24, threshold_pct=1.0):
        """
        Create binary target: 1 if price rises by >= threshold_pct% in the next `horizon` candles.
        """
        future_return = df["Close"].shift(-horizon) / df["Close"] - 1
        df["target_return"] = future_return * 100
        df["target"] = (future_return >= threshold_pct / 100).astype(int)

        # Drop the last `horizon` rows (no target available)
        valid_mask = df["target_return"].notna()
        n_positive = df.loc[valid_mask, "target"].sum()
        n_total = valid_mask.sum()
        print(f"  Target (LONG): {n_positive}/{n_total} positive ({n_positive/n_total*100:.1f}%) "
              f"| horizon={horizon}h, threshold=+{threshold_pct}%")
        return df

    def create_ml_target_short(self, df, horizon=24, threshold_pct=2.0):
        """
        Create binary SHORT target: 1 if price DROPS by >= threshold_pct% in next `horizon` candles.
        Distinct from the long target — a flat day is 0 for both.
        """
        future_return = df["Close"].shift(-horizon) / df["Close"] - 1
        df["target_return"] = future_return * 100
        df["target_short"] = (future_return <= -threshold_pct / 100).astype(int)

        valid_mask = df["target_return"].notna()
        n_positive = df.loc[valid_mask, "target_short"].sum()
        n_total = valid_mask.sum()
        print(f"  Target (SHORT): {n_positive}/{n_total} positive ({n_positive/n_total*100:.1f}%) "
              f"| horizon={horizon}h, threshold=-{threshold_pct}%")
        return df

    # ------------------------------------------------------------------ #
    #  Composite Sentiment Score
    # ------------------------------------------------------------------ #
    def compute_sentiment_score(self, df):
        """
        Build composite score designed to predict bullish next candle.
        """
        print("Computing composite sentiment score...")

        # 1. Funding (inverted) - 15%
        funding_component = (-1 * df.get("funding_zscore", 0)) * 0.15

        # 2. L/S ratio (inverted) - 15%
        lsr_component = (-1 * df.get("long_short_ratio_zscore", 0)) * 0.15

        # 3. Fear & Greed (inverted) - 10%
        fg_component = (-1 * df.get("fg_zscore", 0)) * 0.10

        # 4. OI during flat regime (accumulation) - 15%
        if "oi_zscore" in df.columns and "price_flat" in df.columns:
            oi_flat = df["oi_zscore"].where(df["price_flat"], 0)
            oi_component = oi_flat.fillna(0) * 0.15
        else:
            oi_component = 0

        # 5. Volume during breakout (momentum) - 20%
        if "volume_zscore" in df.columns and "breakout_up" in df.columns:
            vol_breakout = df["volume_zscore"].where(df["breakout_up"], 0)
            vol_component = vol_breakout.fillna(0) * 0.20
        else:
            vol_component = 0

        # 6. Price momentum (ROC) - 25% (Increased for trend capture)
        roc_component = df.get("roc_zscore", 0) * 0.25

        # Combine
        raw_score = (
            funding_component
            + lsr_component
            + fg_component
            + oi_component
            + vol_component
            + roc_component
        )

        # Ensure it's a Series
        if isinstance(raw_score, (int, float)):
            raw_score = pd.Series(0, index=df.index)

        # Normalize to [0, 1] using rolling min-max (500-candle window for stability)
        norm_window = 500
        rolling_min = raw_score.rolling(norm_window, min_periods=50).min()
        rolling_max = raw_score.rolling(norm_window, min_periods=50).max()
        score_range = rolling_max - rolling_min

        df["sentiment_score"] = (raw_score - rolling_min) / score_range.replace(0, 1)
        df["sentiment_score"] = df["sentiment_score"].clip(0, 1).fillna(0.5)

        # Also store the raw score for analysis
        df["sentiment_raw"] = raw_score

        print(
            f"  Score range: {df['sentiment_score'].min():.3f} - {df['sentiment_score'].max():.3f}"
        )
        print(
            f"  Mean: {df['sentiment_score'].mean():.3f}, Std: {df['sentiment_score'].std():.3f}"
        )

        return df


def main():
    """Test feature engineering on cached data."""
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_fetcher import DerivativesDataFetcher

    fetcher = DerivativesDataFetcher("BTCUSDT")
    df = fetcher.fetch_all_data(target_candles=1000)

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.compute_sentiment_score(df)

    print("\nFeature columns:")
    feature_cols = [
        c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]
    ]
    print(feature_cols)

    print("\nSample (last 5 rows):")
    display_cols = [
        "Close",
        "funding_zscore",
        "long_short_ratio_zscore",
        "fg_zscore",
        "roc_zscore",
        "sentiment_score",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].tail())


if __name__ == "__main__":
    main()
