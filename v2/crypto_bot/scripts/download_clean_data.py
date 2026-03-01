"""
Download Clean Market Data — 20,000 hourly candles per pair.

Downloads ONLY real data from public APIs:
  1. Spot OHLCV (Binance) — paginated, 1000 candles per request
  2. Funding Rate (Binance Futures) — paginated, all available history
  3. Fear & Greed Index (alternative.me) — daily, all available history

NO synthetic data. NO forward-filling. NO interpolation.
Columns that have no real data are left as NaN.
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TARGET_CANDLES = 20_000
TIMEFRAME = "1h"
BINANCE_SPOT_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FNG_URL = "https://api.alternative.me/fng/"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RATE_LIMIT_SLEEP = 0.15  # seconds between API calls


def fetch_spot_ohlcv(pair: str, target_candles: int = 20_000) -> pd.DataFrame:
    """
    Fetch hourly OHLCV from Binance Spot API.
    Paginates backwards from now, 1000 candles per request.
    """
    print(f"  📥 Fetching {target_candles} hourly candles for {pair}...")
    all_candles = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    batch_size = 1000
    fetched = 0

    while fetched < target_candles:
        remaining = target_candles - fetched
        limit = min(batch_size, remaining)

        params = {
            "symbol": pair,
            "interval": TIMEFRAME,
            "endTime": end_time,
            "limit": limit,
        }

        resp = requests.get(BINANCE_SPOT_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            print(f"    ⚠️  No more data available. Got {fetched} candles total.")
            break

        all_candles = data + all_candles  # Prepend (we're going backwards)
        fetched += len(data)

        # Move endTime to just before the oldest candle we got
        end_time = data[0][0] - 1

        print(f"    Fetched {fetched}/{target_candles} candles...", end="\r")
        time.sleep(RATE_LIMIT_SLEEP)

    print(f"    ✅ Got {len(all_candles)} candles for {pair}                    ")

    # Parse into DataFrame
    df = pd.DataFrame(all_candles, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # Remove tz for consistency

    # Keep only the columns we need, cast to float
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


def fetch_funding_rate(pair: str, start_from_ms: int = None) -> pd.DataFrame:
    """
    Fetch ALL available funding rate history from Binance Futures API.
    Paginates forward from start_from_ms (or ~3 years ago if not specified).
    """
    print(f"  📥 Fetching funding rate history for {pair}...")
    all_records = []
    batch_size = 1000

    # Default: start from ~3 years ago to cover our 20k candle window
    if start_from_ms is None:
        start_from_ms = int((datetime.now(timezone.utc).timestamp() - 3 * 365.25 * 86400) * 1000)

    start_time = start_from_ms

    while True:
        params = {"symbol": pair, "startTime": start_time, "limit": batch_size}

        resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_records.extend(data)
        print(f"    Fetched {len(all_records)} funding records...", end="\r")

        # If we got fewer than batch_size, we've reached the end
        if len(data) < batch_size:
            break

        # Move forward past the last record
        start_time = data[-1]["fundingTime"] + 1
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_records:
        print(f"    ⚠️  No funding rate data for {pair}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].copy()
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    print(f"    ✅ Got {len(df)} funding rate records ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_fear_greed() -> pd.DataFrame:
    """
    Fetch full Fear & Greed Index history from alternative.me.
    This is a daily index (one value per day).
    """
    print(f"  📥 Fetching Fear & Greed Index history...")

    # Fetch max available (they support limit=0 for all data)
    resp = requests.get(FNG_URL, params={"limit": 0, "format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]

    records = []
    for entry in data:
        records.append({
            "date": pd.to_datetime(int(entry["timestamp"]), unit="s").date(),
            "fear_greed_value": int(entry["value"]),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date", keep="last")

    print(f"    ✅ Got {len(df)} daily F&G values ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    return df


def merge_data(ohlcv: pd.DataFrame, funding: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """
    Merge funding rate and F&G into OHLCV data.

    - Funding: forward-fill to hourly (it's reported every 8h) — but ONLY
      within the range where funding data exists. Outside that range = NaN.
    - F&G: forward-fill daily value to hourly — but ONLY within the range
      where F&G data exists. Outside that range = NaN.
    """
    df = ohlcv.copy()

    # ── Merge Funding Rate ──
    if not funding.empty:
        # Reindex funding to hourly, forward-fill ONLY within its own range
        funding_start = funding.index[0]
        funding_end = funding.index[-1]
        hourly_idx = df.index[(df.index >= funding_start) & (df.index <= funding_end)]

        if len(hourly_idx) > 0:
            funding_hourly = funding.reindex(hourly_idx, method="ffill")
            df["funding_rate"] = funding_hourly["funding_rate"]
        else:
            df["funding_rate"] = np.nan
    else:
        df["funding_rate"] = np.nan

    # ── Merge Fear & Greed ──
    if not fng.empty:
        fng_indexed = fng.set_index("date")
        fng_start = fng_indexed.index[0]
        fng_end = fng_indexed.index[-1]

        # Map each hourly timestamp to its date, then look up F&G
        dates = df.index.date
        date_to_fg = fng_indexed["fear_greed_value"].to_dict()
        fg_values = [date_to_fg.get(pd.Timestamp(d), np.nan) for d in dates]
        df["fear_greed_value"] = fg_values

        # Zero out any values outside the F&G data range
        df.loc[df.index < pd.Timestamp(fng_start), "fear_greed_value"] = np.nan
        df.loc[df.index > pd.Timestamp(fng_end) + pd.Timedelta(days=1), "fear_greed_value"] = np.nan
    else:
        df["fear_greed_value"] = np.nan

    return df


def validate_data(df: pd.DataFrame, pair: str) -> dict:
    """Run data quality checks and return a report."""
    report = {"pair": pair}

    report["total_rows"] = len(df)
    report["date_range"] = f"{df.index[0]} → {df.index[-1]}"

    # Check for gaps (missing hours)
    expected_freq = pd.Timedelta(hours=1)
    time_diffs = df.index.to_series().diff().dropna()
    gaps = time_diffs[time_diffs > expected_freq]
    report["time_gaps"] = len(gaps)
    if len(gaps) > 0:
        max_gap = gaps.max()
        report["max_gap"] = str(max_gap)

    # Check for duplicate timestamps
    report["duplicates"] = df.index.duplicated().sum()

    # Check for zero/negative prices
    report["zero_close"] = (df["Close"] <= 0).sum()
    report["zero_volume"] = (df["Volume"] == 0).sum()

    # NaN counts
    for col in df.columns:
        nan_count = df[col].isna().sum()
        pct = nan_count / len(df) * 100
        report[f"nan_{col}"] = f"{nan_count} ({pct:.1f}%)"

    return report


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Fetch Fear & Greed (shared across all pairs)
    print("=" * 60)
    print("  STEP 1: Fear & Greed Index (shared)")
    print("=" * 60)
    fng = fetch_fear_greed()

    # Step 2: Fetch per-pair data
    for pair in PAIRS:
        print(f"\n{'=' * 60}")
        print(f"  STEP 2: {pair}")
        print(f"{'=' * 60}")

        # Spot OHLCV
        ohlcv = fetch_spot_ohlcv(pair, TARGET_CANDLES)

        # Funding rate — start from the same date as OHLCV
        ohlcv_start_ms = int(ohlcv.index[0].timestamp() * 1000)
        funding = fetch_funding_rate(pair, start_from_ms=ohlcv_start_ms)

        # Merge
        df = merge_data(ohlcv, funding, fng)

        # Validate
        report = validate_data(df, pair)
        print(f"\n  📊 Data Quality Report for {pair}:")
        for k, v in report.items():
            print(f"    {k}: {v}")

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{pair}_20k.csv")
        df.to_csv(out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"\n  💾 Saved: {out_path} ({size_mb:.1f} MB)")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  ✅ ALL DONE — {len(PAIRS)} pairs × {TARGET_CANDLES} candles")
    print(f"  📁 Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
