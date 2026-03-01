"""
Data Fetcher — Spot OHLCV + Derivatives + Fear & Greed
Fetches and merges all data sources needed for the composite sentiment score.

Data sources:
  1. Spot OHLCV (1h) — via CCXT Binance (paginated for 10K+ candles)
  2. Funding Rate — Binance futures API (up to 1000 records)
  3. Long/Short Ratio — Binance futures API (up to 500 records)
  4. Open Interest — Binance futures API (up to 500 records)
  5. Fear & Greed Index — alternative.me API (daily, up to 2000+ days)
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "derivatives_backtest/data"


class DerivativesDataFetcher:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.spot_symbol = f"{symbol.replace('USDT', '')}/USDT"

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET")

        config = {"enableRateLimit": True}
        if api_key and api_key != "your_api_key_here":
            config["apiKey"] = api_key
            config["secret"] = api_secret

        self.exchange = ccxt.binance(config)

    def get_cache_path(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        return f"{DATA_DIR}/{self.symbol}_data.csv"

    # ------------------------------------------------------------------ #
    #  1. Spot OHLCV  (paginated — can fetch 10K+ candles)
    # ------------------------------------------------------------------ #
    def fetch_spot_ohlcv(self, timeframe="1h", target_candles=10000):
        """Fetch spot OHLCV with backward pagination to get target_candles."""
        print(f"  Fetching spot OHLCV ({target_candles} target candles)...")

        all_ohlcv = []
        batch_size = 1000  # Binance max per request

        # First fetch — most recent data
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.spot_symbol, timeframe, limit=batch_size
            )
            all_ohlcv = ohlcv
            print(f"    Batch 1: {len(ohlcv)} candles")
        except Exception as e:
            print(f"    Error on initial fetch: {e}")
            return pd.DataFrame()

        # Paginate backward
        batch_num = 2
        while len(all_ohlcv) < target_candles:
            try:
                oldest_ts = all_ohlcv[0][0]
                # Go back by batch_size candles worth of milliseconds
                since = oldest_ts - (batch_size * 3600 * 1000)  # 1h = 3600s
                ohlcv = self.exchange.fetch_ohlcv(
                    self.spot_symbol, timeframe, since=since, limit=batch_size
                )

                if not ohlcv:
                    print(f"    No more data available")
                    break

                # Filter out any overlap
                new_candles = [c for c in ohlcv if c[0] < oldest_ts]
                if not new_candles:
                    print(f"    Reached beginning of available data")
                    break

                all_ohlcv = new_candles + all_ohlcv
                print(
                    f"    Batch {batch_num}: +{len(new_candles)} candles (total: {len(all_ohlcv)})"
                )
                batch_num += 1
                time.sleep(0.2)  # Rate limit

            except Exception as e:
                print(f"    Pagination error: {e}")
                break

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        print(f"    Final: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    # ------------------------------------------------------------------ #
    #  Generic paginated fetcher for Binance futures endpoints
    # ------------------------------------------------------------------ #
    def _paginate_binance(self, url, params, ts_field, max_per_page, start_time=None, end_time=None):
        """
        Generic FORWARD-paginating fetcher for Binance futures data endpoints.
        """
        all_data = []
        if start_time is None:
            start_time = int((time.time() - (30 * 24 * 3600)) * 1000)
        if end_time is None:
            end_time = int(time.time() * 1000)
            
        current_start = start_time
        page = 1

        while current_start < end_time:
            req_params = {**params, "startTime": current_start, "endTime": end_time, "limit": max_per_page}
            try:
                resp = requests.get(url, params=req_params, timeout=15)
                if resp.status_code != 200:
                    print(f"    HTTP {resp.status_code} on page {page}")
                    break
                data = resp.json()
                if not data:
                    break

                all_data.extend(data)
                last_ts = max(int(d[ts_field]) for d in data)
                print(f"    Page {page}: +{len(data)} records (total: {len(all_data)})")

                # Next page starts just after the last record we got
                current_start = last_ts + 1
                
                if len(data) < max_per_page:
                    break  # no more data in range

                page += 1
                if page > 50: # safety break
                    break
                time.sleep(0.3)  # rate limit

            except Exception as e:
                print(f"    Error on page {page}: {e}")
                break

        return all_data

    # ------------------------------------------------------------------ #
    #  2. Funding Rate  (paginated — 8h intervals, max 1000/page)
    # ------------------------------------------------------------------ #
    def fetch_funding(self, start_time_ms=None):
        """Fetch funding rates with forward pagination."""
        print("  Fetching funding rates (paginated)...")
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": self.symbol}
        
        now = int(time.time() * 1000)
        all_data = self._paginate_binance(url, params, "fundingTime", 1000, start_time_ms, now)
        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        print(f"    Final: {len(df)} funding records ({df.index[0]} to {df.index[-1]})")
        return df[["funding_rate"]]

    # ------------------------------------------------------------------ #
    #  3. Long/Short Ratio  (paginated — 1h intervals, max 500/page)
    # ------------------------------------------------------------------ #
    def fetch_lsr(self, start_time_ms=None):
        """Fetch L/S ratio with forward pagination."""
        print("  Fetching L/S ratio (paginated)...")
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {"symbol": self.symbol, "period": "1h"}
        
        # 30 day limit for this endpoint
        limit_30d = int(time.time() * 1000) - (30 * 24 * 3600 * 1000)
        actual_start = max(start_time_ms or 0, limit_30d)
        now = int(time.time() * 1000)

        all_data = self._paginate_binance(url, params, "timestamp", 500, actual_start, now)
        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        df["long_account_pct"] = df["longAccount"].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        print(f"    Final: {len(df)} L/S records ({df.index[0]} to {df.index[-1]})")
        return df[["long_short_ratio", "long_account_pct"]]

    # ------------------------------------------------------------------ #
    #  4. Open Interest  (paginated — 1h intervals, max 500/page)
    # ------------------------------------------------------------------ #
    def fetch_oi(self, start_time_ms=None):
        """Fetch open interest with forward pagination."""
        print("  Fetching open interest (paginated)...")
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {"symbol": self.symbol, "contractType": "PERPETUAL", "period": "1h"}
        
        # 30 day limit for this endpoint
        limit_30d = int(time.time() * 1000) - (30 * 24 * 3600 * 1000)
        actual_start = max(start_time_ms or 0, limit_30d)
        now = int(time.time() * 1000)

        all_data = self._paginate_binance(url, params, "timestamp", 500, actual_start, now)
        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["open_interest"] = df["sumOpenInterestValue"].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        print(f"    Final: {len(df)} OI records ({df.index[0]} to {df.index[-1]})")
        return df[["open_interest"]]

    # ------------------------------------------------------------------ #
    #  5. Fear & Greed Index  (daily — up to 2000+ days)
    # ------------------------------------------------------------------ #
    def fetch_fear_greed(self, limit=2000):
        """Fetch historical Fear & Greed Index from alternative.me."""
        print("  Fetching Fear & Greed Index...")
        try:
            url = "https://api.alternative.me/fng/"
            resp = requests.get(
                url, params={"limit": limit, "format": "json"}, timeout=15
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    records = []
                    for entry in data:
                        records.append(
                            {
                                "timestamp": pd.to_datetime(
                                    int(entry["timestamp"]), unit="s"
                                ),
                                "fear_greed_value": int(entry["value"]),
                                "fear_greed_class": entry["value_classification"],
                            }
                        )
                    df = pd.DataFrame(records)
                    df.set_index("timestamp", inplace=True)
                    df = df.sort_index()
                    print(
                        f"    Got {len(df)} days of F&G data ({df.index[0].date()} to {df.index[-1].date()})"
                    )
                    return df
        except Exception as e:
            print(f"    Error: {e}")
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Merge all data
    # ------------------------------------------------------------------ #
    def _nearest_merge(self, ohlcv_df, source_df, columns):
        """Merge source_df columns onto ohlcv_df via nearest-timestamp lookup."""
        if source_df.empty:
            return ohlcv_df

        result = ohlcv_df.copy()
        source_sorted = source_df.sort_index()

        for col in columns:
            if col not in source_sorted.columns:
                continue
            col_data = source_sorted[col].dropna()
            if col_data.empty:
                continue

            # Use merge_asof for nearest-match join
            result_reset = result.reset_index()
            col_df = col_data.reset_index()
            col_df.columns = ["timestamp", col]

            merged = pd.merge_asof(
                result_reset.sort_values("timestamp"),
                col_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            merged.set_index("timestamp", inplace=True)
            result[col] = merged[col]

        return result

    def _merge_fear_greed(self, ohlcv_df, fg_df):
        """Merge daily Fear & Greed onto hourly OHLCV (same value for all hours in a day)."""
        if fg_df.empty:
            ohlcv_df["fear_greed_value"] = 50  # Default neutral
            return ohlcv_df

        result = ohlcv_df.copy()
        # Create a date column for merge_asof
        result = result.reset_index()
        result["_merge_date"] = result["timestamp"].dt.normalize()

        fg_daily = fg_df.copy()
        fg_daily = fg_daily.reset_index()
        fg_daily["_merge_date"] = fg_daily["timestamp"].dt.normalize()
        fg_daily = fg_daily.drop_duplicates(subset=["_merge_date"], keep="last")
        fg_lookup = fg_daily[["_merge_date", "fear_greed_value"]].copy()

        # Merge on normalized date
        result = pd.merge_asof(
            result.sort_values("_merge_date"),
            fg_lookup.sort_values("_merge_date"),
            on="_merge_date",
            direction="backward",
        )
        result.set_index("timestamp", inplace=True)
        result.drop(columns=["_merge_date"], errors="ignore", inplace=True)
        result["fear_greed_value"] = result["fear_greed_value"].fillna(50)
        return result

    def fetch_all_data(self, use_cache=True, target_candles=15000):
        """Fetch and merge all data sources. L/S Ratio and OI dropped (30-day limit, near-zero ML importance)."""
        cache_path = self.get_cache_path()

        if use_cache and os.path.exists(cache_path):
            print(f"  Loading cached data from {cache_path}")
            df = pd.read_csv(cache_path, index_col="timestamp", parse_dates=True)
            print(f"  Loaded {len(df)} rows ({df.index[0]} to {df.index[-1]})")
            return df

        print(f"\nFetching fresh data for {self.symbol}...")
        print("=" * 50)

        # 1. Spot OHLCV (paginated)
        ohlcv = self.fetch_spot_ohlcv("1h", target_candles)
        if ohlcv.empty:
            print("  ERROR: Could not fetch OHLCV data")
            return pd.DataFrame()

        df = ohlcv.copy()
        start_ts_ms = int(df.index[0].timestamp() * 1000)

        # 2. Funding rate
        funding = self.fetch_funding(start_time_ms=start_ts_ms)
        if not funding.empty:
            df = self._nearest_merge(df, funding, ["funding_rate"])
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0001

        # 3. Fear & Greed
        fg = self.fetch_fear_greed()
        df = self._merge_fear_greed(df, fg)

        # Forward-fill then backward-fill any remaining NaNs
        df = df.ffill().bfill()

        # Clean up any temporary columns from merges
        temp_cols = [c for c in df.columns if c.startswith("_")]
        if temp_cols:
            df.drop(columns=temp_cols, inplace=True)

        # Save cache
        df.to_csv(cache_path)
        print(f"\n  Saved to {cache_path}")
        print(f"  Final: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {list(df.columns)}")
        return df


def fetch_all_pairs(pairs, use_cache=True, target_candles=15000):
    """Fetch data for all pairs."""
    for pair_name, symbol in pairs.items():
        print(f"\n{'='*60}")
        print(f" FETCHING {pair_name} ({symbol})")
        print(f"{'='*60}")
        fetcher = DerivativesDataFetcher(symbol)
        fetcher.fetch_all_data(use_cache=use_cache, target_candles=target_candles)
        time.sleep(1)


if __name__ == "__main__":
    pairs = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "BNB": "BNBUSDT",
        "XRP": "XRPUSDT",
    }
    fetch_all_pairs(pairs, use_cache=False, target_candles=10000)
