from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import requests


TIMEFRAME_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
}

MAX_CANDLES_PER_REQUEST = 1000
BASE_URL = "https://api.binance.com/api/v3/klines"


def get_interval_ms(tf: str) -> int:
    tf = tf.strip()
    n = int("".join(ch for ch in tf if ch.isdigit()))
    u = "".join(ch for ch in tf if ch.isalpha())
    if u == "m":
        return n * 60_000
    if u == "h":
        return n * 3_600_000
    if u == "d":
        return n * 86_400_000
    if u == "w":
        return n * 7 * 86_400_000
    if u == "M":
        return n * 30 * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def _fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_time_ms: Optional[int],
    end_time_ms: Optional[int],
    limit: int = MAX_CANDLES_PER_REQUEST,
) -> List[List]:
    params = {
        "symbol": symbol.upper(),
        "interval": TIMEFRAME_MAP[interval],
        "limit": min(limit, MAX_CANDLES_PER_REQUEST),
    }
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    resp = session.get(BASE_URL, params=params, timeout=30)
    if resp.status_code == 429:
        wait = int(resp.headers.get("Retry-After", "5"))
        time.sleep(wait)
        resp = session.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def klines_to_df(candles: List[List]) -> pd.DataFrame:
    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "_ignore"
    ]
    df = pd.DataFrame(candles, columns=cols)
    num_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c])
    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df.drop(columns=["_ignore"])  # type: ignore[arg-type]


def download_klines(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    include_unclosed: bool = False,
) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    interval_ms = get_interval_ms(timeframe)

    session = requests.Session()
    all_rows: List[List] = []
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + interval_ms * MAX_CANDLES_PER_REQUEST, end_ms)
        part = _fetch_klines(session, symbol, timeframe, cursor, chunk_end, MAX_CANDLES_PER_REQUEST)
        if not part:
            cursor = chunk_end
            continue
        all_rows.extend(part)
        if len(part) < MAX_CANDLES_PER_REQUEST:
            # likely exhausted
            break
        # next open time = last close open time + interval
        cursor = part[-1][0] + interval_ms

        # be gentle
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = klines_to_df(all_rows)

    if not include_unclosed and len(df) > 0:
        last_open_ms = int(df["timestamp"].iloc[-1].value // 1_000_000)
        last_close_ms = last_open_ms + interval_ms - 1
        if last_close_ms >= end_ms:
            df = df.iloc[:-1]

    # sort & drop duplicates by timestamp
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

