from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def compute_stats(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"rows": 0}
    info = {
        "rows": len(df),
        "columns": list(df.columns),
        "start": df['timestamp'].min(),
        "end": df['timestamp'].max(),
    }
    if 'close' in df.columns:
        info.update({
            "close_min": float(df['close'].min()),
            "close_max": float(df['close'].max()),
            "close_mean": float(df['close'].mean()),
        })
    return info


def daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    ohlc = df['close'].resample('1D').ohlc()
    vol = df['volume'].resample('1D').sum() if 'volume' in df.columns else None
    out = ohlc
    if vol is not None:
        out['volume'] = vol
    return out.reset_index()


def plot_price(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    if df.empty:
        print("No data to plot")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp'], df['close'], label='Close')
    if 'ema_21' in df.columns:
        plt.plot(df['timestamp'], df['ema_21'], label='EMA 21')
    plt.title('Price')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

