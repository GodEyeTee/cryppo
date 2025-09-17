from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': hist
    })


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['rsi_14'] = rsi(out['close'], 14)
    out['ema_21'] = ema(out['close'], 21)
    out = out.join(macd(out['close']))
    return out


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Ensure sorted and timestamp dtype
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add simple returns
    out = df.copy()
    out['return'] = np.log(out['close'].clip(lower=1e-12)).diff()

    # Add indicators
    out = add_basic_indicators(out)

    return out

