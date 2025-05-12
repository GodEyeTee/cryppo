import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    result[f'rsi_{period}'] = rsi
    return result

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    result[f'macd_{fast_period}_{slow_period}'] = macd
    result[f'macd_signal_{signal_period}'] = signal
    result[f'macd_hist'] = histogram
    return result

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    middle_band = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    result[f'bb_upper_{period}'] = upper_band
    result[f'bb_middle_{period}'] = middle_band
    result[f'bb_lower_{period}'] = lower_band
    result[f'bb_bandwidth_{period}'] = (upper_band - lower_band) / middle_band
    result[f'bb_percent_b_{period}'] = (df['close'] - lower_band) / (upper_band - lower_band)
    return result

def calculate_sma(df: pd.DataFrame, periods: List[int] = [10, 50, 200]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    return result

def calculate_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return result

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    result[f'atr_{period}'] = atr
    result[f'atr_percent_{period}'] = (atr / close) * 100
    
    return result
