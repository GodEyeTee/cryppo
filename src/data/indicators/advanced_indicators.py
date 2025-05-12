import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    if slowing > 1:
        k = k.rolling(window=slowing).mean()
    d = k.rolling(window=d_period).mean()
    result[f'stoch_k_{k_period}'] = k
    result[f'stoch_d_{k_period}_{d_period}'] = d
    return result

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    price_change = df['close'].diff()
    obv = pd.Series(index=df.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    result['obv'] = obv
    result['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
    return result

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    
    if 'timestamp' not in df.columns:
        logger.error("DataFrame ไม่มีคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการคำนวณ VWAP")
        result['vwap'] = np.nan
        return result
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            logger.error("ไม่สามารถแปลงคอลัมน์ timestamp เป็น datetime ได้")
            result['vwap'] = np.nan
            return result
    
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = typical_price * df['volume']
    
    vwap = df.groupby('date').apply(
        lambda x: x['tp_volume'].cumsum() / x['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    
    result['vwap'] = vwap
    return result

def calculate_fibonacci_retracement(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    
    for i in range(period, len(df)):
        window = df.iloc[i-period:i]
        high = window['high'].max()
        low = window['low'].min()
        distance = high - low
        
        fib_levels = {
            'fib_0': high,
            'fib_236': high - 0.236 * distance,
            'fib_382': high - 0.382 * distance,
            'fib_500': high - 0.500 * distance,
            'fib_618': high - 0.618 * distance,
            'fib_786': high - 0.786 * distance,
            'fib_1000': low
        }
        
        for level, value in fib_levels.items():
            result.loc[df.index[i], level] = value
    
    return result

def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, 
                         senkou_span_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    
    tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + 
                 df['low'].rolling(window=tenkan_period).min()) / 2
    
    kijun_sen = (df['high'].rolling(window=kijun_period).max() + 
                df['low'].rolling(window=kijun_period).min()) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    senkou_span_b = ((df['high'].rolling(window=senkou_span_b_period).max() + 
                     df['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(displacement)
    
    chikou_span = df['close'].shift(-displacement)
    
    result['ichimoku_tenkan_sen'] = tenkan_sen
    result['ichimoku_kijun_sen'] = kijun_sen
    result['ichimoku_senkou_span_a'] = senkou_span_a
    result['ichimoku_senkou_span_b'] = senkou_span_b
    result['ichimoku_chikou_span'] = chikou_span
    
    return result

def calculate_adx(df: pd.DataFrame, period: int = 14, smoothing: int = 14) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = pd.Series(tr).rolling(window=period).sum()
    pdm_smooth = pd.Series(pdm).rolling(window=period).sum()
    ndm_smooth = pd.Series(ndm).rolling(window=period).sum()
    
    pdi = 100 * pdm_smooth / tr_smooth
    ndi = 100 * ndm_smooth / tr_smooth
    
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    adx = dx.rolling(window=smoothing).mean()
    
    result[f'adx_{period}'] = adx
    result[f'pdi_{period}'] = pdi
    result[f'ndi_{period}'] = ndi
    
    return result

def calculate_parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, 
                            af_max: float = 0.2) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    
    if len(df) < 2:
        result['psar'] = np.nan
        return result
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    psar = np.zeros(len(df))
    bull = np.zeros(len(df), dtype=bool)
    ep = np.zeros(len(df))
    af = np.zeros(len(df))
    
    bull[0] = True
    ep[0] = high[0]
    psar[0] = low[0]
    af[0] = af_start
    
    for i in range(1, len(df)):
        if bull[i-1]:
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            if i >= 2:
                psar[i] = min(psar[i], min(low[i-1], low[i-2]))
            
            if psar[i] > low[i]:
                bull[i] = False
                psar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                bull[i] = True
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
            
            if i >= 2:
                psar[i] = max(psar[i], max(high[i-1], high[i-2]))
            
            if psar[i] < high[i]:
                bull[i] = True
                psar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                bull[i] = False
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    result['psar'] = psar
    result['psar_bull'] = bull
    
    return result

def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    momentum = df['close'] - df['close'].shift(period)
    momentum_pct = (df['close'] / df['close'].shift(period) - 1) * 100
    result[f'momentum_{period}'] = momentum
    result[f'momentum_pct_{period}'] = momentum_pct
    return result

def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)
    money_flow_volume = money_flow_multiplier * volume
    cmf = money_flow_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    result[f'cmf_{period}'] = cmf
    return result
