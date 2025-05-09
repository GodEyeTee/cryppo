"""
ตัวชี้วัดขั้นสูงสำหรับ CRYPPO (Cryptocurrency Position Optimization)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

# ตั้งค่า logger
logger = logging.getLogger(__name__)

def calculate_stochastic(
    df: pd.DataFrame, 
    k_period: int = 14, 
    d_period: int = 3, 
    slowing: int = 3
) -> pd.DataFrame:
    """
    คำนวณ Stochastic Oscillator
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    k_period (int): ช่วงเวลาสำหรับการคำนวณ %K
    d_period (int): ช่วงเวลาสำหรับการคำนวณ %D
    slowing (int): ช่วงเวลาสำหรับการทำให้ %K ช้าลง
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Stochastic Oscillator เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # ทำให้ %K ช้าลง
    if slowing > 1:
        k = k.rolling(window=slowing).mean()
    
    # คำนวณ %D
    d = k.rolling(window=d_period).mean()
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'stoch_k_{k_period}'] = k
    result[f'stoch_d_{k_period}_{d_period}'] = d
    
    return result

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ On-Balance Volume (OBV)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ OBV เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณการเปลี่ยนแปลงราคา
    price_change = df['close'].diff()
    
    # คำนวณ OBV
    obv = pd.Series(index=df.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    # เพิ่มคอลัมน์ OBV ลงใน DataFrame ผลลัพธ์
    result['obv'] = obv
    
    # คำนวณ OBV EMA
    result['obv_ema'] = obv.ewm(span=20, adjust=False).mean()
    
    return result

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ Volume-Weighted Average Price (VWAP)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ VWAP เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
    if 'timestamp' not in df.columns:
        logger.error("DataFrame ไม่มีคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการคำนวณ VWAP")
        result['vwap'] = np.nan
        return result
    
    # แปลง timestamp เป็น datetime ถ้าจำเป็น
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            logger.error("ไม่สามารถแปลงคอลัมน์ timestamp เป็น datetime ได้")
            result['vwap'] = np.nan
            return result
    
    # เพิ่มคอลัมน์วันที่
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    
    # คำนวณราคาเฉลี่ยถ่วงน้ำหนัก
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # คำนวณ VWAP รายวัน
    df['tp_volume'] = typical_price * df['volume']
    
    # กลุ่มตามวันและคำนวณผลรวมสะสม
    vwap = df.groupby('date').apply(
        lambda x: x['tp_volume'].cumsum() / x['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    
    # เพิ่มคอลัมน์ VWAP ลงใน DataFrame ผลลัพธ์
    result['vwap'] = vwap
    
    return result

def calculate_fibonacci_retracement(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    """
    คำนวณ Fibonacci Retracement
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการหาจุดสูงสุดและต่ำสุด
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Fibonacci Retracement เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ Fibonacci Retracement ในแต่ละช่วง
    for i in range(period, len(df)):
        # ดึงข้อมูลในช่วงที่กำหนด
        window = df.iloc[i-period:i]
        
        # หาจุดสูงสุดและต่ำสุดในช่วง
        high = window['high'].max()
        low = window['low'].min()
        
        # คำนวณระยะห่างระหว่างจุดสูงสุดและต่ำสุด
        distance = high - low
        
        # คำนวณระดับ Fibonacci
        fib_0 = high
        fib_236 = high - 0.236 * distance
        fib_382 = high - 0.382 * distance
        fib_500 = high - 0.500 * distance
        fib_618 = high - 0.618 * distance
        fib_786 = high - 0.786 * distance
        fib_1000 = low
        
        # เก็บค่าในช่วงปัจจุบัน
        result.loc[df.index[i], 'fib_0'] = fib_0
        result.loc[df.index[i], 'fib_236'] = fib_236
        result.loc[df.index[i], 'fib_382'] = fib_382
        result.loc[df.index[i], 'fib_500'] = fib_500
        result.loc[df.index[i], 'fib_618'] = fib_618
        result.loc[df.index[i], 'fib_786'] = fib_786
        result.loc[df.index[i], 'fib_1000'] = fib_1000
    
    return result

def calculate_ichimoku_cloud(
    df: pd.DataFrame, 
    tenkan_period: int = 9, 
    kijun_period: int = 26, 
    senkou_span_b_period: int = 52, 
    displacement: int = 26
) -> pd.DataFrame:
    """
    คำนวณ Ichimoku Cloud
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    tenkan_period (int): ช่วงเวลาสำหรับ Tenkan-sen (Conversion Line)
    kijun_period (int): ช่วงเวลาสำหรับ Kijun-sen (Base Line)
    senkou_span_b_period (int): ช่วงเวลาสำหรับ Senkou Span B (Leading Span B)
    displacement (int): ช่วงเวลาที่ Senkou Span ถูกเลื่อนไปข้างหน้า
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Ichimoku Cloud เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 ในช่วง tenkan_period
    tenkan_sen = (
        df['high'].rolling(window=tenkan_period).max() + 
        df['low'].rolling(window=tenkan_period).min()
    ) / 2
    
    # คำนวณ Kijun-sen (Base Line): (highest high + lowest low) / 2 ในช่วง kijun_period
    kijun_sen = (
        df['high'].rolling(window=kijun_period).max() + 
        df['low'].rolling(window=kijun_period).min()
    ) / 2
    
    # คำนวณ Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 เลื่อนไปข้างหน้า displacement ช่วงเวลา
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # คำนวณ Senkou Span B (Leading Span B): (highest high + lowest low) / 2 ในช่วง senkou_span_b_period เลื่อนไปข้างหน้า displacement ช่วงเวลา
    senkou_span_b = ((
        df['high'].rolling(window=senkou_span_b_period).max() + 
        df['low'].rolling(window=senkou_span_b_period).min()
    ) / 2).shift(displacement)
    
    # คำนวณ Chikou Span (Lagging Span): ราคาปิดเลื่อนกลับไป displacement ช่วงเวลา
    chikou_span = df['close'].shift(-displacement)
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result['ichimoku_tenkan_sen'] = tenkan_sen
    result['ichimoku_kijun_sen'] = kijun_sen
    result['ichimoku_senkou_span_a'] = senkou_span_a
    result['ichimoku_senkou_span_b'] = senkou_span_b
    result['ichimoku_chikou_span'] = chikou_span
    
    return result

def calculate_parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """
    คำนวณ Parabolic SAR
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    af_start (float): ค่าเริ่มต้นของ Acceleration Factor
    af_increment (float): ค่าเพิ่มของ Acceleration Factor
    af_max (float): ค่าสูงสุดของ Acceleration Factor
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Parabolic SAR เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
    if len(df) < 2:
        result['psar'] = np.nan
        return result
    
    # เริ่มต้นคำนวณ Parabolic SAR
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # สร้าง arrays สำหรับเก็บค่า
    psar = np.zeros(len(df))
    bull = np.zeros(len(df), dtype=bool)
    ep = np.zeros(len(df))
    af = np.zeros(len(df))
    
    # กำหนดค่าเริ่มต้น
    bull[0] = True  # เริ่มต้นด้วยแนวโน้มขาขึ้น
    ep[0] = high[0]  # Extreme Point เริ่มต้น
    psar[0] = low[0]  # Parabolic SAR เริ่มต้น
    af[0] = af_start  # Acceleration Factor เริ่มต้น
    
    # คำนวณ Parabolic SAR ในแต่ละช่วงเวลา
    for i in range(1, len(df)):
        # อัพเดต Parabolic SAR
        if bull[i-1]:
            # ในแนวโน้มขาขึ้น
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            # ปรับค่า Parabolic SAR ให้ไม่เกินค่าต่ำสุดของ 2 วันที่ผ่านมา
            if i >= 2:
                psar[i] = min(psar[i], min(low[i-1], low[i-2]))
            
            # ตรวจสอบการเปลี่ยนแนวโน้ม
            if psar[i] > low[i]:
                bull[i] = False
                psar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                bull[i] = True
                # อัพเดต Extreme Point และ Acceleration Factor
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            # ในแนวโน้มขาลง
            psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
            
            # ปรับค่า Parabolic SAR ให้ไม่ต่ำกว่าค่าสูงสุดของ 2 วันที่ผ่านมา
            if i >= 2:
                psar[i] = max(psar[i], max(high[i-1], high[i-2]))
            
            # ตรวจสอบการเปลี่ยนแนวโน้ม
            if psar[i] < high[i]:
                bull[i] = True
                psar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                bull[i] = False
                # อัพเดต Extreme Point และ Acceleration Factor
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result['psar'] = psar
    result['psar_bull'] = bull
    
    return result

def calculate_adx(
    df: pd.DataFrame, 
    period: int = 14, 
    smoothing: int = 14
) -> pd.DataFrame:
    """
    คำนวณ Average Directional Index (ADX)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ ADX
    smoothing (int): ช่วงเวลาสำหรับการทำ smoothing
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ ADX เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณการเปลี่ยนแปลงของราคาสูงสุดและต่ำสุด
    high = df['high']
    low = df['low']
    close = df['close']
    
    # คำนวณ True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # คำนวณ Directional Movement (DM)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # คำนวณ Positive Directional Movement (+DM) และ Negative Directional Movement (-DM)
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # คำนวณ Smoothed Averages
    tr_smooth = pd.Series(tr).rolling(window=period).sum()
    pdm_smooth = pd.Series(pdm).rolling(window=period).sum()
    ndm_smooth = pd.Series(ndm).rolling(window=period).sum()
    
    # คำนวณ Directional Indicators (DI)
    pdi = 100 * pdm_smooth / tr_smooth
    ndi = 100 * ndm_smooth / tr_smooth
    
    # คำนวณ Directional Index (DX)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    
    # คำนวณ Average Directional Index (ADX)
    adx = dx.rolling(window=smoothing).mean()
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'adx_{period}'] = adx
    result[f'pdi_{period}'] = pdi
    result[f'ndi_{period}'] = ndi
    
    return result

def calculate_volume_profile(
    df: pd.DataFrame, 
    period: int = 20, 
    bins: int = 10
) -> pd.DataFrame:
    """
    คำนวณ Volume Profile
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ Volume Profile
    bins (int): จำนวนช่วงราคาที่แบ่ง
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Volume Profile เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ Volume Profile ในแต่ละช่วง
    for i in range(period, len(df)):
        # ดึงข้อมูลในช่วงที่กำหนด
        window = df.iloc[i-period:i]
        
        # แบ่งช่วงราคา
        price_bins = np.linspace(window['low'].min(), window['high'].max(), bins+1)
        
        # คำนวณจุดกึ่งกลางของแต่ละช่วง
        price_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # จัดกลุ่มปริมาณตามช่วงราคา
        volumes = np.zeros(bins)
        
        for j, (_, row) in enumerate(window.iterrows()):
            # หาช่วงราคาที่แท่งเทียนนี้ครอบคลุม
            candle_bins = np.logical_and(
                price_bins[:-1] <= row['high'],
                price_bins[1:] >= row['low']
            )
            
            # กระจายปริมาณตามช่วงราคา
            if candle_bins.sum() > 0:
                vol_per_bin = row['volume'] / candle_bins.sum()
                volumes[candle_bins] += vol_per_bin
        
        # หาช่วงราคาที่มีปริมาณสูงสุด (Point of Control - POC)
        poc_idx = np.argmax(volumes)
        poc_price = price_centers[poc_idx]
        
        # เก็บค่าในช่วงปัจจุบัน
        result.loc[df.index[i], f'volume_profile_poc_{period}'] = poc_price
    
    return result

def calculate_moving_average_crossover(
    df: pd.DataFrame, 
    fast_period: int = 10, 
    slow_period: int = 50
) -> pd.DataFrame:
    """
    คำนวณ Moving Average Crossover
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    fast_period (int): ช่วงเวลาสำหรับ Fast Moving Average
    slow_period (int): ช่วงเวลาสำหรับ Slow Moving Average
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Moving Average Crossover เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ Moving Averages
    fast_ma = df['close'].rolling(window=fast_period).mean()
    slow_ma = df['close'].rolling(window=slow_period).mean()
    
    # คำนวณสัญญาณ Crossover
    # 1 = Golden Cross (fast_ma ตัด slow_ma ขึ้น)
    # -1 = Death Cross (fast_ma ตัด slow_ma ลง)
    # 0 = ไม่มี Crossover
    crossover = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if np.isnan(fast_ma.iloc[i]) or np.isnan(slow_ma.iloc[i]) or np.isnan(fast_ma.iloc[i-1]) or np.isnan(slow_ma.iloc[i-1]):
            continue
            
        if fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
            # Golden Cross
            crossover[i] = 1
        elif fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
            # Death Cross
            crossover[i] = -1
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'ma_crossover_{fast_period}_{slow_period}'] = crossover
    result[f'fast_ma_{fast_period}'] = fast_ma
    result[f'slow_ma_{slow_period}'] = slow_ma
    
    return result

def calculate_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ Pivot Points (แบบ Standard)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Pivot Points เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
    if 'timestamp' not in df.columns:
        logger.error("DataFrame ไม่มีคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการคำนวณ Pivot Points")
        return result
    
    # แปลง timestamp เป็น datetime ถ้าจำเป็น
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            logger.error("ไม่สามารถแปลงคอลัมน์ timestamp เป็น datetime ได้")
            return result
    
    # เพิ่มคอลัมน์วันที่
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    
    # คำนวณค่าสูงสุด ต่ำสุด และราคาปิดรายวัน
    daily_high = df.groupby('date')['high'].max()
    daily_low = df.groupby('date')['low'].min()
    daily_close = df.groupby('date')['close'].last()
    
    # คำนวณ Pivot Points สำหรับวันถัดไป
    # P = (H + L + C) / 3
    pivot = (daily_high + daily_low + daily_close) / 3
    
    # Support และ Resistance
    # R1 = (2 * P) - L
    # R2 = P + (H - L)
    # R3 = H + 2 * (P - L)
    # S1 = (2 * P) - H
    # S2 = P - (H - L)
    # S3 = L - 2 * (H - P)
    r1 = (2 * pivot) - daily_low
    r2 = pivot + (daily_high - daily_low)
    r3 = daily_high + 2 * (pivot - daily_low)
    s1 = (2 * pivot) - daily_high
    s2 = pivot - (daily_high - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot)
    
    # สร้าง DataFrame ของ Pivot Points
    pivot_df = pd.DataFrame({
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    })
    
    # เลื่อนวันไป 1 วัน (Pivot Points ใช้สำหรับวันถัดไป)
    pivot_df.index = pd.to_datetime(pivot_df.index) + pd.Timedelta(days=1)
    
    # รวม Pivot Points เข้ากับข้อมูลเดิม
    for date in df['date'].unique():
        next_date = pd.Timestamp(date) + pd.Timedelta(days=1)
        
        if next_date in pivot_df.index:
            date_mask = df['date'] == pd.Timestamp(next_date).date()
            
            if date_mask.any():
                result.loc[df.index[date_mask], 'pivot'] = pivot_df.loc[next_date, 'pivot']
                result.loc[df.index[date_mask], 'r1'] = pivot_df.loc[next_date, 'r1']
                result.loc[df.index[date_mask], 'r2'] = pivot_df.loc[next_date, 'r2']
                result.loc[df.index[date_mask], 'r3'] = pivot_df.loc[next_date, 'r3']
                result.loc[df.index[date_mask], 's1'] = pivot_df.loc[next_date, 's1']
                result.loc[df.index[date_mask], 's2'] = pivot_df.loc[next_date, 's2']
                result.loc[df.index[date_mask], 's3'] = pivot_df.loc[next_date, 's3']
    
    return result

def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    คำนวณ Chaikin Money Flow
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ CMF
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ CMF เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ Money Flow Multiplier
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)
    
    # คำนวณ Money Flow Volume
    money_flow_volume = money_flow_multiplier * volume
    
    # คำนวณ Chaikin Money Flow
    cmf = money_flow_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'cmf_{period}'] = cmf
    
    return result

def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    คำนวณ Momentum
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ Momentum
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Momentum เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ Momentum
    momentum = df['close'] - df['close'].shift(period)
    
    # คำนวณ Momentum เป็นเปอร์เซ็นต์
    momentum_pct = (df['close'] / df['close'].shift(period) - 1) * 100
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'momentum_{period}'] = momentum
    result[f'momentum_pct_{period}'] = momentum_pct
    
    return result