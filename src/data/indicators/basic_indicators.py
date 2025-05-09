"""
ตัวชี้วัดพื้นฐานสำหรับ CRYPPO (Cryptocurrency Position Optimization)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# ตั้งค่า logger
logger = logging.getLogger(__name__)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    คำนวณ Relative Strength Index (RSI)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ RSI
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ RSI เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณการเปลี่ยนแปลงราคา
    delta = df['close'].diff()
    
    # แยกการเปลี่ยนแปลงเป็นบวกและลบ
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # คำนวณค่าเฉลี่ยของ gain และ loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # คำนวณ Relative Strength
    rs = avg_gain / avg_loss
    
    # คำนวณ RSI
    rsi = 100 - (100 / (1 + rs))
    
    # เพิ่มคอลัมน์ RSI ลงใน DataFrame ผลลัพธ์
    result[f'rsi_{period}'] = rsi
    
    return result

def calculate_macd(
    df: pd.DataFrame, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> pd.DataFrame:
    """
    คำนวณ Moving Average Convergence Divergence (MACD)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    fast_period (int): ช่วงเวลาสำหรับ EMA เร็ว
    slow_period (int): ช่วงเวลาสำหรับ EMA ช้า
    signal_period (int): ช่วงเวลาสำหรับเส้นสัญญาณ
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ MACD เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ EMA เร็วและช้า
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # คำนวณ MACD
    macd = ema_fast - ema_slow
    
    # คำนวณเส้นสัญญาณ
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # คำนวณฮิสโตแกรม
    histogram = macd - signal
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'macd_{fast_period}_{slow_period}'] = macd
    result[f'macd_signal_{signal_period}'] = signal
    result[f'macd_hist'] = histogram
    
    return result

def calculate_bollinger_bands(
    df: pd.DataFrame, 
    period: int = 20, 
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    คำนวณ Bollinger Bands
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ
    std_dev (float): จำนวนเท่าของส่วนเบี่ยงเบนมาตรฐาน
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ Bollinger Bands เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณเส้นกลาง (SMA)
    middle_band = df['close'].rolling(window=period).mean()
    
    # คำนวณส่วนเบี่ยงเบนมาตรฐาน
    rolling_std = df['close'].rolling(window=period).std()
    
    # คำนวณแถบบนและล่าง
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
    result[f'bb_upper_{period}'] = upper_band
    result[f'bb_middle_{period}'] = middle_band
    result[f'bb_lower_{period}'] = lower_band
    
    # เพิ่มคอลัมน์ Bollinger Bandwidth และ %B
    result[f'bb_bandwidth_{period}'] = (upper_band - lower_band) / middle_band
    result[f'bb_percent_b_{period}'] = (df['close'] - lower_band) / (upper_band - lower_band)
    
    return result

def calculate_sma(df: pd.DataFrame, periods: List[int] = [10, 50, 200]) -> pd.DataFrame:
    """
    คำนวณ Simple Moving Average (SMA)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    periods (list): รายการช่วงเวลาสำหรับการคำนวณ SMA
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ SMA เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ SMA สำหรับแต่ละช่วงเวลา
    for period in periods:
        result[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    return result

def calculate_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> pd.DataFrame:
    """
    คำนวณ Exponential Moving Average (EMA)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    periods (list): รายการช่วงเวลาสำหรับการคำนวณ EMA
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ EMA เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณ EMA สำหรับแต่ละช่วงเวลา
    for period in periods:
        result[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    return result

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    คำนวณ Average True Range (ATR)
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
    period (int): ช่วงเวลาสำหรับการคำนวณ ATR
    
    Returns:
    pd.DataFrame: DataFrame ที่มีคอลัมน์ ATR เพิ่มเติม
    """
    # สร้าง DataFrame สำหรับผลลัพธ์
    result = pd.DataFrame(index=df.index)
    
    # คำนวณช่วงจริง (True Range)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # หาค่าสูงสุดของ tr1, tr2 และ tr3
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # คำนวณ ATR โดยใช้ SMA
    atr = true_range.rolling(window=period).mean()
    
    # เพิ่มคอลัมน์ ATR ลงใน DataFrame ผลลัพธ์
    result[f'atr_{period}'] = atr
    
    # เพิ่มคอลัมน์ ATR เป็นเปอร์เซ็นต์ของราคา
    result[f'atr_percent_{period}'] = (atr / close) * 100
    
    return result