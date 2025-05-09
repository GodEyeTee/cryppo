"""
ยูทิลิตี้เกี่ยวกับเวลาสำหรับ CRYPPO (Cryptocurrency Position Optimization)
"""

import datetime
import pandas as pd
from typing import Union, Optional, Dict, Tuple, List

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    แยกตัวเลขและหน่วยจากรูปแบบไทม์เฟรม
    
    Parameters:
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    tuple: (จำนวน, หน่วย)
    
    Raises:
    ValueError: หากรูปแบบไทม์เฟรมไม่ถูกต้อง
    """
    if not isinstance(timeframe, str):
        raise ValueError(f"ไทม์เฟรมต้องเป็นสตริง, แต่ได้รับ {type(timeframe)}")
    
    timeframe = timeframe.lower()
    value = ''.join(filter(str.isdigit, timeframe))
    unit = ''.join(filter(str.isalpha, timeframe))
    
    if not value or not unit or unit not in ["m", "h", "d", "w", "M"]:
        raise ValueError(f"รูปแบบไทม์เฟรมไม่ถูกต้อง: {timeframe}")
    
    return int(value), unit

def get_timeframe_delta(timeframe: str) -> datetime.timedelta:
    """
    แปลงไทม์เฟรมเป็น timedelta
    
    Parameters:
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    datetime.timedelta: ช่วงเวลาที่เทียบเท่า
    
    Raises:
    ValueError: หากรูปแบบไทม์เฟรมไม่ถูกต้อง
    """
    value, unit = parse_timeframe(timeframe)
    
    if unit == "m":
        return datetime.timedelta(minutes=value)
    elif unit == "h":
        return datetime.timedelta(hours=value)
    elif unit == "d":
        return datetime.timedelta(days=value)
    elif unit == "w":
        return datetime.timedelta(weeks=value)
    elif unit == "M":
        # ประมาณ 30 วันต่อเดือน
        return datetime.timedelta(days=30 * value)
    else:
        raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")

def get_timeframe_in_minutes(timeframe: str) -> int:
    """
    แปลงไทม์เฟรมเป็นจำนวนนาที
    
    Parameters:
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    int: จำนวนนาทีที่เทียบเท่า
    
    Raises:
    ValueError: หากรูปแบบไทม์เฟรมไม่รองรับ
    """
    value, unit = parse_timeframe(timeframe)
    
    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 24 * 60
    elif unit == "w":
        return value * 7 * 24 * 60
    elif unit == "M":
        # ประมาณ 30 วันต่อเดือน
        return value * 30 * 24 * 60
    else:
        raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")

def get_timeframe_in_milliseconds(timeframe: str) -> int:
    """
    แปลงไทม์เฟรมเป็นจำนวนมิลลิวินาที
    
    Parameters:
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    int: จำนวนมิลลิวินาทีที่เทียบเท่า
    
    Raises:
    ValueError: หากรูปแบบไทม์เฟรมไม่รองรับ
    """
    return get_timeframe_in_minutes(timeframe) * 60 * 1000

def get_pandas_freq(timeframe: str) -> str:
    """
    แปลงไทม์เฟรมเป็นรูปแบบความถี่ของ pandas
    
    Parameters:
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    str: รูปแบบความถี่ของ pandas
    
    Raises:
    ValueError: หากรูปแบบไทม์เฟรมไม่รองรับ
    """
    value, unit = parse_timeframe(timeframe)
    
    if unit == "m":
        return f"{value}min"
    elif unit == "h":
        return f"{value}h"
    elif unit == "d":
        return f"{value}d"
    elif unit == "w":
        return f"{value}W"
    elif unit == "M":
        return f"{value}M"
    else:
        raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")

def is_timeframe_valid(timeframe: str) -> bool:
    """
    ตรวจสอบว่าไทม์เฟรมถูกต้องหรือไม่
    
    Parameters:
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    bool: True หากถูกต้อง, False หากไม่ถูกต้อง
    """
    try:
        parse_timeframe(timeframe)
        return True
    except ValueError:
        return False

def align_timestamp_to_timeframe(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """
    ปรับ timestamp ให้ตรงกับรูปแบบไทม์เฟรม
    
    Parameters:
    timestamp (pd.Timestamp): timestamp ที่ต้องการปรับ
    timeframe (str): ไทม์เฟรม เช่น "1m", "5m", "1h"
    
    Returns:
    pd.Timestamp: timestamp ที่ปรับแล้ว
    
    Raises:
    ValueError: หากรูปแบบไทม์เฟรมไม่รองรับ
    """
    value, unit = parse_timeframe(timeframe)
    
    if unit == "m":
        return timestamp.replace(
            second=0, microsecond=0,
            minute=(timestamp.minute // value) * value
        )
    elif unit == "h":
        return timestamp.replace(
            second=0, microsecond=0, minute=0,
            hour=(timestamp.hour // value) * value
        )
    elif unit == "d":
        return timestamp.replace(
            second=0, microsecond=0, minute=0, hour=0
        )
    elif unit == "w":
        # วันจันทร์เป็นวันแรกของสัปดาห์ (weekday=0 คือวันจันทร์)
        days_since_monday = timestamp.weekday()
        return (timestamp - pd.Timedelta(days=days_since_monday)).replace(
            second=0, microsecond=0, minute=0, hour=0
        )
    elif unit == "M":
        return timestamp.replace(
            second=0, microsecond=0, minute=0, hour=0, day=1
        )
    else:
        raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")

def convert_to_datetime(timestamp: Union[str, int, pd.Timestamp, datetime.datetime]) -> pd.Timestamp:
    """
    แปลง timestamp หลายรูปแบบเป็น pandas Timestamp
    
    Parameters:
    timestamp: timestamp หลายรูปแบบ (str, int, pd.Timestamp, datetime.datetime)
    
    Returns:
    pd.Timestamp: pandas Timestamp ที่แปลงแล้ว
    
    Raises:
    ValueError: หากรูปแบบไม่รองรับ
    """
    if isinstance(timestamp, pd.Timestamp):
        return timestamp
    elif isinstance(timestamp, datetime.datetime):
        return pd.Timestamp(timestamp)
    elif isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    elif isinstance(timestamp, (int, float)):
        # ถ้าเป็น Unix timestamp ในรูปแบบวินาที
        if timestamp < 1e12:
            return pd.Timestamp(timestamp, unit='s')
        # ถ้าเป็น Unix timestamp ในรูปแบบมิลลิวินาที
        else:
            return pd.Timestamp(timestamp, unit='ms')
    else:
        raise ValueError(f"ไม่สามารถแปลงประเภท {type(timestamp)} เป็น datetime ได้")