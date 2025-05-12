import datetime
import pandas as pd
from typing import Union, Optional, Dict, Tuple, List

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    if not isinstance(timeframe, str):
        raise ValueError(f"ไทม์เฟรมต้องเป็นสตริง, แต่ได้รับ {type(timeframe)}")
    
    timeframe = timeframe.lower()
    value = ''.join(filter(str.isdigit, timeframe))
    unit = ''.join(filter(str.isalpha, timeframe))
    
    if not value or not unit or unit not in ["m", "h", "d", "w", "M"]:
        raise ValueError(f"รูปแบบไทม์เฟรมไม่ถูกต้อง: {timeframe}")
    
    return int(value), unit

def get_timeframe_delta(timeframe: str) -> datetime.timedelta:
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
        return datetime.timedelta(days=30 * value)
    else:
        raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")

def get_timeframe_in_minutes(timeframe: str) -> int:
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
        return value * 30 * 24 * 60
    else:
        raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")

def get_timeframe_in_milliseconds(timeframe: str) -> int:
    return get_timeframe_in_minutes(timeframe) * 60 * 1000

def get_pandas_freq(timeframe: str) -> str:
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
    try:
        parse_timeframe(timeframe)
        return True
    except ValueError:
        return False

def align_timestamp_to_timeframe(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
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
    if isinstance(timestamp, pd.Timestamp):
        return timestamp
    elif isinstance(timestamp, datetime.datetime):
        return pd.Timestamp(timestamp)
    elif isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    elif isinstance(timestamp, (int, float)):
        return pd.Timestamp(timestamp, unit='s' if timestamp < 1e12 else 'ms')
    else:
        raise ValueError(f"ไม่สามารถแปลงประเภท {type(timestamp)} เป็น datetime ได้")
