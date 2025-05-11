import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class BaseDownloader(ABC):    
    def __init__(self):
        pass
    
    @abstractmethod
    def download_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime, None] = None,
        output_dir: str = "data/raw",
        file_format: str = "both",
        include_current_candle: bool = False
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def update_data(
        self,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/raw",
        file_format: str = "both"
    ) -> Optional[pd.DataFrame]:
        pass
    
    @abstractmethod
    def validate_data(
        self,
        df: pd.DataFrame,
        timeframe: str,
        fill_missing: bool = True,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        pass
    
    def _parse_date(self, date: Union[str, datetime]) -> int:
        if isinstance(date, str):
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                try:
                    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValueError(f"รูปแบบวันที่ไม่ถูกต้อง: {date}. รูปแบบที่รองรับ: YYYY-MM-DD หรือ YYYY-MM-DD HH:MM:SS")
        elif isinstance(date, datetime):
            date_obj = date
        else:
            raise ValueError(f"ประเภทข้อมูลไม่ถูกต้อง: {type(date)}. ต้องเป็น str หรือ datetime")
        
        # แปลงเป็น timestamp (มิลลิวินาที)
        return int(date_obj.timestamp() * 1000)
    
    def _find_latest_file(self, directory: str) -> Optional[str]:
        files = [f for f in os.listdir(directory) if f.endswith(('.csv', '.parquet'))]
        
        if not files:
            return None
        
        # เรียงไฟล์ตามวันที่ในชื่อไฟล์ (ถ้ามี)
        date_files = []
        combined_files = []
        
        for f in files:
            if 'combined' in f:
                combined_files.append(f)
            else:
                # แยกวันที่จากชื่อไฟล์ (รูปแบบ: symbol_timeframe_YYYYMMDD_YYYYMMDD.ext)
                parts = f.split('_')
                if len(parts) >= 4:
                    try:
                        end_date = datetime.strptime(parts[-1].split('.')[0], '%Y%m%d')
                        date_files.append((f, end_date))
                    except ValueError:
                        date_files.append((f, datetime.min))
                else:
                    date_files.append((f, datetime.min))
        
        # ถ้ามีไฟล์ combined ให้ใช้ไฟล์นั้น (เพราะมักจะรวมข้อมูลทั้งหมด)
        if combined_files:
            # เลือกไฟล์ที่มี format ที่ต้องการ (.parquet มีความสำคัญสูงกว่า .csv)
            parquet_files = [f for f in combined_files if f.endswith('.parquet')]
            if parquet_files:
                return os.path.join(directory, parquet_files[0])
            return os.path.join(directory, combined_files[0])
        
        # ถ้าไม่มีไฟล์ combined ให้ใช้ไฟล์ที่มีวันที่ล่าสุด
        if date_files:
            latest_file, _ = max(date_files, key=lambda x: x[1])
            return os.path.join(directory, latest_file)
        
        # ถ้าไม่สามารถระบุได้จากชื่อไฟล์ ให้ใช้ไฟล์ที่แก้ไขล่าสุด
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        return os.path.join(directory, latest_file)
    
    def _candles_to_dataframe(self, candles: List) -> pd.DataFrame:
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base_volume",
            "taker_buy_quote_volume", "ignored"
        ]
        
        df = pd.DataFrame(candles, columns=columns)
        
        # แปลงประเภทข้อมูล
        numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume", 
                         "taker_buy_base_volume", "taker_buy_quote_volume"]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        df["trades"] = df["trades"].astype(int)
        
        # แปลง timestamp เป็น datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # ลบคอลัมน์ที่ไม่จำเป็น
        df = df.drop(columns=["ignored"])
        
        return df
