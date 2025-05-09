"""
คลาสพื้นฐานสำหรับการดาวน์โหลดข้อมูลตลาด
"""

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
    """
    คลาสพื้นฐานสำหรับการดาวน์โหลดข้อมูลตลาด
    
    คลาสนี้กำหนดอินเตอร์เฟซพื้นฐานที่ downloader ทุกตัวต้องมี
    """
    
    def __init__(self):
        """
        กำหนดค่าเริ่มต้นสำหรับ downloader พื้นฐาน
        """
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
        """
        ดาวน์โหลดข้อมูลประวัติราคาย้อนหลัง
        
        Parameters:
        symbol (str): คู่สกุลเงิน (เช่น "BTCUSDT")
        timeframe (str): ไทม์เฟรม (เช่น "1m", "5m", "1h")
        start_date (str or datetime): วันเริ่มต้น (YYYY-MM-DD หรือ datetime)
        end_date (str or datetime, optional): วันสิ้นสุด (YYYY-MM-DD หรือ datetime, ค่าเริ่มต้น = วันปัจจุบัน)
        output_dir (str): ไดเรกทอรีที่จะบันทึกข้อมูล
        file_format (str): รูปแบบไฟล์ที่จะบันทึก ("csv", "parquet", หรือ "both")
        include_current_candle (bool): รวมแท่งเทียนปัจจุบันที่ยังไม่ปิดหรือไม่
        
        Returns:
        pd.DataFrame: ข้อมูลประวัติราคาที่ดาวน์โหลด
        """
        pass
    
    @abstractmethod
    def update_data(
        self,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/raw",
        file_format: str = "both"
    ) -> Optional[pd.DataFrame]:
        """
        อัพเดตข้อมูลให้เป็นปัจจุบัน
        
        Parameters:
        symbol (str): คู่สกุลเงิน (เช่น "BTCUSDT")
        timeframe (str): ไทม์เฟรม (เช่น "1m", "5m", "1h")
        data_dir (str): ไดเรกทอรีที่เก็บข้อมูล
        file_format (str): รูปแบบไฟล์ที่จะบันทึก ("csv", "parquet", หรือ "both")
        
        Returns:
        pd.DataFrame หรือ None: ข้อมูลที่อัพเดตแล้ว หรือ None หากไม่มีไฟล์เดิม
        """
        pass
    
    @abstractmethod
    def validate_data(
        self,
        df: pd.DataFrame,
        timeframe: str,
        fill_missing: bool = True,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        ตรวจสอบและแก้ไขข้อมูลให้สมบูรณ์
        
        Parameters:
        df (pd.DataFrame): DataFrame ที่ต้องการตรวจสอบ
        timeframe (str): ไทม์เฟรมของข้อมูล
        fill_missing (bool): เติมข้อมูลที่หายไปหรือไม่
        remove_duplicates (bool): ลบข้อมูลที่ซ้ำกันหรือไม่
        
        Returns:
        pd.DataFrame: DataFrame ที่ผ่านการตรวจสอบแล้ว
        """
        pass
    
    def _parse_date(self, date: Union[str, datetime]) -> int:
        """
        แปลงวันที่เป็น timestamp (มิลลิวินาที)
        
        Parameters:
        date (str or datetime): วันที่ในรูปแบบ "YYYY-MM-DD" หรือ datetime
        
        Returns:
        int: timestamp ในรูปแบบมิลลิวินาที
        
        Raises:
        ValueError: หากรูปแบบวันที่ไม่ถูกต้อง
        """
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
        """
        ค้นหาไฟล์ล่าสุดในไดเรกทอรี (โดยดูจากชื่อไฟล์)
        
        Parameters:
        directory (str): ไดเรกทอรีที่ต้องการค้นหา
        
        Returns:
        str หรือ None: พาธของไฟล์ล่าสุด หรือ None หากไม่พบไฟล์
        """
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
        """
        แปลงข้อมูลแท่งเทียนเป็น pandas DataFrame
        
        Parameters:
        candles (list): ข้อมูลแท่งเทียนจาก API
        
        Returns:
        pd.DataFrame: ข้อมูลแท่งเทียนในรูปแบบ DataFrame
        """
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