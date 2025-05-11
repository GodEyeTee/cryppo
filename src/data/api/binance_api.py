import os
import requests
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class BinanceAPI:
    BASE_URL = "https://api.binance.com/api/v3"
    KLINES_ENDPOINT = "/klines"
    EXCHANGE_INFO_ENDPOINT = "/exchangeInfo"
    
    # ตารางแปลงไทม์เฟรมเป็นรูปแบบที่ Binance API ยอมรับ
    TIMEFRAME_MAP = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
    }
    
    # จำนวนแท่งเทียนสูงสุดที่ Binance อนุญาตให้ดาวน์โหลดในแต่ละครั้ง
    MAX_CANDLES_PER_REQUEST = 1000
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        
        # สร้าง headers พื้นฐาน
        self.headers = {}
        if api_key:
            self.headers["X-MBX-APIKEY"] = api_key
        
        # ตรวจสอบการเชื่อมต่อกับ Binance API
        self._check_connection()
    
    def _check_connection(self) -> bool:      
        try:
            response = self.session.get(f"{self.BASE_URL}/ping", headers=self.headers)
            response.raise_for_status()
            logger.info("เชื่อมต่อกับ Binance API สำเร็จ")
            return True
        except Exception as e:
            logger.error(f"ไม่สามารถเชื่อมต่อกับ Binance API ได้: {e}")
            raise ConnectionError(f"ไม่สามารถเชื่อมต่อกับ Binance API ได้: {e}")
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        url = f"{self.BASE_URL}{self.EXCHANGE_INFO_ENDPOINT}"
        if symbol:
            url += f"?symbol={symbol.upper()}"
        
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List:
        url = f"{self.BASE_URL}{self.KLINES_ENDPOINT}"
        params = {
            "symbol": symbol,
            "interval": self.TIMEFRAME_MAP[interval]
        }
        
        if start_time is not None:
            params["startTime"] = start_time
        
        if end_time is not None:
            params["endTime"] = end_time
        
        if limit is not None:
            params["limit"] = min(limit, self.MAX_CANDLES_PER_REQUEST)
        
        try:
            response = self.session.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(response, 'status_code') and response.status_code == 429:
                # Rate limit exceeded
                wait_time = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                return self.get_klines(symbol, interval, start_time, end_time, limit)
            else:
                logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e} - Response: {response.text if hasattr(response, 'text') else 'No response text'}")
                raise
    
    def validate_timeframe(self, timeframe: str) -> str:
        timeframe = timeframe.lower()
        if timeframe not in self.TIMEFRAME_MAP:
            valid_timeframes = ", ".join(self.TIMEFRAME_MAP.keys())
            raise ValueError(f"ไทม์เฟรมไม่ถูกต้อง: {timeframe}. ไทม์เฟรมที่ใช้ได้: {valid_timeframes}")
        return timeframe
    
    def get_interval_ms(self, timeframe: str) -> int:
        timeframe = timeframe.lower()
        value = int(''.join(filter(str.isdigit, timeframe)))
        unit = ''.join(filter(str.isalpha, timeframe))
        
        if unit == "m":
            return value * 60 * 1000
        elif unit == "h":
            return value * 60 * 60 * 1000
        elif unit == "d":
            return value * 24 * 60 * 60 * 1000
        elif unit == "w":
            return value * 7 * 24 * 60 * 60 * 1000
        elif unit == "M":
            # ประมาณ 30 วันต่อเดือน
            return value * 30 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"หน่วยไทม์เฟรมไม่รองรับ: {unit}")
