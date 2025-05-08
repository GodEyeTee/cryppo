import os
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
import requests
from pathlib import Path

# ตั้งค่า logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
   handler = logging.StreamHandler()
   handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
   logger.addHandler(handler)

class BinanceDownloader:
   """
   ดาวน์โหลดข้อมูลประวัติราคาจาก Binance API
   
   คุณสมบัติ:
   - ดาวน์โหลดข้อมูล OHLCV (Open, High, Low, Close, Volume)
   - รองรับหลายไทม์เฟรม (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
   - จัดการการจำกัด rate limit ของ Binance โดยอัตโนมัติ
   - บันทึกข้อมูลในรูปแบบ CSV และ/หรือ Parquet
   """
   
   BASE_URL = "https://api.binance.com/api/v3"
   KLINES_ENDPOINT = "/klines"
   EXCHANGE_INFO_ENDPOINT = "/exchangeInfo"
   
   # ตารางแปลงไทม์เฟรมเป็นรูปแบบที่ Binance API ยอมรับ
   TIMEFRAME_MAP = {
       "1m": "1m",
       "3m": "3m",
       "5m": "5m",
       "15m": "15m",
       "30m": "30m",
       "1h": "1h",
       "2h": "2h",
       "4h": "4h",
       "6h": "6h",
       "8h": "8h",
       "12h": "12h",
       "1d": "1d",
       "3d": "3d",
       "1w": "1w",
       "1M": "1M"
   }
   
   # จำนวนแท่งเทียนสูงสุดที่ Binance อนุญาตให้ดาวน์โหลดในแต่ละครั้ง
   MAX_CANDLES_PER_REQUEST = 1000
   
   def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
       """
       กำหนดค่าเริ่มต้นสำหรับ Binance Downloader
       
       Parameters:
       api_key (str, optional): Binance API key
       api_secret (str, optional): Binance API secret
       
       หมายเหตุ: API key และ API secret ไม่จำเป็นสำหรับการดึงข้อมูลประวัติราคา
       แต่อาจจำเป็นสำหรับฟีเจอร์อื่นๆ ในอนาคต และช่วยเพิ่ม rate limits
       """
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
       """
       ตรวจสอบการเชื่อมต่อกับ Binance API
       
       Returns:
       bool: True หากการเชื่อมต่อสำเร็จ, False หากไม่สำเร็จ
       
       Raises:
       ConnectionError: หากไม่สามารถเชื่อมต่อกับ Binance API ได้
       """
       try:
           response = self.session.get(f"{self.BASE_URL}/ping", headers=self.headers)
           response.raise_for_status()
           logger.info("เชื่อมต่อกับ Binance API สำเร็จ")
           return True
       except Exception as e:
           logger.error(f"ไม่สามารถเชื่อมต่อกับ Binance API ได้: {e}")
           raise ConnectionError(f"ไม่สามารถเชื่อมต่อกับ Binance API ได้: {e}")
   
   def _get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
       """
       ดึงข้อมูลเกี่ยวกับคู่สกุลเงินจาก Binance
       
       Parameters:
       symbol (str, optional): คู่สกุลเงินที่ต้องการข้อมูล (ไม่มี = ทุกคู่)
       
       Returns:
       dict: ข้อมูลคู่สกุลเงิน
       """
       url = f"{self.BASE_URL}{self.EXCHANGE_INFO_ENDPOINT}"
       if symbol:
           url += f"?symbol={symbol.upper()}"
       
       response = self.session.get(url, headers=self.headers)
       response.raise_for_status()
       return response.json()
   
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
       ดาวน์โหลดข้อมูลประวัติราคาย้อนหลังจาก Binance
       
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
       # ตรวจสอบและแปลงพารามิเตอร์
       symbol = symbol.upper()
       timeframe = self._validate_timeframe(timeframe)
       start_ts = self._parse_date(start_date)
       
       # ถ้าไม่ระบุ end_date ให้ใช้วันปัจจุบัน
       if end_date is None:
           end_ts = int(datetime.now().timestamp() * 1000)
       else:
           end_ts = self._parse_date(end_date)
       
       # สร้างไดเรกทอรีสำหรับบันทึกข้อมูล
       symbol_dir = os.path.join(output_dir, symbol.replace("/", "-"))
       timeframe_dir = os.path.join(symbol_dir, timeframe)
       os.makedirs(timeframe_dir, exist_ok=True)
       
       logger.info(f"เริ่มดาวน์โหลดข้อมูล {symbol} ไทม์เฟรม {timeframe} ตั้งแต่ {datetime.fromtimestamp(start_ts/1000)} ถึง {datetime.fromtimestamp(end_ts/1000)}")
       
       # คำนวณจำนวนแท่งเทียนทั้งหมดที่ต้องดาวน์โหลด
       interval_ms = self._get_interval_ms(timeframe)
       total_candles = (end_ts - start_ts) // interval_ms
       
       if total_candles <= 0:
           logger.warning(f"ไม่มีข้อมูลในช่วงเวลาที่กำหนด: {start_date} ถึง {end_date}")
           return pd.DataFrame()
       
       logger.info(f"ประมาณการจำนวนแท่งเทียนที่ต้องดาวน์โหลด: {total_candles}")
       
       # ดาวน์โหลดข้อมูลเป็นชุดๆ
       all_candles = []
       current_start_ts = start_ts
       
       with tqdm(total=total_candles, desc=f"ดาวน์โหลด {symbol} {timeframe}") as pbar:
           while current_start_ts < end_ts:
               # คำนวณเวลาสิ้นสุดสำหรับแต่ละชุด
               current_end_ts = min(
                   current_start_ts + (self.MAX_CANDLES_PER_REQUEST * interval_ms),
                   end_ts
               )
               
               # ดาวน์โหลดแท่งเทียนสำหรับช่วงเวลานี้
               candles = self._get_klines(
                   symbol=symbol,
                   interval=timeframe,
                   start_time=current_start_ts,
                   end_time=current_end_ts,
                   limit=self.MAX_CANDLES_PER_REQUEST
               )
               
               if not candles:
                   logger.warning(f"ไม่พบข้อมูลในช่วง {datetime.fromtimestamp(current_start_ts/1000)} ถึง {datetime.fromtimestamp(current_end_ts/1000)}")
                   current_start_ts = current_end_ts
                   continue
               
               # เก็บข้อมูลที่ดาวน์โหลดได้
               all_candles.extend(candles)
               
               # อัพเดต progress bar
               pbar.update(len(candles))
               
               # เตรียมสำหรับการดาวน์โหลดชุดต่อไป
               if len(candles) < self.MAX_CANDLES_PER_REQUEST:
                   # ถ้าได้รับข้อมูลน้อยกว่าที่ขอ แสดงว่าไม่มีข้อมูลเพิ่มเติมแล้ว
                   break
               
               # เลื่อนเวลาเริ่มต้นสำหรับชุดต่อไป
               current_start_ts = candles[-1][0] + interval_ms
               
               # หน่วงเวลาเพื่อไม่ให้เกิน rate limit
               time.sleep(0.5)
       
       # ตรวจสอบว่ามีข้อมูลหรือไม่
       if not all_candles:
           logger.warning(f"ไม่พบข้อมูลสำหรับ {symbol} ไทม์เฟรม {timeframe} ในช่วงเวลาที่กำหนด")
           return pd.DataFrame()
       
       # แปลงข้อมูลเป็น DataFrame
       df = self._candles_to_dataframe(all_candles)
       
       # ลบแท่งเทียนปัจจุบันที่ยังไม่ปิด (ถ้าต้องการ)
       if not include_current_candle:
           current_time = datetime.now()
           if timeframe == "1m":
               # ถ้าเป็นไทม์เฟรม 1 นาที ให้ลบแท่งเทียนสุดท้ายที่อยู่ในนาทีปัจจุบัน
               df = df[df['timestamp'] < pd.Timestamp(current_time.replace(second=0, microsecond=0))]
           else:
               # สำหรับไทม์เฟรมอื่นๆ ให้ลบแท่งเทียนสุดท้ายเพื่อความปลอดภัย
               df = df.iloc[:-1]
       
       # บันทึกข้อมูล
       file_prefix = f"{symbol.lower()}_{timeframe}_{datetime.fromtimestamp(start_ts/1000).strftime('%Y%m%d')}_{datetime.fromtimestamp(end_ts/1000).strftime('%Y%m%d')}"
       
       if file_format.lower() in ["csv", "both"]:
           csv_path = os.path.join(timeframe_dir, f"{file_prefix}.csv")
           df.to_csv(csv_path, index=False)
           logger.info(f"บันทึกข้อมูลเป็น CSV ที่: {csv_path}")
       
       if file_format.lower() in ["parquet", "both"]:
           parquet_path = os.path.join(timeframe_dir, f"{file_prefix}.parquet")
           df.to_parquet(parquet_path, index=False)
           logger.info(f"บันทึกข้อมูลเป็น Parquet ที่: {parquet_path}")
       
       logger.info(f"ดาวน์โหลดข้อมูลสำเร็จ: {len(df)} แท่งเทียน")
       return df
   
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
       symbol = symbol.upper()
       timeframe = self._validate_timeframe(timeframe)
       
       # ค้นหาไฟล์ล่าสุด
       symbol_dir = os.path.join(data_dir, symbol.replace("/", "-"))
       timeframe_dir = os.path.join(symbol_dir, timeframe)
       
       if not os.path.exists(timeframe_dir):
           logger.warning(f"ไม่พบไดเรกทอรี {timeframe_dir}")
           return None
       
       # ค้นหาไฟล์ Parquet หรือ CSV ล่าสุด
       latest_file = self._find_latest_file(timeframe_dir)
       
       if not latest_file:
           logger.warning(f"ไม่พบไฟล์ข้อมูลใน {timeframe_dir}")
           return None
       
       # โหลดข้อมูลจากไฟล์ล่าสุด
       if latest_file.endswith('.parquet'):
           df = pd.read_parquet(latest_file)
       else:  # .csv
           df = pd.read_csv(latest_file)
           df['timestamp'] = pd.to_datetime(df['timestamp'])
       
       # หาวันที่ล่าสุดในข้อมูล
       last_timestamp = df['timestamp'].max()
       
       # เพิ่ม 1 ช่วงเวลาเพื่อหลีกเลี่ยงการดาวน์โหลดซ้ำ
       interval_ms = self._get_interval_ms(timeframe)
       start_ts = int(last_timestamp.timestamp() * 1000) + interval_ms
       
       # ถ้าเวลาเริ่มต้นอยู่ในอนาคต ให้ถือว่าข้อมูลเป็นปัจจุบันแล้ว
       if start_ts > int(datetime.now().timestamp() * 1000):
           logger.info(f"ข้อมูล {symbol} {timeframe} เป็นปัจจุบันแล้ว")
           return df
       
       # ดาวน์โหลดข้อมูลใหม่
       logger.info(f"กำลังอัพเดตข้อมูล {symbol} {timeframe} ตั้งแต่ {datetime.fromtimestamp(start_ts/1000)}")
       
       # ดาวน์โหลดข้อมูลใหม่
       new_df = self.download_historical_data(
           symbol=symbol,
           timeframe=timeframe,
           start_date=datetime.fromtimestamp(start_ts/1000),
           output_dir=data_dir,
           file_format=file_format,
           include_current_candle=False
       )
       
       if new_df.empty:
           logger.info(f"ไม่มีข้อมูลใหม่สำหรับ {symbol} {timeframe}")
           return df
       
       # รวมข้อมูลเก่าและใหม่
       combined_df = pd.concat([df, new_df], ignore_index=True)
       combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
       
       # บันทึกข้อมูลที่รวมแล้ว
       file_prefix = f"{symbol.lower()}_{timeframe}_combined"
       
       if file_format.lower() in ["csv", "both"]:
           csv_path = os.path.join(timeframe_dir, f"{file_prefix}.csv")
           combined_df.to_csv(csv_path, index=False)
           logger.info(f"บันทึกข้อมูลรวมเป็น CSV ที่: {csv_path}")
       
       if file_format.lower() in ["parquet", "both"]:
           parquet_path = os.path.join(timeframe_dir, f"{file_prefix}.parquet")
           combined_df.to_parquet(parquet_path, index=False)
           logger.info(f"บันทึกข้อมูลรวมเป็น Parquet ที่: {parquet_path}")
       
       logger.info(f"อัพเดตข้อมูลสำเร็จ: เพิ่ม {len(new_df)} แท่งเทียน รวมทั้งหมด {len(combined_df)} แท่งเทียน")
       return combined_df
   
   def _validate_timeframe(self, timeframe: str) -> str:
       """
       ตรวจสอบว่าไทม์เฟรมที่ระบุถูกต้องหรือไม่
       
       Parameters:
       timeframe (str): ไทม์เฟรมที่ต้องการตรวจสอบ
       
       Returns:
       str: ไทม์เฟรมที่ถูกต้อง
       
       Raises:
       ValueError: หากไทม์เฟรมไม่ถูกต้อง
       """
       timeframe = timeframe.lower()
       if timeframe not in self.TIMEFRAME_MAP:
           valid_timeframes = ", ".join(self.TIMEFRAME_MAP.keys())
           raise ValueError(f"ไทม์เฟรมไม่ถูกต้อง: {timeframe}. ไทม์เฟรมที่ใช้ได้: {valid_timeframes}")
       return timeframe
   
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
   
   def _get_interval_ms(self, timeframe: str) -> int:
       """
       คำนวณความยาวของไทม์เฟรมในหน่วยมิลลิวินาที
       
       Parameters:
       timeframe (str): ไทม์เฟรม (เช่น "1m", "5m", "1h")
       
       Returns:
       int: ความยาวของไทม์เฟรมในหน่วยมิลลิวินาที
       """
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
   
   def _get_klines(
       self,
       symbol: str,
       interval: str,
       start_time: Optional[int] = None,
       end_time: Optional[int] = None,
       limit: Optional[int] = None
   ) -> List:
       """
       ดึงข้อมูลแท่งเทียนจาก Binance API
       
       Parameters:
       symbol (str): คู่สกุลเงิน
       interval (str): ไทม์เฟรม
       start_time (int, optional): เวลาเริ่มต้นในรูปแบบ timestamp (มิลลิวินาที)
       end_time (int, optional): เวลาสิ้นสุดในรูปแบบ timestamp (มิลลิวินาที)
       limit (int, optional): จำนวนแท่งเทียนสูงสุดที่ต้องการ (สูงสุด 1000)
       
       Returns:
       list: ข้อมูลแท่งเทียน
       
       Raises:
       requests.exceptions.RequestException: หากเกิดข้อผิดพลาดในการดึงข้อมูล
       """
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
           if response.status_code == 429:
               # Rate limit exceeded
               wait_time = int(response.headers.get("Retry-After", 10))
               logger.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
               time.sleep(wait_time)
               return self._get_klines(symbol, interval, start_time, end_time, limit)
           else:
               logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e} - Response: {response.text}")
               raise
   
   def _candles_to_dataframe(self, candles: List) -> pd.DataFrame:
       """
       แปลงข้อมูลแท่งเทียนเป็น pandas DataFrame
       
       Parameters:
       candles (list): ข้อมูลแท่งเทียนจาก Binance API
       
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
   
   def download_multiple_timeframes(
       self,
       symbol: str,
       timeframes: List[str],
       start_date: Union[str, datetime],
       end_date: Union[str, datetime, None] = None,
       output_dir: str = "data/raw",
       file_format: str = "both"
   ) -> Dict[str, pd.DataFrame]:
       """
       ดาวน์โหลดข้อมูลหลายไทม์เฟรมพร้อมกัน
       
       Parameters:
       symbol (str): คู่สกุลเงิน (เช่น "BTCUSDT")
       timeframes (list): รายการไทม์เฟรม (เช่น ["1m", "5m", "1h"])
       start_date (str or datetime): วันเริ่มต้น (YYYY-MM-DD หรือ datetime)
       end_date (str or datetime, optional): วันสิ้นสุด (YYYY-MM-DD หรือ datetime, ค่าเริ่มต้น = วันปัจจุบัน)
       output_dir (str): ไดเรกทอรีที่จะบันทึกข้อมูล
       file_format (str): รูปแบบไฟล์ที่จะบันทึก ("csv", "parquet", หรือ "both")
       
       Returns:
       dict: Dictionary ที่มีไทม์เฟรมเป็นคีย์และ DataFrame เป็นค่า
       """
       results = {}
       
       for timeframe in timeframes:
           logger.info(f"กำลังดาวน์โหลดข้อมูล {symbol} ไทม์เฟรม {timeframe}")
           df = self.download_historical_data(
               symbol=symbol,
               timeframe=timeframe,
               start_date=start_date,
               end_date=end_date,
               output_dir=output_dir,
               file_format=file_format
           )
           results[timeframe] = df
       
       return results
   
   def update_multiple_timeframes(
       self,
       symbol: str,
       timeframes: List[str],
       data_dir: str = "data/raw",
       file_format: str = "both"
   ) -> Dict[str, Optional[pd.DataFrame]]:
       """
       อัพเดตข้อมูลหลายไทม์เฟรมพร้อมกัน
       
       Parameters:
       symbol (str): คู่สกุลเงิน (เช่น "BTCUSDT")
       timeframes (list): รายการไทม์เฟรม (เช่น ["1m", "5m", "1h"])
       data_dir (str): ไดเรกทอรีที่เก็บข้อมูล
       file_format (str): รูปแบบไฟล์ที่จะบันทึก ("csv", "parquet", หรือ "both")
       
       Returns:
       dict: Dictionary ที่มีไทม์เฟรมเป็นคีย์และ DataFrame เป็นค่า
       """
       results = {}
       
       for timeframe in timeframes:
           logger.info(f"กำลังอัพเดตข้อมูล {symbol} ไทม์เฟรม {timeframe}")
           df = self.update_data(
               symbol=symbol,
               timeframe=timeframe,
               data_dir=data_dir,
               file_format=file_format
           )
           results[timeframe] = df
       
       return results
   
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
       if df.empty:
           logger.warning("DataFrame ว่างเปล่า")
           return df
       
       # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นครบหรือไม่
       required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
       missing_columns = [col for col in required_columns if col not in df.columns]
       
       if missing_columns:
           raise ValueError(f"คอลัมน์ที่จำเป็นหายไป: {missing_columns}")
       
       # ตรวจสอบว่า timestamp เป็น datetime หรือไม่
       if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
           df['timestamp'] = pd.to_datetime(df['timestamp'])
       
       original_length = len(df)
       
       # ลบข้อมูลที่ซ้ำกัน
       if remove_duplicates:
           df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
           if len(df) < original_length:
               logger.info(f"ลบข้อมูลที่ซ้ำกัน {original_length - len(df)} แถว")
       
       # เรียงข้อมูลตาม timestamp
       df = df.sort_values('timestamp').reset_index(drop=True)
       
       # ตรวจสอบและเติมข้อมูลที่หายไป
       if fill_missing and len(df) > 1:
           # คำนวณช่วงเวลาระหว่างแท่งเทียน
           interval_ms = self._get_interval_ms(timeframe)
           expected_diff = pd.Timedelta(milliseconds=interval_ms)
           
           # ตรวจสอบว่ามีช่องว่างในข้อมูลหรือไม่
           timestamps = df['timestamp'].copy()
           diffs = timestamps.diff()
           
           # หาช่องว่างในข้อมูล
           gaps = diffs[diffs > expected_diff]
           
           if not gaps.empty:
               logger.info(f"พบช่องว่างในข้อมูล {len(gaps)} ช่อง")
               
               if fill_missing:
                   # สร้างช่วงเวลาที่สมบูรณ์
                   start_time = df['timestamp'].min()
                   end_time = df['timestamp'].max()
                   
                   # สร้าง timestamp ที่สมบูรณ์
                   complete_timestamps = pd.date_range(
                       start=start_time,
                       end=end_time,
                       freq=pd.Timedelta(milliseconds=interval_ms)
                   )
                   
                   # สร้าง DataFrame ใหม่ด้วย timestamp ที่สมบูรณ์
                   complete_df = pd.DataFrame({'timestamp': complete_timestamps})
                   
                   # รวมกับข้อมูลเดิม
                   merged_df = pd.merge(complete_df, df, on='timestamp', how='left')
                   
                   # เติมข้อมูลที่หายไป
                   # ใช้ราคาปิดล่าสุดเป็นราคาเปิด สูง ต่ำ และปิดของแท่งเทียนที่หายไป
                   # ตั้งปริมาณเป็น 0
                   merged_df['close'] = merged_df['close'].fillna(method='ffill')
                   merged_df['open'] = merged_df['open'].fillna(merged_df['close'])
                   merged_df['high'] = merged_df['high'].fillna(merged_df['close'])
                   merged_df['low'] = merged_df['low'].fillna(merged_df['close'])
                   merged_df['volume'] = merged_df['volume'].fillna(0)
                   
                   # เติมคอลัมน์อื่นๆ
                   for col in df.columns:
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                           if col in ['quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']:
                               merged_df[col] = merged_df[col].fillna(0)
                           elif col in ['trades']:
                               merged_df[col] = merged_df[col].fillna(0).astype(int)
                           elif col == 'close_time':
                               # คำนวณ close_time จาก timestamp (timestamp + interval - 1ms)
                               if 'close_time' not in merged_df.columns:
                                   merged_df['close_time'] = merged_df['timestamp'] + pd.Timedelta(milliseconds=interval_ms-1)
                               else:
                                   merged_df['close_time'] = merged_df['close_time'].fillna(
                                       merged_df['timestamp'] + pd.Timedelta(milliseconds=interval_ms-1)
                                   )
                           else:
                               merged_df[col] = merged_df[col].fillna(method='ffill')
                   
                   logger.info(f"เติมข้อมูลที่หายไป {len(merged_df) - len(df)} แถว")
                   return merged_df
       
       return df
   
   def convert_timeframe(
       self,
       df: pd.DataFrame,
       source_timeframe: str,
       target_timeframe: str
   ) -> pd.DataFrame:
       """
       แปลงข้อมูลจากไทม์เฟรมหนึ่งไปยังอีกไทม์เฟรมหนึ่ง
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ต้องการแปลง
       source_timeframe (str): ไทม์เฟรมต้นฉบับ
       target_timeframe (str): ไทม์เฟรมเป้าหมาย
       
       Returns:
       pd.DataFrame: DataFrame ที่ผ่านการแปลงแล้ว
       
       Raises:
       ValueError: หากไม่สามารถแปลงไทม์เฟรมได้
       """
       if df.empty:
           logger.warning("DataFrame ว่างเปล่า")
           return df
       
       # ตรวจสอบว่า timestamp เป็น datetime หรือไม่
       if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
           df['timestamp'] = pd.to_datetime(df['timestamp'])
       
       # ตรวจสอบว่าไทม์เฟรมถูกต้องหรือไม่
       source_timeframe = self._validate_timeframe(source_timeframe)
       target_timeframe = self._validate_timeframe(target_timeframe)
       
       # ตรวจสอบว่าสามารถแปลงไทม์เฟรมได้หรือไม่
       source_interval_ms = self._get_interval_ms(source_timeframe)
       target_interval_ms = self._get_interval_ms(target_timeframe)
       
       if target_interval_ms < source_interval_ms:
           raise ValueError(f"ไม่สามารถแปลงจากไทม์เฟรมใหญ่ ({source_timeframe}) ไปเป็นไทม์เฟรมเล็ก ({target_timeframe}) ได้")
       
       if target_interval_ms % source_interval_ms != 0:
           raise ValueError(f"ไม่สามารถแปลงจากไทม์เฟรม {source_timeframe} ไปเป็น {target_timeframe} ได้ (ไม่เป็นสัดส่วนกัน)")
       
       # กำหนดรูปแบบการรวมข้อมูล
       ohlc_dict = {
           'open': 'first',
           'high': 'max',
           'low': 'min',
           'close': 'last',
           'volume': 'sum',
       }
       
       # เพิ่มคอลัมน์อื่นๆ ถ้ามี
       for col in df.columns:
           if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']:
               if col in ['quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']:
                   ohlc_dict[col] = 'sum'
               elif col in ['trades']:
                   ohlc_dict[col] = 'sum'
               else:
                   ohlc_dict[col] = 'last'
       
       # กำหนดขอบเขตเวลาสำหรับการรวมข้อมูล
       # ตัดเวลาให้ตรงกับขอบเขตของไทม์เฟรมเป้าหมาย
       
       # หาจุดเริ่มต้นที่พอดีกับไทม์เฟรมเป้าหมาย
       first_timestamp = df['timestamp'].min()
       
       # คำนวณ offset จาก timestamp แรกไปยังเวลาที่พอดีกับไทม์เฟรมเป้าหมาย
       if target_timeframe.endswith('m'):
           # ไทม์เฟรมนาที
           target_start = first_timestamp.replace(
               second=0, microsecond=0,
               minute=(first_timestamp.minute // int(target_timeframe[:-1])) * int(target_timeframe[:-1])
           )
       elif target_timeframe.endswith('h'):
           # ไทม์เฟรมชั่วโมง
           target_start = first_timestamp.replace(
               second=0, microsecond=0, minute=0,
               hour=(first_timestamp.hour // int(target_timeframe[:-1])) * int(target_timeframe[:-1])
           )
       elif target_timeframe.endswith('d'):
           # ไทม์เฟรมวัน
           target_start = first_timestamp.replace(
               second=0, microsecond=0, minute=0, hour=0
           )
       elif target_timeframe.endswith('w'):
           # ไทม์เฟรมสัปดาห์
           # วันจันทร์เป็นวันแรกของสัปดาห์ (weekday=0 คือวันจันทร์)
           days_since_monday = first_timestamp.weekday()
           target_start = (first_timestamp - pd.Timedelta(days=days_since_monday)).replace(
               second=0, microsecond=0, minute=0, hour=0
           )
       elif target_timeframe.endswith('M'):
           # ไทม์เฟรมเดือน
           target_start = first_timestamp.replace(
               second=0, microsecond=0, minute=0, hour=0, day=1
           )
       else:
           target_start = first_timestamp
       
       # กรองข้อมูลที่อยู่ก่อนจุดเริ่มต้นที่พอดีกับไทม์เฟรมเป้าหมาย
       df = df[df['timestamp'] >= target_start].copy()
       
       # สร้างคอลัมน์สำหรับการรวมข้อมูล
       if target_timeframe.endswith('m'):
           # ไทม์เฟรมนาที
           df['group'] = df['timestamp'].dt.floor(f"{target_timeframe[:-1]}min")
       elif target_timeframe.endswith('h'):
           # ไทม์เฟรมชั่วโมง
           df['group'] = df['timestamp'].dt.floor(f"{target_timeframe[:-1]}h")
       elif target_timeframe.endswith('d'):
           # ไทม์เฟรมวัน
           df['group'] = df['timestamp'].dt.floor('d')
       elif target_timeframe.endswith('w'):
           # ไทม์เฟรมสัปดาห์
           df['group'] = df['timestamp'].dt.to_period('W').dt.start_time
       elif target_timeframe.endswith('M'):
           # ไทม์เฟรมเดือน
           df['group'] = df['timestamp'].dt.to_period('M').dt.start_time
       else:
           raise ValueError(f"ไม่รองรับไทม์เฟรม: {target_timeframe}")
       
       # รวมข้อมูลตามกลุ่ม
       df_ohlc = df.groupby('group').agg(ohlc_dict)
       
       # รีเซ็ตดัชนีและเปลี่ยนชื่อคอลัมน์
       df_ohlc = df_ohlc.reset_index()
       df_ohlc = df_ohlc.rename(columns={'group': 'timestamp'})
       
       # เพิ่มคอลัมน์ close_time ถ้ายังไม่มี
       if 'close_time' not in df_ohlc.columns:
           df_ohlc['close_time'] = df_ohlc['timestamp'] + pd.Timedelta(milliseconds=target_interval_ms-1)
       
       return df_ohlc

# ฟังก์ชันสำหรับใช้งานเป็น command line tool
def main():
   """
   ฟังก์ชันหลักสำหรับใช้งานเป็น command line tool
   """
   import argparse
   from tqdm import tqdm
   
   parser = argparse.ArgumentParser(description="ดาวน์โหลดข้อมูลประวัติราคาจาก Binance")
   parser.add_argument("--symbol", type=str, required=True, help="คู่สกุลเงิน (เช่น BTCUSDT)")
   parser.add_argument("--timeframe", type=str, required=True, help="ไทม์เฟรม (เช่น 1m, 5m, 1h)")
   parser.add_argument("--start", type=str, required=True, help="วันเริ่มต้น (YYYY-MM-DD)")
   parser.add_argument("--end", type=str, default=None, help="วันสิ้นสุด (YYYY-MM-DD, ค่าเริ่มต้น = วันปัจจุบัน)")
   parser.add_argument("--output", type=str, default="data/raw", help="ไดเรกทอรีที่จะบันทึกข้อมูล")
   parser.add_argument("--format", type=str, default="both", choices=["csv", "parquet", "both"], help="รูปแบบไฟล์ที่จะบันทึก")
   parser.add_argument("--multiple", action="store_true", help="ดาวน์โหลดหลายไทม์เฟรม (ใช้คอมม่าคั่น)")
   parser.add_argument("--update", action="store_true", help="อัพเดตข้อมูลที่มีอยู่แล้ว")
   parser.add_argument("--api-key", type=str, default=None, help="Binance API key")
   parser.add_argument("--api-secret", type=str, default=None, help="Binance API secret")
   
   args = parser.parse_args()
   
   # สร้าง Binance Downloader
   downloader = BinanceDownloader(api_key=args.api_key, api_secret=args.api_secret)
   
   if args.multiple:
       # แยกไทม์เฟรมด้วยคอมม่า
       timeframes = [tf.strip() for tf in args.timeframe.split(',')]
       
       if args.update:
           # อัพเดตข้อมูลหลายไทม์เฟรม
           results = downloader.update_multiple_timeframes(
               symbol=args.symbol,
               timeframes=timeframes,
               data_dir=args.output,
               file_format=args.format
           )
       else:
           # ดาวน์โหลดข้อมูลหลายไทม์เฟรม
           results = downloader.download_multiple_timeframes(
               symbol=args.symbol,
               timeframes=timeframes,
               start_date=args.start,
               end_date=args.end,
               output_dir=args.output,
               file_format=args.format
           )
   else:
       if args.update:
           # อัพเดตข้อมูล
           df = downloader.update_data(
               symbol=args.symbol,
               timeframe=args.timeframe,
               data_dir=args.output,
               file_format=args.format
           )
       else:
           # ดาวน์โหลดข้อมูล
           df = downloader.download_historical_data(
               symbol=args.symbol,
               timeframe=args.timeframe,
               start_date=args.start,
               end_date=args.end,
               output_dir=args.output,
               file_format=args.format
           )

if __name__ == "__main__":
   # ทำให้สามารถใช้งานจาก command line ได้
   from tqdm import tqdm
   main()