import os
import numpy as np
import pandas as pd
import torch
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# นำเข้าการตั้งค่าจาก config.py
from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
   handler = logging.StreamHandler()
   handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
   logger.addHandler(handler)

class DataProcessor:
   """
   ประมวลผลข้อมูลสำหรับ Reinforcement Learning
   
   คลาสนี้ใช้สำหรับประมวลผลข้อมูลตลาดดิบให้เป็นรูปแบบที่เหมาะสมสำหรับการฝึกโมเดล RL
   รองรับการทำ Log Transform, Z-score Normalization, การจัดการค่าที่หายไป และการสร้างคุณลักษณะ (feature engineering)
   """
   
   def __init__(self, config=None):
       """
       กำหนดค่าเริ่มต้นสำหรับ DataProcessor
       
       Parameters:
       config (Config, optional): อ็อบเจ็กต์การตั้งค่า หรือ None เพื่อโหลดตั้งค่าเริ่มต้น
       """
       # โหลดการตั้งค่า
       self.config = config if config is not None else get_config()
       
       # ดึงการตั้งค่าที่เกี่ยวข้อง
       preprocessing_config = self.config.extract_subconfig("preprocessing")
       data_config = self.config.extract_subconfig("data")
       cuda_config = self.config.extract_subconfig("cuda")
       
       # ตัวเลือกการประมวลผล
       self.use_log_transform = preprocessing_config.get("use_log_transform", True)
       self.use_z_score = preprocessing_config.get("use_z_score", True)
       self.handle_missing = preprocessing_config.get("handle_missing_values", True)
       self.remove_outliers = preprocessing_config.get("remove_outliers", False)
       self.outlier_std_threshold = preprocessing_config.get("outlier_std_threshold", 3.0)
       self.fill_missing_strategy = preprocessing_config.get("fill_missing_strategy", "ffill")
       
       # ขนาดของชุดข้อมูล
       self.batch_size = data_config.get("batch_size", 1024)
       self.window_size = data_config.get("window_size", 60)
       
       # การตั้งค่า GPU
       self.use_gpu = cuda_config.get("use_cuda", True) and torch.cuda.is_available()
       self.device = f"cuda:{cuda_config.get('device', 0)}" if self.use_gpu else "cpu"
       self.precision = getattr(torch, cuda_config.get("precision", "float32"))
       
       # ข้อมูลทางสถิติสำหรับการทำ normalization
       self.stats = {}
       
       logger.info(f"กำลังใช้อุปกรณ์: {self.device}")
       logger.info(f"Log Transform: {self.use_log_transform}, Z-score: {self.use_z_score}")
   
   def process_file(
       self,
       input_file: str,
       output_file: Optional[str] = None,
       additional_indicators: Optional[List[str]] = None
   ) -> pd.DataFrame:
       """
       ประมวลผลไฟล์ข้อมูลตลาดแบบครบวงจร
       
       Parameters:
       input_file (str): พาธไปยังไฟล์ข้อมูลนำเข้า (CSV หรือ Parquet)
       output_file (str, optional): พาธไปยังไฟล์ข้อมูลส่งออก (Parquet)
       additional_indicators (List[str], optional): รายการตัวชี้วัดเพิ่มเติมที่ต้องการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่ผ่านการประมวลผลแล้ว
       """
       logger.info(f"กำลังประมวลผลไฟล์: {input_file}")
       
       # โหลดข้อมูล
       df = self.load_data(input_file)
       
       if df.empty:
           logger.error(f"ไม่สามารถโหลดข้อมูลจาก {input_file} หรือข้อมูลว่างเปล่า")
           return df
       
       # จัดการกับค่าที่หายไป
       if self.handle_missing:
           df = self.handle_missing_values(df)
       
       # คำนวณตัวชี้วัดเพิ่มเติม
       if additional_indicators:
           df = self.calculate_indicators(df, additional_indicators)
       
       # กำจัด outliers
       if self.remove_outliers:
           df = self.remove_outliers_from_df(df)
       
       # ประมวลผลข้อมูล
       processed_df = self.preprocess_data(df)
       
       # บันทึกไฟล์
       if output_file:
           self.save_processed_data(processed_df, output_file)
           self.save_stats(os.path.splitext(output_file)[0] + "_stats.json")
       
       return processed_df
   
   def load_data(self, file_path: str) -> pd.DataFrame:
       """
       โหลดข้อมูลจากไฟล์
       
       Parameters:
       file_path (str): พาธไปยังไฟล์ข้อมูล (CSV หรือ Parquet)
       
       Returns:
       pd.DataFrame: DataFrame ที่โหลดแล้ว
       """
       try:
           # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
           if not os.path.exists(file_path):
               logger.error(f"ไม่พบไฟล์: {file_path}")
               return pd.DataFrame()
           
           # โหลดข้อมูลตามนามสกุลไฟล์
           if file_path.endswith('.csv'):
               df = pd.read_csv(file_path)
           elif file_path.endswith('.parquet'):
               df = pd.read_parquet(file_path)
           else:
               logger.error(f"นามสกุลไฟล์ไม่รองรับ: {file_path}")
               return pd.DataFrame()
           
           # ตรวจสอบคอลัมน์ที่จำเป็น
           required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
           missing_columns = [col for col in required_columns if col not in df.columns]
           
           if missing_columns:
               logger.error(f"ไฟล์ขาดคอลัมน์ที่จำเป็น: {missing_columns}")
               return pd.DataFrame()
           
           # แปลง timestamp เป็น datetime
           if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
               if isinstance(df['timestamp'].iloc[0], (int, np.int64)):
                   # ถ้าเป็น UNIX timestamp (มิลลิวินาที)
                   df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
               else:
                   # ถ้าเป็นสตริง
                   df['timestamp'] = pd.to_datetime(df['timestamp'])
           
           # เรียงลำดับตาม timestamp
           df = df.sort_values('timestamp').reset_index(drop=True)
           
           logger.info(f"โหลดข้อมูลสำเร็จ: {len(df)} แถว, {df.columns.tolist()}")
           return df
           
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
           return pd.DataFrame()
   
   def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       จัดการกับค่าที่หายไปในข้อมูล
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ต้องการจัดการค่าที่หายไป
       
       Returns:
       pd.DataFrame: DataFrame ที่จัดการค่าที่หายไปแล้ว
       """
       if df.empty:
           return df
       
       # ตรวจสอบว่ามีค่าที่หายไปหรือไม่
       missing_values = df.isnull().sum()
       
       if missing_values.sum() > 0:
           logger.info(f"พบค่าที่หายไป: {missing_values[missing_values > 0].to_dict()}")
           
           # เติมค่าที่หายไปตามกลยุทธ์ที่กำหนด
           if self.fill_missing_strategy == "ffill":
               # เติมด้วยค่าก่อนหน้า
               df = df.ffill()
               # ถ้ายังมีค่าที่หายไป (เช่น ค่าแรกๆ) ให้เติมด้วยค่าถัดไป
               df = df.bfill()
           elif self.fill_missing_strategy == "bfill":
               # เติมด้วยค่าถัดไป
               df = df.bfill()
               # ถ้ายังมีค่าที่หายไป (เช่น ค่าสุดท้าย) ให้เติมด้วยค่าก่อนหน้า
               df = df.ffill()
           elif self.fill_missing_strategy == "zero":
               # เติมด้วยศูนย์
               df = df.fillna(0)
           elif self.fill_missing_strategy == "mean":
               # เติมด้วยค่าเฉลี่ย
               for col in df.columns:
                   if col != 'timestamp' and df[col].dtype in [np.float64, np.int64]:
                       df[col] = df[col].fillna(df[col].mean())
               # สำหรับคอลัมน์ที่ไม่ใช่ตัวเลขหรือยังมีค่าที่หายไป ให้เติมด้วยค่าก่อนหน้า
               df = df.ffill().bfill()
           else:
               # เติมด้วยค่าก่อนหน้าเป็นค่าเริ่มต้น
               df = df.ffill().bfill()
           
           # ตรวจสอบว่ายังมีค่าที่หายไปหรือไม่
           missing_after = df.isnull().sum().sum()
           if missing_after > 0:
               logger.warning(f"ยังมีค่าที่หายไปหลังจากเติมแล้ว: {missing_after} ค่า")
           else:
               logger.info("เติมค่าที่หายไปสำเร็จ")
       
       return df
   
   def remove_outliers_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       กำจัด outliers จาก DataFrame โดยใช้ Z-score
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ต้องการกำจัด outliers
       
       Returns:
       pd.DataFrame: DataFrame ที่กำจัด outliers แล้ว
       """
       if df.empty:
           return df
       
       # คัดลอก DataFrame
       df_clean = df.copy()
       outlier_count = 0
       
       # ตรวจสอบ outliers ในคอลัมน์ตัวเลขเท่านั้น
       numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
       
       # ไม่รวมคอลัมน์ timestamp และคอลัมน์ดัชนี
       exclude_columns = ['timestamp', 'date', 'time', 'index']
       numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
       
       for col in numeric_columns:
           # คำนวณ Z-score
           mean = df[col].mean()
           std = df[col].std()
           
           if std == 0:
               continue  # ข้ามคอลัมน์ที่มีค่าเบี่ยงเบนมาตรฐานเป็น 0
           
           z_scores = (df[col] - mean) / std
           
           # หา outliers
           outliers = (z_scores.abs() > self.outlier_std_threshold)
           outlier_count += outliers.sum()
           
           if outliers.sum() > 0:
               logger.info(f"พบ outliers ในคอลัมน์ {col}: {outliers.sum()} ค่า")
               
               # แทนที่ outliers ด้วยค่าเฉลี่ย
               df_clean.loc[outliers, col] = mean
       
       if outlier_count > 0:
           logger.info(f"กำจัด outliers ทั้งหมด {outlier_count} ค่า")
       else:
           logger.info("ไม่พบ outliers")
       
       return df_clean
   
   def calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
       """
       คำนวณตัวชี้วัดทางเทคนิคเพิ่มเติม
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ต้องการคำนวณตัวชี้วัด
       indicators (List[str]): รายการตัวชี้วัดที่ต้องการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีตัวชี้วัดเพิ่มเติม
       """
       if df.empty:
           return df
       
       # คัดลอก DataFrame
       result_df = df.copy()
       
       # ดึงการตั้งค่าตัวชี้วัด
       indicator_config = self.config.extract_subconfig("indicators")
       
       # คำนวณตัวชี้วัดตามที่ระบุ
       for indicator in indicators:
           if indicator.lower() == "rsi":
               # Relative Strength Index
               period = indicator_config.get("rsi_period", 14)
               result_df[f'rsi_{period}'] = self._calculate_rsi(result_df['close'], period)
               logger.info(f"คำนวณ RSI (period={period})")
           
           elif indicator.lower() == "macd":
               # Moving Average Convergence Divergence
               fast_period = indicator_config.get("macd_fast_period", 12)
               slow_period = indicator_config.get("macd_slow_period", 26)
               signal_period = indicator_config.get("macd_signal_period", 9)
               
               macd, signal, hist = self._calculate_macd(
                   result_df['close'], 
                   fast_period, 
                   slow_period, 
                   signal_period
               )
               
               result_df[f'macd_{fast_period}_{slow_period}'] = macd
               result_df[f'macd_signal_{signal_period}'] = signal
               result_df[f'macd_hist'] = hist
               
               logger.info(f"คำนวณ MACD (fast={fast_period}, slow={slow_period}, signal={signal_period})")
           
           elif indicator.lower() == "bollinger_bands":
               # Bollinger Bands
               period = indicator_config.get("bollinger_period", 20)
               std_dev = indicator_config.get("bollinger_std", 2.0)
               
               upper, middle, lower = self._calculate_bollinger_bands(
                   result_df['close'], 
                   period, 
                   std_dev
               )
               
               result_df[f'bb_upper_{period}'] = upper
               result_df[f'bb_middle_{period}'] = middle
               result_df[f'bb_lower_{period}'] = lower
               
               logger.info(f"คำนวณ Bollinger Bands (period={period}, std={std_dev})")
           
           elif indicator.lower() == "ema":
               # Exponential Moving Average
               periods = indicator_config.get("ema_periods", [9, 21, 50, 200])
               
               for period in periods:
                   result_df[f'ema_{period}'] = self._calculate_ema(result_df['close'], period)
               
               logger.info(f"คำนวณ EMA สำหรับช่วงเวลา: {periods}")
           
           elif indicator.lower() == "sma":
               # Simple Moving Average
               periods = indicator_config.get("sma_periods", [10, 50, 200])
               
               for period in periods:
                   result_df[f'sma_{period}'] = self._calculate_sma(result_df['close'], period)
               
               logger.info(f"คำนวณ SMA สำหรับช่วงเวลา: {periods}")
           
           elif indicator.lower() == "atr":
               # Average True Range
               period = indicator_config.get("atr_period", 14)
               result_df[f'atr_{period}'] = self._calculate_atr(
                   result_df['high'], 
                   result_df['low'], 
                   result_df['close'], 
                   period
               )
               
               logger.info(f"คำนวณ ATR (period={period})")
           
           elif indicator.lower() == "relative_volume":
               # Relative Volume
               period = indicator_config.get("relative_volume_period", 10)
               result_df['relative_volume'] = self._calculate_relative_volume(
                   result_df,
                   period
               )
               
               logger.info(f"คำนวณ Relative Volume (period={period})")
           
           else:
               logger.warning(f"ไม่รองรับตัวชี้วัด: {indicator}")
       
       return result_df
   
   def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
       """
       คำนวณ Relative Strength Index (RSI)
       
       Parameters:
       prices (pd.Series): อนุกรมราคา
       period (int): ช่วงเวลาสำหรับ RSI
       
       Returns:
       pd.Series: ค่า RSI
       """
       # คำนวณการเปลี่ยนแปลงราคา
       delta = prices.diff()
       
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
       
       return rsi
   
   def _calculate_macd(
       self, 
       prices: pd.Series, 
       fast_period: int = 12, 
       slow_period: int = 26, 
       signal_period: int = 9
   ) -> Tuple[pd.Series, pd.Series, pd.Series]:
       """
       คำนวณ Moving Average Convergence Divergence (MACD)
       
       Parameters:
       prices (pd.Series): อนุกรมราคา
       fast_period (int): ช่วงเวลาสำหรับ EMA เร็ว
       slow_period (int): ช่วงเวลาสำหรับ EMA ช้า
       signal_period (int): ช่วงเวลาสำหรับเส้นสัญญาณ
       
       Returns:
       Tuple[pd.Series, pd.Series, pd.Series]: (MACD, Signal, Histogram)
       """
       # คำนวณ EMA เร็วและช้า
       ema_fast = self._calculate_ema(prices, fast_period)
       ema_slow = self._calculate_ema(prices, slow_period)
       
       # คำนวณ MACD
       macd = ema_fast - ema_slow
       
       # คำนวณเส้นสัญญาณ
       signal = self._calculate_ema(macd, signal_period)
       
       # คำนวณฮิสโตแกรม
       histogram = macd - signal
       
       return macd, signal, histogram
   
   def _calculate_bollinger_bands(
       self, 
       prices: pd.Series, 
       period: int = 20, 
       std_dev: float = 2.0
   ) -> Tuple[pd.Series, pd.Series, pd.Series]:
       """
       คำนวณ Bollinger Bands
       
       Parameters:
       prices (pd.Series): อนุกรมราคา
       period (int): ช่วงเวลาสำหรับค่าเฉลี่ย
       std_dev (float): จำนวนส่วนเบี่ยงเบนมาตรฐาน
       
       Returns:
       Tuple[pd.Series, pd.Series, pd.Series]: (Upper Band, Middle Band, Lower Band)
       """
       # คำนวณเส้นกลาง (SMA)
       middle_band = self._calculate_sma(prices, period)
       
       # คำนวณส่วนเบี่ยงเบนมาตรฐาน
       rolling_std = prices.rolling(window=period).std()
       
       # คำนวณแถบบนและล่าง
       upper_band = middle_band + (rolling_std * std_dev)
       lower_band = middle_band - (rolling_std * std_dev)
       
       return upper_band, middle_band, lower_band
   
   def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
       """
       คำนวณ Exponential Moving Average (EMA)
       
       Parameters:
       prices (pd.Series): อนุกรมราคา
       period (int): ช่วงเวลาสำหรับ EMA
       
       Returns:
       pd.Series: ค่า EMA
       """
       return prices.ewm(span=period, adjust=False).mean()
   
   def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
       """
       คำนวณ Simple Moving Average (SMA)
       
       Parameters:
       prices (pd.Series): อนุกรมราคา
       period (int): ช่วงเวลาสำหรับ SMA
       
       Returns:
       pd.Series: ค่า SMA
       """
       return prices.rolling(window=period).mean()
   
   def _calculate_atr(
       self, 
       high: pd.Series, 
       low: pd.Series, 
       close: pd.Series, 
       period: int = 14
   ) -> pd.Series:
       """
       คำนวณ Average True Range (ATR)
       
       Parameters:
       high (pd.Series): อนุกรมราคาสูงสุด
       low (pd.Series): อนุกรมราคาต่ำสุด
       close (pd.Series): อนุกรมราคาปิด
       period (int): ช่วงเวลาสำหรับ ATR
       
       Returns:
       pd.Series: ค่า ATR
       """
       # คำนวณช่วงจริง (True Range)
       prev_close = close.shift(1)
       tr1 = high - low
       tr2 = (high - prev_close).abs()
       tr3 = (low - prev_close).abs()
       
       # หาค่าสูงสุดของ tr1, tr2 และ tr3
       true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
       
       # คำนวณ ATR โดยใช้ SMA
       atr = true_range.rolling(window=period).mean()
       
       return atr
   
   def _calculate_relative_volume(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
       """
       คำนวณ Relative Volume
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ volume และ timestamp
       period (int): ช่วงเวลาสำหรับคำนวณค่าเฉลี่ย
       
       Returns:
       pd.Series: ค่า Relative Volume
       """
       # สร้างคอลัมน์วันที่จาก timestamp
       if 'date' not in df.columns:
           df['date'] = df['timestamp'].dt.date
       
       # สร้างคอลัมน์เวลาภายในวัน
       if 'time' not in df.columns:
           df['time'] = df['timestamp'].dt.time
       
       # สร้าง DataFrame เพื่อเก็บค่า relative volume
       rel_vol = pd.Series(index=df.index, dtype=float)
       
       # วนลูปผ่านแต่ละแถวใน DataFrame
       for idx, row in df.iterrows():
           # หาข้อมูลในวันก่อนหน้า period วัน ที่เวลาเดียวกัน
           current_time = row['time']
           current_date = row['date']
           
           # หาวันที่ย้อนหลังตามจำนวน period
           past_dates = [current_date - timedelta(days=i+1) for i in range(period)]
           
           # ดึงข้อมูลปริมาณการซื้อขายในวันและเวลาที่คล้ายกัน
           past_volumes = []
           
           for past_date in past_dates:
               # หาข้อมูลของวันนั้น
               day_data = df[df['date'] == past_date]
               
               if not day_data.empty:
                   # หาข้อมูลที่มีเวลาใกล้เคียงกับเวลาปัจจุบัน
                   time_str = current_time.strftime('%H:%M:%S')
                   closest_idx = day_data['time'].astype(str).apply(
                       lambda x: abs((datetime.strptime(x, '%H:%M:%S') - 
                                  datetime.strptime(time_str, '%H:%M:%S')).total_seconds())
                   ).idxmin()
                   
                   if closest_idx in day_data.index:
                       past_volumes.append(day_data.loc[closest_idx, 'volume'])
           
           # คำนวณค่าเฉลี่ยของปริมาณการซื้อขายในอดีต
           if past_volumes:
               avg_volume = sum(past_volumes) / len(past_volumes)
               rel_vol[idx] = row['volume'] / avg_volume if avg_volume > 0 else 1.0
           else:
               rel_vol[idx] = 1.0  # ค่าเริ่มต้นถ้าไม่มีข้อมูลในอดีต
       
       return rel_vol
   
   def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       ประมวลผลข้อมูลด้วย Log Transform และ Z-score Normalization
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ต้องการประมวลผล
       
       Returns:
       pd.DataFrame: DataFrame ที่ผ่านการประมวลผลแล้ว
       """
       if df.empty:
           return df
       
       # คัดลอก DataFrame
       processed_df = df.copy()
       
       # แยกคอลัมน์ timestamp
       if 'timestamp' in processed_df.columns:
           timestamp = processed_df['timestamp']
           processed_df = processed_df.drop('timestamp', axis=1)
       
       # แยกคอลัมน์ตัวเลขและไม่ใช่ตัวเลข
       numeric_cols = processed_df.select_dtypes(include=np.number).columns
       non_numeric_cols = processed_df.select_dtypes(exclude=np.number).columns
       
       # ประมวลผลคอลัมน์ตัวเลข
       if len(numeric_cols) > 0:
           # จัดคอลัมน์ OHLCV ให้อยู่ตำแหน่งแรก
           ohlcv_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in numeric_cols]
           other_numeric_cols = [col for col in numeric_cols if col not in ohlcv_cols]
           
           # สร้าง mask สำหรับคอลัมน์ volume
           is_volume_col = np.zeros(len(ohlcv_cols), dtype=bool)
           if "volume" in ohlcv_cols:
               is_volume_col[ohlcv_cols.index("volume")] = True
           
           # ประมวลผลคอลัมน์ OHLCV
           if ohlcv_cols:
               ohlcv_data = processed_df[ohlcv_cols].values
               
               # ทำ Log Transform (ถ้าเลือกใช้)
               if self.use_log_transform:
                   ohlcv_data = self._log_transform(ohlcv_data, is_volume_col)
               
               # ทำ Z-score Normalization (ถ้าเลือกใช้)
               if self.use_z_score:
                   normalized_ohlcv, ohlcv_stats = self._z_score_normalize(ohlcv_data)
                   
                   # เก็บสถิติสำหรับการแปลงกลับ
                   self.stats["ohlcv"] = {
                       "means": ohlcv_stats["means"].tolist(),
                       "stds": ohlcv_stats["stds"].tolist(),
                       "columns": ohlcv_cols,
                       "is_volume_col": is_volume_col.tolist(),
                       "log_transform": self.use_log_transform
                   }
               else:
                   normalized_ohlcv = ohlcv_data
               
               # แทนที่ข้อมูลใน DataFrame
               for i, col in enumerate(ohlcv_cols):
                   processed_df[col] = normalized_ohlcv[:, i]
           
           # ประมวลผลคอลัมน์ตัวเลขอื่นๆ
           if other_numeric_cols:
               other_data = processed_df[other_numeric_cols].values
               
               # ทำ Z-score Normalization (ถ้าเลือกใช้)
               if self.use_z_score:
                   normalized_other, other_stats = self._z_score_normalize(other_data)
                   
                   # เก็บสถิติสำหรับการแปลงกลับ
                   self.stats["other"] = {
                       "means": other_stats["means"].tolist(),
                       "stds": other_stats["stds"].tolist(),
                       "columns": other_numeric_cols,
                       "log_transform": False
                   }
               else:
                   normalized_other = other_data
               
               # แทนที่ข้อมูลใน DataFrame
               for i, col in enumerate(other_numeric_cols):
                   processed_df[col] = normalized_other[:, i]
       
       # เพิ่มคอลัมน์ timestamp กลับเข้าไป
       if 'timestamp' in df.columns:
           processed_df['timestamp'] = timestamp
       
       return processed_df
   
   def _log_transform(self, data: np.ndarray, is_volume_col: np.ndarray) -> np.ndarray:
       """
       ทำ Log Transform กับข้อมูล
       
       Parameters:
       data (np.ndarray): ข้อมูลที่ต้องการแปลง
       is_volume_col (np.ndarray): Boolean array ระบุว่าคอลัมน์ไหนเป็นปริมาณ (ใช้ log1p)
       
       Returns:
       np.ndarray: ข้อมูลที่ผ่านการแปลงแล้ว
       """
       # สร้าง array ใหม่สำหรับเก็บข้อมูลหลัง log transform
       log_data = np.zeros_like(data, dtype=np.float64)
       
       # ทำ Log transform
       for col in range(data.shape[1]):
           # แทนค่าที่ <= 0 หรือ NaN ด้วยค่าเล็กๆ
           clean_data = np.copy(data[:, col])
           clean_data[clean_data <= 0] = 1e-8
           clean_data[np.isnan(clean_data)] = 1e-8
           
           if is_volume_col[col]:
               # สำหรับปริมาณ ใช้ log1p
               log_data[:, col] = np.log1p(clean_data)
           else:
               # สำหรับราคา ใช้ log
               log_data[:, col] = np.log(clean_data)
       
       return log_data
   
   def _z_score_normalize(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
       """
       ทำ Z-score Normalization กับข้อมูล
       
       Parameters:
       data (np.ndarray): ข้อมูลที่ต้องการทำ normalize
       
       Returns:
       Tuple[np.ndarray, Dict]: (ข้อมูลที่ผ่านการ normalize แล้ว, สถิติ)
       """
       # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
       means = np.mean(data, axis=0)
       stds = np.std(data, axis=0)
       
       # ป้องกันการหารด้วย 0
       stds[stds == 0] = 1.0
       
       # ทำ Z-score Normalization
       normalized_data = (data - means) / stds
       
       # เก็บสถิติสำหรับการแปลงกลับ
       stats = {
           "means": means,
           "stds": stds
       }
       
       return normalized_data, stats
   
   def inverse_transform(self, data: np.ndarray, stat_type: str = "ohlcv") -> np.ndarray:
       """
       แปลงข้อมูลกลับเป็นค่าดิบ
       
       Parameters:
       data (np.ndarray): ข้อมูลที่ต้องการแปลงกลับ
       stat_type (str): ประเภทของสถิติที่ใช้ ("ohlcv" หรือ "other")
       
       Returns:
       np.ndarray: ข้อมูลที่แปลงกลับแล้ว
       """
       if stat_type not in self.stats:
           logger.error(f"ไม่พบสถิติประเภท: {stat_type}")
           return data
       
       stats = self.stats[stat_type]
       
       # แปลงจาก list เป็น numpy array
       means = np.array(stats["means"])
       stds = np.array(stats["stds"])
       
       # แปลงกลับจาก Z-score
       if self.use_z_score:
           data = data * stds + means
       
       # แปลงกลับจาก Log transform
       if stats.get("log_transform", False):
           is_volume_col = np.array(stats.get("is_volume_col", [False] * data.shape[1]))
           
           for col in range(data.shape[1]):
               if col < len(is_volume_col) and is_volume_col[col]:
                   # สำหรับปริมาณ ใช้ expm1
                   data[:, col] = np.expm1(data[:, col])
               else:
                   # สำหรับราคา ใช้ exp
                   data[:, col] = np.exp(data[:, col])
       
       return data
   
   def inverse_transform_price(self, price: float, column: str = "close") -> float:
       """
       แปลงราคาเดี่ยวกลับเป็นค่าดิบ
       
       Parameters:
       price (float): ราคาที่ต้องการแปลงกลับ
       column (str): ชื่อคอลัมน์ราคา ("open", "high", "low", "close")
       
       Returns:
       float: ราคาที่แปลงกลับแล้ว
       """
       if "ohlcv" not in self.stats:
           logger.error("ไม่พบสถิติ OHLCV")
           return price
       
       stats = self.stats["ohlcv"]
       columns = stats["columns"]
       
       if column not in columns:
           logger.error(f"ไม่พบคอลัมน์: {column}")
           return price
       
       # หาดัชนีของคอลัมน์
       col_idx = columns.index(column)
       
       # แปลงค่าเดี่ยว
       value = price
       
       # แปลงกลับจาก Z-score
       if self.use_z_score:
           mean = stats["means"][col_idx]
           std = stats["stds"][col_idx]
           value = value * std + mean
       
       # แปลงกลับจาก Log transform
       if stats.get("log_transform", False):
           is_volume_col = stats.get("is_volume_col", [False] * len(columns))
           
           if col_idx < len(is_volume_col) and is_volume_col[col_idx]:
               # สำหรับปริมาณ ใช้ expm1
               value = math.expm1(value)
           else:
               # สำหรับราคา ใช้ exp
               value = math.exp(value)
       
       return value
   
   def save_processed_data(self, df: pd.DataFrame, output_file: str) -> None:
       """
       บันทึกข้อมูลที่ผ่านการประมวลผลแล้ว
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ต้องการบันทึก
       output_file (str): พาธไปยังไฟล์ส่งออก
       """
       try:
           # สร้างโฟลเดอร์หากไม่มี
           os.makedirs(os.path.dirname(output_file), exist_ok=True)
           
           # บันทึกไฟล์
           df.to_parquet(output_file)
           logger.info(f"บันทึกข้อมูลที่ผ่านการประมวลผลแล้วที่: {output_file}")
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
   
   def save_stats(self, output_file: str) -> None:
       """
       บันทึกสถิติสำหรับการแปลงกลับ
       
       Parameters:
       output_file (str): พาธไปยังไฟล์ส่งออก
       """
       try:
           # สร้างโฟลเดอร์หากไม่มี
           os.makedirs(os.path.dirname(output_file), exist_ok=True)
           
           # บันทึกสถิติเป็น JSON
           import json
           with open(output_file, 'w') as f:
               json.dump(self.stats, f, indent=2)
           
           logger.info(f"บันทึกสถิติที่: {output_file}")
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการบันทึกสถิติ: {e}")
   
   def load_stats(self, input_file: str) -> None:
       """
       โหลดสถิติสำหรับการแปลงกลับ
       
       Parameters:
       input_file (str): พาธไปยังไฟล์นำเข้า
       """
       try:
           # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
           if not os.path.exists(input_file):
               logger.error(f"ไม่พบไฟล์: {input_file}")
               return
           
           # โหลดสถิติจาก JSON
           import json
           with open(input_file, 'r') as f:
               self.stats = json.load(f)
           
           logger.info(f"โหลดสถิติจาก: {input_file}")
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการโหลดสถิติ: {e}")
   
   def process_directory(
       self,
       input_dir: str,
       output_dir: str,
       file_pattern: str = "*.csv",
       additional_indicators: Optional[List[str]] = None
   ) -> List[str]:
       """
       ประมวลผลไฟล์ทั้งหมดในไดเรกทอรี
       
       Parameters:
       input_dir (str): ไดเรกทอรีนำเข้า
       output_dir (str): ไดเรกทอรีส่งออก
       file_pattern (str): รูปแบบไฟล์ที่ต้องการประมวลผล
       additional_indicators (List[str], optional): รายการตัวชี้วัดเพิ่มเติมที่ต้องการคำนวณ
       
       Returns:
       List[str]: รายการไฟล์ที่ผ่านการประมวลผลแล้ว
       """
       import glob
       
       # ค้นหาไฟล์ทั้งหมดที่ตรงกับรูปแบบ
       input_files = glob.glob(os.path.join(input_dir, file_pattern))
       
       if not input_files:
           logger.warning(f"ไม่พบไฟล์ที่ตรงกับรูปแบบ: {file_pattern} ในไดเรกทอรี: {input_dir}")
           return []
       
       # สร้างไดเรกทอรีส่งออกหากไม่มี
       os.makedirs(output_dir, exist_ok=True)
       
       # ประมวลผลไฟล์ทีละไฟล์
       processed_files = []
       
       for input_file in tqdm(input_files, desc="Processing files"):
           # สร้างชื่อไฟล์ส่งออก
           file_name = os.path.basename(input_file)
           file_base = os.path.splitext(file_name)[0]
           output_file = os.path.join(output_dir, f"{file_base}_processed.parquet")
           
           # ประมวลผลไฟล์
           try:
               self.process_file(input_file, output_file, additional_indicators)
               processed_files.append(output_file)
           except Exception as e:
               logger.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {input_file}: {e}")
       
       logger.info(f"ประมวลผลไฟล์ทั้งหมด {len(processed_files)} ไฟล์")
       return processed_files
   
   def create_training_data(
       self,
       df: pd.DataFrame,
       window_size: Optional[int] = None,
       batch_size: Optional[int] = None,
       validation_ratio: float = 0.1,
       test_ratio: float = 0.1,
       shuffle: bool = True
   ) -> Dict[str, Any]:
       """
       สร้างชุดข้อมูลสำหรับการเทรนโมเดล
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่ผ่านการประมวลผลแล้ว
       window_size (int, optional): ขนาดของหน้าต่างสำหรับแต่ละตัวอย่าง
       batch_size (int, optional): ขนาดของแต่ละ batch
       validation_ratio (float): สัดส่วนของข้อมูลสำหรับ validation
       test_ratio (float): สัดส่วนของข้อมูลสำหรับ test
       shuffle (bool): สุ่มข้อมูลหรือไม่
       
       Returns:
       Dict[str, Any]: Dictionary ของชุดข้อมูล
       """
       # ใช้ค่าเริ่มต้นจากการตั้งค่าถ้าไม่ได้ระบุ
       if window_size is None:
           window_size = self.window_size
       
       if batch_size is None:
           batch_size = self.batch_size
       
       # แยกคอลัมน์ timestamp
       timestamp = None
       if 'timestamp' in df.columns:
           timestamp = df['timestamp'].values
           df = df.drop('timestamp', axis=1)
       
       # แปลงเป็น NumPy array
       data = df.values
       
       # สร้าง sliding windows
       windows = []
       
       for i in range(len(data) - window_size + 1):
           window = data[i:i+window_size]
           windows.append(window)
       
       # แปลงเป็น NumPy array
       windows = np.array(windows)
       
       # แบ่งข้อมูลเป็นชุด train, validation และ test
       n_samples = len(windows)
       indices = np.arange(n_samples)
       
       if shuffle:
           np.random.shuffle(indices)
       
       # คำนวณจำนวนตัวอย่างในแต่ละชุด
       test_size = int(test_ratio * n_samples)
       val_size = int(validation_ratio * n_samples)
       train_size = n_samples - test_size - val_size
       
       # แบ่งดัชนี
       train_indices = indices[:train_size]
       val_indices = indices[train_size:train_size+val_size]
       test_indices = indices[train_size+val_size:]
       
       # แบ่งข้อมูล
       train_data = windows[train_indices]
       val_data = windows[val_indices]
       test_data = windows[test_indices]
       
       # บันทึกช่วงเวลาที่ตรงกับแต่ละชุด
       train_timestamps = None
       val_timestamps = None
       test_timestamps = None
       
       if timestamp is not None:
           # สร้าง timestamp สำหรับแต่ละ window (ใช้ timestamp สุดท้ายของแต่ละ window)
           window_timestamps = np.array([timestamp[i+window_size-1] for i in range(len(data) - window_size + 1)])
           
           train_timestamps = window_timestamps[train_indices]
           val_timestamps = window_timestamps[val_indices]
           test_timestamps = window_timestamps[test_indices]
       
       # สร้าง Tensor และย้ายไปยัง GPU ถ้าจำเป็น
       if self.use_gpu:
           train_tensor = torch.tensor(train_data, dtype=self.precision).to(self.device)
           val_tensor = torch.tensor(val_data, dtype=self.precision).to(self.device)
           test_tensor = torch.tensor(test_data, dtype=self.precision).to(self.device)
       else:
           train_tensor = torch.tensor(train_data, dtype=self.precision)
           val_tensor = torch.tensor(val_data, dtype=self.precision)
           test_tensor = torch.tensor(test_data, dtype=self.precision)
       
       # สร้าง DataLoader
       from torch.utils.data import TensorDataset, DataLoader
       
       train_dataset = TensorDataset(train_tensor)
       val_dataset = TensorDataset(val_tensor)
       test_dataset = TensorDataset(test_tensor)
       
       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=batch_size)
       test_loader = DataLoader(test_dataset, batch_size=batch_size)
       
       return {
           "train_data": train_data,
           "val_data": val_data,
           "test_data": test_data,
           "train_loader": train_loader,
           "val_loader": val_loader,
           "test_loader": test_loader,
           "train_timestamps": train_timestamps,
           "val_timestamps": val_timestamps,
           "test_timestamps": test_timestamps,
           "window_size": window_size,
           "batch_size": batch_size,
           "feature_size": data.shape[1],
           "train_size": train_size,
           "val_size": val_size,
           "test_size": test_size
       }

# ฟังก์ชันสำหรับใช้งานเป็น command line tool
def main():
   """
   ฟังก์ชันหลักสำหรับใช้งานเป็น command line tool
   """
   import argparse
   
   parser = argparse.ArgumentParser(description="ประมวลผลข้อมูลตลาดสำหรับ Reinforcement Learning")
   parser.add_argument("--input", type=str, required=True, help="ไฟล์หรือไดเรกทอรีนำเข้า")
   parser.add_argument("--output", type=str, required=True, help="ไฟล์หรือไดเรกทอรีส่งออก")
   parser.add_argument("--window-size", type=int, help="ขนาดของหน้าต่างสำหรับแต่ละตัวอย่าง")
   parser.add_argument("--log-transform", action="store_true", help="ใช้ Log Transform")
   parser.add_argument("--no-log-transform", action="store_false", dest="log_transform", help="ไม่ใช้ Log Transform")
   parser.add_argument("--z-score", action="store_true", help="ใช้ Z-score Normalization")
   parser.add_argument("--no-z-score", action="store_false", dest="z_score", help="ไม่ใช้ Z-score Normalization")
   parser.add_argument("--handle-missing", action="store_true", help="จัดการกับค่าที่หายไป")
   parser.add_argument("--no-handle-missing", action="store_false", dest="handle_missing", help="ไม่จัดการกับค่าที่หายไป")
   parser.add_argument("--indicators", type=str, help="รายการตัวชี้วัดที่ต้องการคำนวณ (คั่นด้วยเครื่องหมายจุลภาค)")
   parser.add_argument("--file-pattern", type=str, default="*.csv", help="รูปแบบไฟล์ที่ต้องการประมวลผล (สำหรับโหมดไดเรกทอรี)")
   parser.add_argument("--cuda", action="store_true", help="ใช้ CUDA/GPU")
   parser.add_argument("--no-cuda", action="store_false", dest="cuda", help="ไม่ใช้ CUDA/GPU")
   
   # ตั้งค่า defaults จาก config
   config = get_config()
   preprocessing_config = config.extract_subconfig("preprocessing")
   data_config = config.extract_subconfig("data")
   cuda_config = config.extract_subconfig("cuda")
   
   parser.set_defaults(
       log_transform=preprocessing_config.get("use_log_transform", True),
       z_score=preprocessing_config.get("use_z_score", True),
       handle_missing=preprocessing_config.get("handle_missing_values", True),
       window_size=data_config.get("window_size", 60),
       cuda=cuda_config.get("use_cuda", True) and torch.cuda.is_available()
   )
   
   args = parser.parse_args()
   
   # สร้าง DataProcessor
   processor = DataProcessor()
   
   # กำหนดตัวเลือกตาม command line
   processor.use_log_transform = args.log_transform
   processor.use_z_score = args.z_score
   processor.handle_missing = args.handle_missing
   processor.window_size = args.window_size
   processor.use_gpu = args.cuda and torch.cuda.is_available()
   
   # แยกตัวชี้วัด
   indicators = None
   if args.indicators:
       indicators = [indicator.strip() for indicator in args.indicators.split(',')]
   
   # ตรวจสอบว่าเป็นไฟล์หรือไดเรกทอรี
   input_path = Path(args.input)
   output_path = Path(args.output)
   
   if input_path.is_file():
       # ประมวลผลไฟล์เดียว
       processor.process_file(str(input_path), str(output_path), indicators)
   elif input_path.is_dir():
       # ประมวลผลทั้งไดเรกทอรี
       processor.process_directory(str(input_path), str(output_path), args.file_pattern, indicators)
   else:
       logger.error(f"ไม่พบไฟล์หรือไดเรกทอรี: {args.input}")

if __name__ == "__main__":
   # ทำให้สามารถใช้งานจาก command line ได้
   main()