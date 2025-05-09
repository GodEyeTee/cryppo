import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import os

# นำเข้าการตั้งค่าจาก config.py
from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
   handler = logging.StreamHandler()
   handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
   logger.addHandler(handler)

class TechnicalIndicators:
   """
   คลาสสำหรับคำนวณตัวชี้วัดทางเทคนิคต่างๆ
   
   รองรับตัวชี้วัดพื้นฐานและตัวชี้วัดขั้นสูงหลายประเภท
   สามารถเรียกใช้แบบเป็น singleton หรือสร้างอินสแตนซ์ใหม่ก็ได้
   """
   
   # singleton instance
   _instance = None
   
   @classmethod
   def get_instance(cls):
       """
       ดึง singleton instance ของ TechnicalIndicators
       
       Returns:
       TechnicalIndicators: instance ของ TechnicalIndicators
       """
       if cls._instance is None:
           cls._instance = cls()
       return cls._instance
   
   def __init__(self, config=None):
       """
       กำหนดค่าเริ่มต้นสำหรับ TechnicalIndicators
       
       Parameters:
       config (Config, optional): อ็อบเจ็กต์การตั้งค่า หรือ None เพื่อโหลดตั้งค่าเริ่มต้น
       """
       # โหลดการตั้งค่า
       self.config = config if config is not None else get_config()
       
       # ดึงการตั้งค่าที่เกี่ยวข้อง
       self.indicator_config = self.config.extract_subconfig("indicators")
       
       # ตัวแปรเก็บฟังก์ชันสำหรับคำนวณตัวชี้วัดแต่ละประเภท
       self.indicator_functions = {
           "rsi": self.rsi,
           "macd": self.macd,
           "bollinger_bands": self.bollinger_bands,
           "ema": self.ema,
           "sma": self.sma,
           "atr": self.atr,
           "relative_volume": self.relative_volume,
           "stochastic": self.stochastic,
           "obv": self.obv,
           "vwap": self.vwap,
           "fibonacci_retracement": self.fibonacci_retracement,
           "ichimoku_cloud": self.ichimoku_cloud,
           "parabolic_sar": self.parabolic_sar,
           "adx": self.adx,
           "volume_profile": self.volume_profile,
           "moving_average_crossover": self.moving_average_crossover,
           "pivot_points": self.pivot_points,
           "chaikin_money_flow": self.chaikin_money_flow,
           "momentum": self.momentum,
           "hull_moving_average": self.hull_moving_average,
           "keltner_channels": self.keltner_channels,
           "price_rate_of_change": self.price_rate_of_change,
           "bears_vs_bulls": self.bears_vs_bulls,
           "williams_r": self.williams_r,
           "standard_deviation": self.standard_deviation,
           "trix": self.trix,
           "average_directional_index": self.average_directional_index,
           "money_flow_index": self.money_flow_index,
           "accumulation_distribution_line": self.accumulation_distribution_line,
           "time_segmented_volume": self.time_segmented_volume
       }
   
   def calculate_indicators(
       self, 
       df: pd.DataFrame, 
       indicators: List[str] = None
   ) -> pd.DataFrame:
       """
       คำนวณตัวชี้วัดที่ระบุในรายการ
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีข้อมูล OHLCV
       indicators (List[str], optional): รายการตัวชี้วัดที่ต้องการคำนวณ
                 ถ้าไม่ระบุจะใช้รายการตัวชี้วัดเริ่มต้นจากการตั้งค่า
       
       Returns:
       pd.DataFrame: DataFrame ที่มีตัวชี้วัดเพิ่มเติม
       """
       if df.empty:
           return df
       
       # ตรวจสอบคอลัมน์ที่จำเป็น
       required_columns = ["open", "high", "low", "close", "volume"]
       missing_columns = [col for col in required_columns if col not in df.columns]
       
       if missing_columns:
           logger.error(f"DataFrame ขาดคอลัมน์ที่จำเป็น: {missing_columns}")
           return df
       
       # ถ้าไม่ระบุรายการตัวชี้วัด ให้ใช้รายการเริ่มต้นจากการตั้งค่า
       if indicators is None:
           indicators = self.indicator_config.get("default_indicators", [])
       
       # คัดลอก DataFrame
       result_df = df.copy()
       
       # คำนวณตัวชี้วัดตามที่ระบุ
       for indicator in indicators:
           indicator_name = indicator.lower()
           
           if indicator_name in self.indicator_functions:
               try:
                   # เรียกใช้ฟังก์ชันคำนวณตัวชี้วัด
                   result = self.indicator_functions[indicator_name](result_df)
                   
                   # ถ้าผลลัพธ์เป็น DataFrame ให้รวมกับ result_df
                   if isinstance(result, pd.DataFrame):
                       # ลบคอลัมน์ที่ซ้ำกัน
                       duplicate_columns = [col for col in result.columns if col in result_df.columns]
                       if duplicate_columns:
                           result = result.drop(columns=duplicate_columns)
                       
                       # รวม DataFrame
                       result_df = pd.concat([result_df, result], axis=1)
                   else:
                       logger.warning(f"ฟังก์ชันคำนวณตัวชี้วัด {indicator_name} ไม่คืนค่าเป็น DataFrame")
                   
                   logger.info(f"คำนวณตัวชี้วัด: {indicator_name}")
               except Exception as e:
                   logger.error(f"เกิดข้อผิดพลาดในการคำนวณตัวชี้วัด {indicator_name}: {e}")
           else:
               logger.warning(f"ไม่รองรับตัวชี้วัด: {indicator_name}")
       
       return result_df
   
   #region พื้นฐาน
   
   def rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
       """
       คำนวณ Relative Strength Index (RSI)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int, optional): ช่วงเวลาสำหรับ RSI
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ RSI
       """
       # ถ้าไม่ระบุ period ให้ใช้ค่าจากการตั้งค่า
       if period is None:
           period = self.indicator_config.get("rsi_period", 14)
       
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
   
   def macd(
       self, 
       df: pd.DataFrame, 
       fast_period: int = None, 
       slow_period: int = None, 
       signal_period: int = None
   ) -> pd.DataFrame:
       """
       คำนวณ Moving Average Convergence Divergence (MACD)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       fast_period (int, optional): ช่วงเวลาสำหรับ EMA เร็ว
       slow_period (int, optional): ช่วงเวลาสำหรับ EMA ช้า
       signal_period (int, optional): ช่วงเวลาสำหรับเส้นสัญญาณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ MACD, MACD Signal และ MACD Histogram
       """
       # ถ้าไม่ระบุ period ให้ใช้ค่าจากการตั้งค่า
       if fast_period is None:
           fast_period = self.indicator_config.get("macd_fast_period", 12)
       
       if slow_period is None:
           slow_period = self.indicator_config.get("macd_slow_period", 26)
       
       if signal_period is None:
           signal_period = self.indicator_config.get("macd_signal_period", 9)
       
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
   
   def bollinger_bands(
       self, 
       df: pd.DataFrame, 
       period: int = None, 
       std_dev: float = None
   ) -> pd.DataFrame:
       """
       คำนวณ Bollinger Bands
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int, optional): ช่วงเวลาสำหรับค่าเฉลี่ย
       std_dev (float, optional): จำนวนส่วนเบี่ยงเบนมาตรฐาน
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Upper Band, Middle Band และ Lower Band
       """
       # ถ้าไม่ระบุ period หรือ std_dev ให้ใช้ค่าจากการตั้งค่า
       if period is None:
           period = self.indicator_config.get("bollinger_period", 20)
       
       if std_dev is None:
           std_dev = self.indicator_config.get("bollinger_std", 2.0)
       
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
   
   def ema(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
       """
       คำนวณ Exponential Moving Average (EMA)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       periods (List[int], optional): รายการช่วงเวลาสำหรับ EMA
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ EMA สำหรับแต่ละช่วงเวลา
       """
       # ถ้าไม่ระบุ periods ให้ใช้ค่าจากการตั้งค่า
       if periods is None:
           periods = self.indicator_config.get("ema_periods", [9, 21, 50, 200])
       
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ EMA สำหรับแต่ละช่วงเวลา
       for period in periods:
           result[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
       
       return result
   
   def sma(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
       """
       คำนวณ Simple Moving Average (SMA)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       periods (List[int], optional): รายการช่วงเวลาสำหรับ SMA
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ SMA สำหรับแต่ละช่วงเวลา
       """
       # ถ้าไม่ระบุ periods ให้ใช้ค่าจากการตั้งค่า
       if periods is None:
           periods = self.indicator_config.get("sma_periods", [10, 50, 200])
       
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ SMA สำหรับแต่ละช่วงเวลา
       for period in periods:
           result[f'sma_{period}'] = df['close'].rolling(window=period).mean()
       
       return result
   
   def atr(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
       """
       คำนวณ Average True Range (ATR)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       period (int, optional): ช่วงเวลาสำหรับ ATR
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ ATR
       """
       # ถ้าไม่ระบุ period ให้ใช้ค่าจากการตั้งค่า
       if period is None:
           period = self.indicator_config.get("atr_period", 14)
       
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
   
   def relative_volume(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
       """
       คำนวณ Relative Volume
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ volume และ timestamp
       period (int, optional): ช่วงเวลาสำหรับคำนวณค่าเฉลี่ย
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Relative Volume
       """
       # ถ้าไม่ระบุ period ให้ใช้ค่าจากการตั้งค่า
       if period is None:
           period = self.indicator_config.get("relative_volume_period", 10)
       
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
       if 'timestamp' not in df.columns:
           logger.error("DataFrame ไม่มีคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการคำนวณ Relative Volume")
           result['relative_volume'] = np.nan
           return result
       
       # สร้างคอลัมน์วันที่และเวลาจาก timestamp
       df_with_datetime = df.copy()
       
       if pd.api.types.is_datetime64_any_dtype(df_with_datetime['timestamp']):
           df_with_datetime['date'] = df_with_datetime['timestamp'].dt.date
           df_with_datetime['time'] = df_with_datetime['timestamp'].dt.time
       else:
           try:
               # แปลง timestamp เป็น datetime
               df_with_datetime['timestamp'] = pd.to_datetime(df_with_datetime['timestamp'])
               df_with_datetime['date'] = df_with_datetime['timestamp'].dt.date
               df_with_datetime['time'] = df_with_datetime['timestamp'].dt.time
           except:
               logger.error("ไม่สามารถแปลงคอลัมน์ timestamp เป็น datetime ได้")
               result['relative_volume'] = np.nan
               return result
       
       # คำนวณ Relative Volume
       relative_volume = np.zeros(len(df_with_datetime))
       
       for i, row in df_with_datetime.iterrows():
           # หาวันที่ย้อนหลังตามจำนวน period
           past_dates = [(row['date'] - timedelta(days=d)) for d in range(1, period + 1)]
           
           # ดึงข้อมูลในวันและเวลาที่คล้ายกัน
           similar_time_volumes = []
           
           for past_date in past_dates:
               # หาข้อมูลในวันนั้น
               past_day_data = df_with_datetime[df_with_datetime['date'] == past_date]
               
               if not past_day_data.empty:
                   # หาข้อมูลที่เวลาใกล้เคียง
                   row_time = row['time']
                   
                   # เวลาที่ใกล้ที่สุด
                   if len(past_day_data) > 0:
                       # แปลงเวลาเป็นวินาที
                       row_seconds = row_time.hour * 3600 + row_time.minute * 60 + row_time.second
                       
                       # คำนวณความต่างของเวลา
                       past_day_data['time_diff'] = past_day_data['time'].apply(
                           lambda t: abs((t.hour * 3600 + t.minute * 60 + t.second) - row_seconds)
                       )
                       
                       # ข้อมูลที่เวลาใกล้ที่สุด
                       closest_row = past_day_data.loc[past_day_data['time_diff'].idxmin()]
                       similar_time_volumes.append(closest_row['volume'])
           
           # คำนวณค่าเฉลี่ยของปริมาณการซื้อขายในวันและเวลาคล้ายกัน
           if similar_time_volumes:
               avg_volume = np.mean(similar_time_volumes)
               # ถ้าค่าเฉลี่ยไม่เป็น 0
               if avg_volume > 0:
                   relative_volume[i] = row['volume'] / avg_volume
               else:
                   relative_volume[i] = 1.0
           else:
               relative_volume[i] = 1.0
       
       # เพิ่มคอลัมน์ Relative Volume ลงใน DataFrame ผลลัพธ์
       result['relative_volume'] = relative_volume
       
       return result
   
   #endregion
   
   #region ตัวชี้วัดขั้นสูง
   
   def stochastic(
       self, 
       df: pd.DataFrame, 
       k_period: int = 14, 
       d_period: int = 3, 
       slowing: int = 3
   ) -> pd.DataFrame:
       """
       คำนวณ Stochastic Oscillator
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       k_period (int): ช่วงเวลาสำหรับ %K
       d_period (int): ช่วงเวลาสำหรับ %D
       slowing (int): ช่วงเวลาสำหรับการทำให้ช้าลง
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ %K และ %D
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
   
   def obv(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       คำนวณ On-Balance Volume (OBV)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close และ volume
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ OBV
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
   
   def vwap(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       คำนวณ Volume-Weighted Average Price (VWAP)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low, close และ volume
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ VWAP
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

   def fibonacci_retracement(self, df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
       """
       คำนวณระดับ Fibonacci Retracement จากข้อมูลราคาย้อนหลัง
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high และ low
       period (int): จำนวนแท่งเทียนสำหรับหาจุดสูงสุดและต่ำสุด
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ระดับ Fibonacci ต่างๆ
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # กำหนดสัดส่วน Fibonacci
       fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
       
       # คำนวณค่าสูงสุดและต่ำสุดในช่วงเวลาที่กำหนด
       for i in range(len(df)):
           if i < period:
               # ยังไม่มีข้อมูลพอ
               for ratio in fib_ratios:
                   result.loc[df.index[i], f'fib_{ratio:.3f}'] = np.nan
               continue
           
           # หาจุดสูงสุดและต่ำสุดในช่วง period แท่งล่าสุด
           lookback_period = period
           high_value = df['high'].iloc[i-lookback_period:i].max()
           low_value = df['low'].iloc[i-lookback_period:i].min()
           
           # ตรวจสอบทิศทางของตลาด (ขาขึ้นหรือขาลง)
           if df['close'].iloc[i] > df['close'].iloc[i-lookback_period]:
               # ขาขึ้น: คำนวณระดับ Fibonacci จากต่ำไปสูง
               diff = high_value - low_value
               
               for ratio in fib_ratios:
                   level = low_value + (diff * ratio)
                   result.loc[df.index[i], f'fib_{ratio:.3f}'] = level
           else:
               # ขาลง: คำนวณระดับ Fibonacci จากสูงไปต่ำ
               diff = high_value - low_value
               
               for ratio in fib_ratios:
                   level = high_value - (diff * ratio)
                   result.loc[df.index[i], f'fib_{ratio:.3f}'] = level
       
       return result
   
   def ichimoku_cloud(
       self, 
       df: pd.DataFrame, 
       tenkan_period: int = 9, 
       kijun_period: int = 26, 
       senkou_span_b_period: int = 52, 
       displacement: int = 26
   ) -> pd.DataFrame:
       """
       คำนวณ Ichimoku Cloud
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high และ low
       tenkan_period (int): ช่วงเวลาสำหรับ Tenkan-sen (Conversion Line)
       kijun_period (int): ช่วงเวลาสำหรับ Kijun-sen (Base Line)
       senkou_span_b_period (int): ช่วงเวลาสำหรับ Senkou Span B (Leading Span B)
       displacement (int): ช่วงเวลาสำหรับการเลื่อนเส้น (Displacement)
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Ichimoku Cloud
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ Tenkan-sen (Conversion Line)
       high_tenkan = df['high'].rolling(window=tenkan_period).max()
       low_tenkan = df['low'].rolling(window=tenkan_period).min()
       tenkan_sen = (high_tenkan + low_tenkan) / 2
       
       # คำนวณ Kijun-sen (Base Line)
       high_kijun = df['high'].rolling(window=kijun_period).max()
       low_kijun = df['low'].rolling(window=kijun_period).min()
       kijun_sen = (high_kijun + low_kijun) / 2
       
       # คำนวณ Senkou Span A (Leading Span A)
       senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
       
       # คำนวณ Senkou Span B (Leading Span B)
       high_senkou = df['high'].rolling(window=senkou_span_b_period).max()
       low_senkou = df['low'].rolling(window=senkou_span_b_period).min()
       senkou_span_b = ((high_senkou + low_senkou) / 2).shift(displacement)
       
       # คำนวณ Chikou Span (Lagging Span)
       chikou_span = df['close'].shift(-displacement)
       
       # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
       result['ichimoku_tenkan_sen'] = tenkan_sen
       result['ichimoku_kijun_sen'] = kijun_sen
       result['ichimoku_senkou_span_a'] = senkou_span_a
       result['ichimoku_senkou_span_b'] = senkou_span_b
       result['ichimoku_chikou_span'] = chikou_span
       
       return result
   
   def parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
       """
       คำนวณ Parabolic SAR (Stop and Reverse)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high และ low
       af_start (float): ค่าเริ่มต้นของ Acceleration Factor
       af_max (float): ค่าสูงสุดของ Acceleration Factor
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Parabolic SAR
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # ใช้วิธีการคำนวณแบบ manual แทนการใช้ library เพื่อให้เข้าใจได้ง่าย
       high = df['high'].values
       low = df['low'].values
       close = df['close'].values
       
       # สร้าง array สำหรับเก็บค่า SAR
       sar = np.zeros_like(close)
       
       # กำหนดค่าเริ่มต้น
       if len(high) < 2:
           result['psar'] = np.nan
           return result
       
       # กำหนดแนวโน้มเริ่มต้น (True = ขึ้น, False = ลง)
       trend = close[1] > close[0]
       
       # กำหนดค่า SAR เริ่มต้น
       sar[0] = low[0] if trend else high[0]
       
       # กำหนดค่า Extreme Point เริ่มต้น
       ep = high[1] if trend else low[1]
       
       # กำหนดค่า Acceleration Factor เริ่มต้น
       af = af_start
       
       # คำนวณ SAR สำหรับแต่ละวัน
       for i in range(1, len(close)):
           # ใช้ค่า SAR ก่อนหน้า
           sar[i] = sar[i-1]
           
           # ปรับค่า SAR ตามแนวโน้ม
           if trend:
               # แนวโน้มขึ้น
               sar[i] = sar[i-1] + af * (ep - sar[i-1])
               
               # ตรวจสอบว่า SAR อยู่ต่ำกว่าราคาต่ำสุด 2 วันล่าสุดหรือไม่
               if i >= 2:
                   sar[i] = min(sar[i], min(low[i-1], low[i-2]))
               
               # ตรวจสอบว่าแนวโน้มเปลี่ยนหรือไม่
               if low[i] < sar[i]:
                   # เปลี่ยนเป็นแนวโน้มลง
                   trend = False
                   sar[i] = ep
                   ep = low[i]
                   af = af_start
               else:
                   # ยังคงเป็นแนวโน้มขึ้น
                   if high[i] > ep:
                       # มีจุดสูงสุดใหม่
                       ep = high[i]
                       af = min(af + af_start, af_max)
           else:
               # แนวโน้มลง
               sar[i] = sar[i-1] - af * (sar[i-1] - ep)
               
               # ตรวจสอบว่า SAR อยู่สูงกว่าราคาสูงสุด 2 วันล่าสุดหรือไม่
               if i >= 2:
                   sar[i] = max(sar[i], max(high[i-1], high[i-2]))
               
               # ตรวจสอบว่าแนวโน้มเปลี่ยนหรือไม่
               if high[i] > sar[i]:
                   # เปลี่ยนเป็นแนวโน้มขึ้น
                   trend = True
                   sar[i] = ep
                   ep = high[i]
                   af = af_start
               else:
                   # ยังคงเป็นแนวโน้มลง
                   if low[i] < ep:
                       # มีจุดต่ำสุดใหม่
                       ep = low[i]
                       af = min(af + af_start, af_max)
       
       # เพิ่มคอลัมน์ SAR ลงใน DataFrame ผลลัพธ์
       result['psar'] = sar
       
       # เพิ่มคอลัมน์แนวโน้ม (1 = ขึ้น, -1 = ลง)
       result['psar_trend'] = np.where(df['close'] > result['psar'], 1, -1)
       
       return result
   
   def adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
       """
       คำนวณ Average Directional Index (ADX)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ ADX, +DI และ -DI
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณการเปลี่ยนแปลงของ high, low และ close
       high = df['high']
       low = df['low']
       close = df['close']
       
       # คำนวณ True Range (TR)
       prev_close = close.shift(1)
       tr1 = high - low
       tr2 = (high - prev_close).abs()
       tr3 = (low - prev_close).abs()
       true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
       
       # คำนวณ Directional Movement (DM)
       up_move = high - high.shift(1)
       down_move = low.shift(1) - low
       
       # คำนวณ +DM และ -DM
       plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
       minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
       
       # แปลงเป็น Series
       plus_dm = pd.Series(plus_dm, index=df.index)
       minus_dm = pd.Series(minus_dm, index=df.index)
       
       # คำนวณ Smoothed TR, +DM และ -DM
       smoothed_tr = true_range.rolling(window=period).sum()
       smoothed_plus_dm = plus_dm.rolling(window=period).sum()
       smoothed_minus_dm = minus_dm.rolling(window=period).sum()
       
       # คำนวณ +DI และ -DI
       plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
       minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
       
       # คำนวณ DX
       dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
       
       # คำนวณ ADX
       adx = dx.rolling(window=period).mean()
       
       # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
       result[f'adx_{period}'] = adx
       result[f'plus_di_{period}'] = plus_di
       result[f'minus_di_{period}'] = minus_di
       
       return result
   
   def volume_profile(self, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
       """
       คำนวณ Volume Profile
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low, close และ volume
       n_bins (int): จำนวนช่วงราคา
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Volume Profile
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
       if 'timestamp' not in df.columns:
           logger.error("DataFrame ไม่มีคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการคำนวณ Volume Profile")
           return result
       
       # แปลง timestamp เป็น datetime ถ้าจำเป็น
       df_temp = df.copy()
       if not pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
           try:
               df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
           except:
               logger.error("ไม่สามารถแปลงคอลัมน์ timestamp เป็น datetime ได้")
               return result
       
       # สร้างคอลัมน์วันที่
       df_temp['date'] = df_temp['timestamp'].dt.date
       
       # คำนวณ Volume Profile สำหรับแต่ละวัน
       for date in df_temp['date'].unique():
           # ข้อมูลในวันนี้
           day_data = df_temp[df_temp['date'] == date]
           
           # หาช่วงราคาต่ำสุดและสูงสุด
           price_min = day_data['low'].min()
           price_max = day_data['high'].max()
           
           # สร้างช่วงราคา
           price_bins = np.linspace(price_min, price_max, n_bins + 1)
           
           # คำนวณราคากลางของแต่ละแท่ง
           typical_price = (day_data['high'] + day_data['low'] + day_data['close']) / 3
           
           # หาว่าแต่ละราคาอยู่ในช่วงใด
           bin_indices = np.digitize(typical_price, price_bins) - 1
           bin_indices = np.clip(bin_indices, 0, n_bins - 1)
           
           # คำนวณปริมาณการซื้อขายในแต่ละช่วงราคา
           volume_profile = np.zeros(n_bins)
           
           for i, bin_idx in enumerate(bin_indices):
               volume_profile[bin_idx] += day_data['volume'].iloc[i]
           
           # บันทึกข้อมูล Volume Profile
           for i, price_level in enumerate(price_bins[:-1]):
               mid_price = (price_bins[i] + price_bins[i+1]) / 2
               bin_name = f'volume_profile_{mid_price:.2f}'
               result.loc[day_data.index, bin_name] = volume_profile[i]
           
           # หาช่วงราคาที่มีปริมาณสูงสุด (Point of Control)
           poc_bin = np.argmax(volume_profile)
           poc_price = (price_bins[poc_bin] + price_bins[poc_bin+1]) / 2
           
           result.loc[day_data.index, 'volume_profile_poc'] = poc_price
       
       return result
   
   def moving_average_crossover(
       self, 
       df: pd.DataFrame, 
       short_period: int = 9, 
       long_period: int = 21
   ) -> pd.DataFrame:
       """
       คำนวณจุดตัดของค่าเฉลี่ยเคลื่อนที่ (Moving Average Crossover)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       short_period (int): ช่วงเวลาสำหรับค่าเฉลี่ยเคลื่อนที่ระยะสั้น
       long_period (int): ช่วงเวลาสำหรับค่าเฉลี่ยเคลื่อนที่ระยะยาว
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ crossover signals
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณค่าเฉลี่ยเคลื่อนที่
       short_ma = df['close'].rolling(window=short_period).mean()
       long_ma = df['close'].rolling(window=long_period).mean()
       
       # เพิ่มคอลัมน์ค่าเฉลี่ยเคลื่อนที่ลงใน DataFrame ผลลัพธ์
       result[f'ma_{short_period}'] = short_ma
       result[f'ma_{long_period}'] = long_ma
       
       # คำนวณสัญญาณตัด
       # 1 = ตัดขึ้น (Golden Cross), -1 = ตัดลง (Death Cross), 0 = ไม่มีการตัด
       crossover = pd.Series(0, index=df.index)
       
       prev_diff = short_ma.shift(1) - long_ma.shift(1)
       curr_diff = short_ma - long_ma
       
       crossover[(prev_diff <= 0) & (curr_diff > 0)] = 1    # Golden Cross
       crossover[(prev_diff >= 0) & (curr_diff < 0)] = -1   # Death Cross
       
       result['ma_crossover'] = crossover
       
       # คำนวณระยะห่างระหว่างค่าเฉลี่ยเคลื่อนที่ (เป็นเปอร์เซ็นต์)
       result['ma_diff_percent'] = 100 * (short_ma - long_ma) / long_ma
       
       return result
   
   def pivot_points(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
       """
       คำนวณจุดหมุน (Pivot Points)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       method (str): วิธีการคำนวณ ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Pivot Points
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
       if 'timestamp' not in df.columns:
           logger.error("DataFrame ไม่มีคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการคำนวณ Pivot Points")
           return result
       
       # แปลง timestamp เป็น datetime ถ้าจำเป็น
       df_temp = df.copy()
       if not pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
           try:
               df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
           except:
               logger.error("ไม่สามารถแปลงคอลัมน์ timestamp เป็น datetime ได้")
               return result
       
       # สร้างคอลัมน์วันที่
       df_temp['date'] = df_temp['timestamp'].dt.date
       
       # คำนวณ Pivot Points สำหรับแต่ละวัน
       for i, date in enumerate(df_temp['date'].unique()):
           if i == 0:
               # ไม่สามารถคำนวณได้สำหรับวันแรก
               continue
           
           # ข้อมูลในวันก่อนหน้า
           prev_date = df_temp['date'].unique()[i-1]
           prev_day_data = df_temp[df_temp['date'] == prev_date]
           
           # ข้อมูลในวันปัจจุบัน
           current_day_data = df_temp[df_temp['date'] == date]
           
           # ถ้าไม่มีข้อมูลวันก่อนหน้า ให้ข้าม
           if len(prev_day_data) == 0 or len(current_day_data) == 0:
               continue
           
           # ราคาสูงสุด ต่ำสุด และปิดของวันก่อนหน้า
           prev_high = prev_day_data['high'].max()
           prev_low = prev_day_data['low'].min()
           prev_close = prev_day_data['close'].iloc[-1]
           
           # คำนวณ Pivot Points ตามวิธีที่เลือก
           if method == 'standard':
               pivot = (prev_high + prev_low + prev_close) / 3
               s1 = (2 * pivot) - prev_high
               s2 = pivot - (prev_high - prev_low)
               s3 = s2 - (prev_high - prev_low)
               r1 = (2 * pivot) - prev_low
               r2 = pivot + (prev_high - prev_low)
               r3 = r2 + (prev_high - prev_low)
               
               # บันทึกค่าลงใน DataFrame ผลลัพธ์
               result.loc[current_day_data.index, 'pivot'] = pivot
               result.loc[current_day_data.index, 'support1'] = s1
               result.loc[current_day_data.index, 'support2'] = s2
               result.loc[current_day_data.index, 'support3'] = s3
               result.loc[current_day_data.index, 'resistance1'] = r1
               result.loc[current_day_data.index, 'resistance2'] = r2
               result.loc[current_day_data.index, 'resistance3'] = r3
           
           elif method == 'fibonacci':
               pivot = (prev_high + prev_low + prev_close) / 3
               s1 = pivot - 0.382 * (prev_high - prev_low)
               s2 = pivot - 0.618 * (prev_high - prev_low)
               s3 = pivot - 1.0 * (prev_high - prev_low)
               r1 = pivot + 0.382 * (prev_high - prev_low)
               r2 = pivot + 0.618 * (prev_high - prev_low)
               r3 = pivot + 1.0 * (prev_high - prev_low)
               
               # บันทึกค่าลงใน DataFrame ผลลัพธ์
               result.loc[current_day_data.index, 'pivot_fib'] = pivot
               result.loc[current_day_data.index, 'support1_fib'] = s1
               result.loc[current_day_data.index, 'support2_fib'] = s2
               result.loc[current_day_data.index, 'support3_fib'] = s3
               result.loc[current_day_data.index, 'resistance1_fib'] = r1
               result.loc[current_day_data.index, 'resistance2_fib'] = r2
               result.loc[current_day_data.index, 'resistance3_fib'] = r3
           
           elif method == 'woodie':
               pivot = (prev_high + prev_low + (2 * prev_close)) / 4
               s1 = (2 * pivot) - prev_high
               s2 = pivot - (prev_high - prev_low)
               s3 = s2 - (prev_high - prev_low)
               r1 = (2 * pivot) - prev_low
               r2 = pivot + (prev_high - prev_low)
               r3 = r2 + (prev_high - prev_low)
               
               # บันทึกค่าลงใน DataFrame ผลลัพธ์
               result.loc[current_day_data.index, 'pivot_woodie'] = pivot
               result.loc[current_day_data.index, 'support1_woodie'] = s1
               result.loc[current_day_data.index, 'support2_woodie'] = s2
               result.loc[current_day_data.index, 'support3_woodie'] = s3
               result.loc[current_day_data.index, 'resistance1_woodie'] = r1
               result.loc[current_day_data.index, 'resistance2_woodie'] = r2
               result.loc[current_day_data.index, 'resistance3_woodie'] = r3
           
           elif method == 'camarilla':
               range_val = prev_high - prev_low
               pivot = (prev_high + prev_low + prev_close) / 3
               s1 = prev_close - (range_val * 1.1 / 12)
               s2 = prev_close - (range_val * 1.1 / 6)
               s3 = prev_close - (range_val * 1.1 / 4)
               s4 = prev_close - (range_val * 1.1 / 2)
               r1 = prev_close + (range_val * 1.1 / 12)
               r2 = prev_close + (range_val * 1.1 / 6)
               r3 = prev_close + (range_val * 1.1 / 4)
               r4 = prev_close + (range_val * 1.1 / 2)
               
               # บันทึกค่าลงใน DataFrame ผลลัพธ์
               result.loc[current_day_data.index, 'pivot_cam'] = pivot
               result.loc[current_day_data.index, 'support1_cam'] = s1
               result.loc[current_day_data.index, 'support2_cam'] = s2
               result.loc[current_day_data.index, 'support3_cam'] = s3
               result.loc[current_day_data.index, 'support4_cam'] = s4
               result.loc[current_day_data.index, 'resistance1_cam'] = r1
               result.loc[current_day_data.index, 'resistance2_cam'] = r2
               result.loc[current_day_data.index, 'resistance3_cam'] = r3
               result.loc[current_day_data.index, 'resistance4_cam'] = r4

           elif method == 'demark':
               # คำนวณ X
               if prev_close < prev_open:
                   x = prev_high + (2 * prev_low) + prev_close
               elif prev_close > prev_open:
                   x = (2 * prev_high) + prev_low + prev_close
               else:
                   x = prev_high + prev_low + (2 * prev_close)
               
               pivot = x / 4
               s1 = x / 2 - prev_high
               r1 = x / 2 - prev_low
               
               # บันทึกค่าลงใน DataFrame ผลลัพธ์
               result.loc[current_day_data.index, 'pivot_demark'] = pivot
               result.loc[current_day_data.index, 'support1_demark'] = s1
               result.loc[current_day_data.index, 'resistance1_demark'] = r1
       
       return result
   
   def chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
       """
       คำนวณ Chaikin Money Flow (CMF)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low, close และ volume
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ CMF
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
       
       # เพิ่มคอลัมน์ CMF ลงใน DataFrame ผลลัพธ์
       result[f'cmf_{period}'] = cmf
       
       return result
   
   def momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
       """
       คำนวณ Momentum
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Momentum
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ Momentum
       momentum = df['close'] - df['close'].shift(period)
       
       # คำนวณ Momentum เป็นเปอร์เซ็นต์
       momentum_pct = (df['close'] / df['close'].shift(period) - 1) * 100
       
       # เพิ่มคอลัมน์ Momentum ลงใน DataFrame ผลลัพธ์
       result[f'momentum_{period}'] = momentum
       result[f'momentum_pct_{period}'] = momentum_pct
       
       return result
   
   def hull_moving_average(self, df: pd.DataFrame, period: int = 16) -> pd.DataFrame:
       """
       คำนวณ Hull Moving Average (HMA)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ HMA
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ Hull Moving Average
       wma1 = df['close'].rolling(window=period//2).apply(
           lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)),
           raw=True
       )
       wma2 = df['close'].rolling(window=period).apply(
           lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)),
           raw=True
       )
       
       # 2 * WMA(n/2) - WMA(n)
       sq_wma = 2 * wma1 - wma2
       
       # WMA(sqrt(n))
       sqrt_period = int(np.sqrt(period))
       hma = sq_wma.rolling(window=sqrt_period).apply(
           lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)),
           raw=True
       )
       
       # เพิ่มคอลัมน์ HMA ลงใน DataFrame ผลลัพธ์
       result[f'hma_{period}'] = hma
       
       return result
   
   def keltner_channels(
       self, 
       df: pd.DataFrame, 
       period: int = 20, 
       atr_period: int = 10, 
       multiplier: float = 2.0
   ) -> pd.DataFrame:
       """
       คำนวณ Keltner Channels
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       period (int): ช่วงเวลาสำหรับค่าเฉลี่ยเคลื่อนที่
       atr_period (int): ช่วงเวลาสำหรับ ATR
       multiplier (float): ตัวคูณสำหรับความกว้างของช่อง
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Keltner Channels
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ EMA ของราคาปิด
       middle_line = df['close'].ewm(span=period, adjust=False).mean()
       
       # คำนวณ ATR
       high = df['high']
       low = df['low']
       close = df['close']
       
       prev_close = close.shift(1)
       tr1 = high - low
       tr2 = (high - prev_close).abs()
       tr3 = (low - prev_close).abs()
       
       true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
       atr = true_range.rolling(window=atr_period).mean()
       
       # คำนวณแถบบนและล่าง
       upper_line = middle_line + (multiplier * atr)
       lower_line = middle_line - (multiplier * atr)
       
       # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
       result[f'keltner_middle_{period}'] = middle_line
       result[f'keltner_upper_{period}'] = upper_line
       result[f'keltner_lower_{period}'] = lower_line
       
       return result
   
   def price_rate_of_change(self, df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
       """
       คำนวณ Price Rate of Change (ROC)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ ROC
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ Price Rate of Change
       roc = (df['close'] / df['close'].shift(period) - 1) * 100
       
       # เพิ่มคอลัมน์ ROC ลงใน DataFrame ผลลัพธ์
       result[f'roc_{period}'] = roc
       
       return result
   
   def bears_vs_bulls(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
       """
       คำนวณ Bears vs Bulls Power
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       period (int): ช่วงเวลาสำหรับ EMA
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Bears Power และ Bulls Power
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ EMA ของราคาปิด
       ema = df['close'].ewm(span=period, adjust=False).mean()
       
       # คำนวณ Bears Power และ Bulls Power
       bears_power = df['low'] - ema
       bulls_power = df['high'] - ema
       
       # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
       result[f'bears_power_{period}'] = bears_power
       result[f'bulls_power_{period}'] = bulls_power
       
       return result
   
   def williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
       """
       คำนวณ Williams %R
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Williams %R
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณค่าสูงสุดและต่ำสุดในช่วงเวลาที่กำหนด
       high_max = df['high'].rolling(window=period).max()
       low_min = df['low'].rolling(window=period).min()
       
       # คำนวณ Williams %R
       williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
       
       # เพิ่มคอลัมน์ Williams %R ลงใน DataFrame ผลลัพธ์
       result[f'williams_r_{period}'] = williams_r
       
       return result
   
   def standard_deviation(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
       """
       คำนวณ Standard Deviation
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Standard Deviation
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณส่วนเบี่ยงเบนมาตรฐาน
       std = df['close'].rolling(window=period).std()
       
       # คำนวณส่วนเบี่ยงเบนมาตรฐานเป็นเปอร์เซ็นต์ของราคา
       std_percent = (std / df['close']) * 100
       
       # เพิ่มคอลัมน์ลงใน DataFrame ผลลัพธ์
       result[f'std_{period}'] = std
       result[f'std_percent_{period}'] = std_percent
       
       return result
   
   def trix(self, df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
       """
       คำนวณ Triple Exponential Average (TRIX)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ TRIX
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ EMA ครั้งที่ 1
       ema1 = df['close'].ewm(span=period, adjust=False).mean()
       
       # คำนวณ EMA ครั้งที่ 2
       ema2 = ema1.ewm(span=period, adjust=False).mean()
       
       # คำนวณ EMA ครั้งที่ 3
       ema3 = ema2.ewm(span=period, adjust=False).mean()
       
       # คำนวณ TRIX
       trix = (ema3 / ema3.shift(1) - 1) * 100
       
       # เพิ่มคอลัมน์ TRIX ลงใน DataFrame ผลลัพธ์
       result[f'trix_{period}'] = trix
       
       # คำนวณ TRIX Signal (EMA ของ TRIX)
       signal_period = 9
       result[f'trix_signal_{period}_{signal_period}'] = trix.ewm(span=signal_period, adjust=False).mean()
       
       return result
   
   def average_directional_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
       """
       คำนวณ Average Directional Index (ADX)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low และ close
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ ADX, +DI และ -DI
       """
       # แทนที่ด้วยฟังก์ชัน adx ที่มีอยู่แล้ว
       return self.adx(df, period)
   
   def money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
       """
       คำนวณ Money Flow Index (MFI)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low, close และ volume
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ MFI
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ Typical Price
       typical_price = (df['high'] + df['low'] + df['close']) / 3
       
       # คำนวณ Money Flow
       money_flow = typical_price * df['volume']
       
       # คำนวณ Positive และ Negative Money Flow
       typical_price_change = typical_price.diff()
       
       positive_flow = money_flow.copy()
       negative_flow = money_flow.copy()
       
       positive_flow[typical_price_change <= 0] = 0
       negative_flow[typical_price_change >= 0] = 0
       
       # คำนวณผลรวมของ Positive และ Negative Money Flow ในช่วงเวลาที่กำหนด
       positive_flow_sum = positive_flow.rolling(window=period).sum()
       negative_flow_sum = negative_flow.rolling(window=period).sum()
       
       # ป้องกันการหารด้วย 0
       negative_flow_sum = negative_flow_sum.replace(0, np.nan)
       
       # คำนวณ Money Ratio
       money_ratio = positive_flow_sum / negative_flow_sum
       
       # คำนวณ Money Flow Index
       mfi = 100 - (100 / (1 + money_ratio))
       
       # เพิ่มคอลัมน์ MFI ลงใน DataFrame ผลลัพธ์
       result[f'mfi_{period}'] = mfi
       
       return result
   
   def accumulation_distribution_line(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       คำนวณ Accumulation/Distribution Line
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ high, low, close และ volume
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ Accumulation/Distribution Line
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณ Money Flow Multiplier
       mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
       mfm = mfm.replace([np.inf, -np.inf], 0)
       
       # คำนวณ Money Flow Volume
       mfv = mfm * df['volume']
       
       # คำนวณ Accumulation/Distribution Line
       adl = mfv.cumsum()
       
       # เพิ่มคอลัมน์ ADL ลงใน DataFrame ผลลัพธ์
       result['adl'] = adl
       
       return result
   
   def time_segmented_volume(self, df: pd.DataFrame, period: int = 21) -> pd.DataFrame:
       """
       คำนวณ Time Segmented Volume (TSV)
       
       Parameters:
       df (pd.DataFrame): DataFrame ที่มีคอลัมน์ close และ volume
       period (int): ช่วงเวลาสำหรับการคำนวณ
       
       Returns:
       pd.DataFrame: DataFrame ที่มีคอลัมน์ TSV
       """
       # สร้าง DataFrame สำหรับผลลัพธ์
       result = pd.DataFrame(index=df.index)
       
       # คำนวณการเปลี่ยนแปลงของราคา
       price_change = df['close'].diff()
       
       # คำนวณ Volume Force
       volume_force = price_change * df['volume']
       
       # คำนวณ TSV
       tsv = volume_force.rolling(window=period).sum()
       
       # เพิ่มคอลัมน์ TSV ลงใน DataFrame ผลลัพธ์
       result[f'tsv_{period}'] = tsv
       
       # คำนวณ TSV Signal Line
       signal_period = 7
       result[f'tsv_signal_{period}_{signal_period}'] = tsv.ewm(span=signal_period, adjust=False).mean()
       
       return result

   #endregion

# สร้าง singleton instance
technical_indicators = TechnicalIndicators()

def get_technical_indicators() -> TechnicalIndicators:
   """
   ดึง singleton instance ของ TechnicalIndicators
   
   Returns:
   TechnicalIndicators: instance ของ TechnicalIndicators
   """
   return technical_indicators

# ฟังก์ชันสำหรับใช้งานเป็น command line tool
def main():
   """
   ฟังก์ชันหลักสำหรับใช้งานเป็น command line tool
   """
   import argparse
   
   parser = argparse.ArgumentParser(description="คำนวณตัวชี้วัดทางเทคนิคสำหรับข้อมูลการเทรด")
   parser.add_argument("--input", type=str, required=True, help="ไฟล์ข้อมูลนำเข้า (CSV หรือ Parquet)")
   parser.add_argument("--output", type=str, required=True, help="ไฟล์ข้อมูลส่งออก (Parquet)")
   parser.add_argument("--indicators", type=str, help="รายการตัวชี้วัดที่ต้องการคำนวณ (คั่นด้วยเครื่องหมายจุลภาค)")
   parser.add_argument("--list", action="store_true", help="แสดงรายการตัวชี้วัดที่รองรับทั้งหมด")
   
   args = parser.parse_args()
   
   # แสดงรายการตัวชี้วัดที่รองรับทั้งหมด
   if args.list:
       indicators = TechnicalIndicators().indicator_functions.keys()
       print("รายการตัวชี้วัดที่รองรับทั้งหมด:")
       for i, indicator in enumerate(sorted(indicators), 1):
           print(f"{i}. {indicator}")
       return
   
   # โหลดข้อมูล
   try:
       if args.input.endswith('.csv'):
           df = pd.read_csv(args.input)
       elif args.input.endswith('.parquet'):
           df = pd.read_parquet(args.input)
       else:
           print(f"นามสกุลไฟล์ไม่รองรับ: {args.input}")
           return
       
       print(f"โหลดข้อมูลสำเร็จ: {len(df)} แถว")
   except Exception as e:
       print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
       return
   
   # แยกรายการตัวชี้วัด
   indicators = None
   if args.indicators:
       indicators = [indicator.strip() for indicator in args.indicators.split(',')]
   
   # คำนวณตัวชี้วัด
   try:
       result_df = TechnicalIndicators().calculate_indicators(df, indicators)
       print(f"คำนวณตัวชี้วัดสำเร็จ: {len(result_df.columns) - len(df.columns)} ตัวชี้วัด")
   except Exception as e:
       print(f"เกิดข้อผิดพลาดในการคำนวณตัวชี้วัด: {e}")
       return
   
   # บันทึกข้อมูล
   try:
       # สร้างโฟลเดอร์หากไม่มี
       os.makedirs(os.path.dirname(args.output), exist_ok=True)
       
       # บันทึกไฟล์
       result_df.to_parquet(args.output)
       print(f"บันทึกข้อมูลสำเร็จที่: {args.output}")
   except Exception as e:
       print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")

if __name__ == "__main__":
   # ทำให้สามารถใช้งานจาก command line ได้
   main()