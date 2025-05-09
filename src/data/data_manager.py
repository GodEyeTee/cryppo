# src/data/data_manager.py

import os
import numpy as np
import pandas as pd
import torch
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime, timedelta

# นำเข้าการตั้งค่าจาก config.py
from src.utils.config import get_config

# นำเข้าคลาสและฟังก์ชันที่เกี่ยวข้อง
from src.data.data_processor import DataProcessor
from src.data.indicators import TechnicalIndicators

# ตั้งค่า logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
   handler = logging.StreamHandler()
   handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
   logger.addHandler(handler)

class MarketDataManager:
   """
   จัดการข้อมูลตลาดสำหรับระบบจำลองการเทรด
   
   คลาสนี้รับผิดชอบการโหลด จัดการ และเตรียมข้อมูลสำหรับสภาพแวดล้อมจำลองการเทรด
   รวมถึงการจัดการข้อมูลไทม์เฟรมต่างๆ และการเชื่อมโยงข้อมูลรายละเอียด (เช่น 1m) กับข้อมูลหลัก (เช่น 5m)
   """
   
   def __init__(
       self,
       file_path: Optional[str] = None,
       symbol: Optional[str] = None,
       base_timeframe: Optional[str] = None,
       detail_timeframe: Optional[str] = None,
       start_date: Optional[str] = None,
       end_date: Optional[str] = None,
       batch_size: Optional[int] = None,
       window_size: Optional[int] = None,
       indicators: Optional[List[str]] = None,
       use_gpu: Optional[bool] = None,
       config = None
   ):
       """
       กำหนดค่าเริ่มต้นสำหรับ MarketDataManager
       
       Parameters:
       file_path (str, optional): พาธไปยังไฟล์ข้อมูลหลัก
       symbol (str, optional): สัญลักษณ์คู่สกุลเงิน (เช่น "BTCUSDT")
       base_timeframe (str, optional): ไทม์เฟรมหลัก (เช่น "5m")
       detail_timeframe (str, optional): ไทม์เฟรมรายละเอียด (เช่น "1m")
       start_date (str, optional): วันที่เริ่มต้น ("YYYY-MM-DD")
       end_date (str, optional): วันที่สิ้นสุด ("YYYY-MM-DD")
       batch_size (int, optional): ขนาดของแต่ละ batch
       window_size (int, optional): ขนาดของหน้าต่างสำหรับแต่ละตัวอย่าง
       indicators (List[str], optional): รายการตัวชี้วัดที่ต้องการคำนวณ
       use_gpu (bool, optional): ใช้ GPU หรือไม่
       config (Config, optional): อ็อบเจ็กต์การตั้งค่า หรือ None เพื่อโหลดตั้งค่าเริ่มต้น
       """
       # โหลดการตั้งค่า
       self.config = config if config is not None else get_config()
       
       # ดึงการตั้งค่าที่เกี่ยวข้อง
       data_config = self.config.extract_subconfig("data")
       cuda_config = self.config.extract_subconfig("cuda")
       
       # กำหนดค่าพารามิเตอร์
       self.file_path = file_path if file_path else data_config.get("file_path")
       self.symbol = symbol if symbol else data_config.get("default_symbol", "BTCUSDT")
       self.base_timeframe = base_timeframe if base_timeframe else data_config.get("base_timeframe", "5m")
       self.detail_timeframe = detail_timeframe if detail_timeframe else data_config.get("detail_timeframe", "1m")
       self.start_date = start_date if start_date else data_config.get("default_start_date")
       self.end_date = end_date if end_date else data_config.get("default_end_date")
       self.batch_size = batch_size if batch_size else data_config.get("batch_size", 1024)
       self.window_size = window_size if window_size else data_config.get("window_size", 60)
       self.indicators = indicators if indicators else data_config.get("indicators", [])
       
       # ตั้งค่า GPU
       default_use_gpu = cuda_config.get("use_cuda", True) and torch.cuda.is_available()
       self.use_gpu = use_gpu if use_gpu is not None else default_use_gpu
       self.device = f"cuda:{cuda_config.get('device', 0)}" if self.use_gpu else "cpu"
       
       # ข้อมูลทางสถิติสำหรับ normalization
       self.stats = {}
       
       # สร้างอ็อบเจ็กต์สำหรับประมวลผลข้อมูล
       self.data_processor = DataProcessor(self.config)
       self.technical_indicators = TechnicalIndicators(self.config)
       
       # ข้อมูลที่โหลดแล้ว
       self.data = None
       self.raw_data = None
       self.detail_data = None
       self.detail_raw_data = None
       
       # ขนาดของข้อมูล
       self.data_length = 0
       self.num_batches = 0
       
       # สถานะการโหลดข้อมูล
       self.data_loaded = False
       self.detail_data_loaded = False
       
       # ข้อมูล GPU
       if self.use_gpu:
           self.gpu_mem = torch.cuda.get_device_properties(0).total_memory
           self.gpu_mem_fraction = cuda_config.get("memory_fraction", 0.7)
           self.max_gpu_batch = 0  # จะคำนวณหลังจากโหลดข้อมูลแล้ว
       
       logger.info(f"กำลังใช้อุปกรณ์: {self.device}")
       logger.info(f"Symbol: {self.symbol}, ไทม์เฟรมหลัก: {self.base_timeframe}, ไทม์เฟรมรายละเอียด: {self.detail_timeframe}")
       
       # โหลดข้อมูลถ้ามีการระบุ file_path
       if self.file_path:
           self.load_data()
   
   def load_data(self, file_path: Optional[str] = None, reload: bool = False) -> bool:
       """
       โหลดข้อมูลจากไฟล์
       
       Parameters:
       file_path (str, optional): พาธไปยังไฟล์ข้อมูล (ถ้าไม่ระบุจะใช้ค่าจาก self.file_path)
       reload (bool): โหลดข้อมูลใหม่แม้จะเคยโหลดแล้ว
       
       Returns:
       bool: True ถ้าโหลดสำเร็จ, False ถ้าไม่สำเร็จ
       """
       # ถ้าไม่ระบุ file_path ให้ใช้ค่าเดิม
       if file_path:
           self.file_path = file_path
       
       # ถ้าไม่มี file_path หรือมีการโหลดข้อมูลแล้วและไม่ต้องการโหลดใหม่
       if not self.file_path or (self.data_loaded and not reload):
           return self.data_loaded
       
       try:
           logger.info(f"กำลังโหลดข้อมูลจาก: {self.file_path}")
           
           # โหลดข้อมูลตามนามสกุลไฟล์
           if self.file_path.endswith('.csv'):
               self.raw_data = pd.read_csv(self.file_path)
           elif self.file_path.endswith('.parquet'):
               self.raw_data = pd.read_parquet(self.file_path)
           else:
               logger.error(f"นามสกุลไฟล์ไม่รองรับ: {self.file_path}")
               return False
           
           # จัดการข้อมูล timestamp
           if 'timestamp' in self.raw_data.columns and not pd.api.types.is_datetime64_any_dtype(self.raw_data['timestamp']):
               if isinstance(self.raw_data['timestamp'].iloc[0], (int, np.int64)):
                   # ถ้าเป็น UNIX timestamp (มิลลิวินาที)
                   self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'], unit='ms')
               else:
                   # ถ้าเป็นสตริง
                   self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
           
           # เรียงลำดับข้อมูลตาม timestamp
           self.raw_data = self.raw_data.sort_values('timestamp').reset_index(drop=True)
           
           # กรองข้อมูลตามช่วงวันที่
           if self.start_date:
               start_date = pd.to_datetime(self.start_date)
               self.raw_data = self.raw_data[self.raw_data['timestamp'] >= start_date]
           
           if self.end_date:
               end_date = pd.to_datetime(self.end_date)
               self.raw_data = self.raw_data[self.raw_data['timestamp'] <= end_date]
           
           # รีเซ็ตดัชนี
           self.raw_data = self.raw_data.reset_index(drop=True)
           
           # ประมวลผลข้อมูล
           self.process_data()
           
           # พยายามโหลดข้อมูลรายละเอียด (ถ้ามี)
           self.load_detail_data()
           
           # ตั้งค่าสถานะการโหลดข้อมูล
           self.data_loaded = True
           
           # คำนวณขนาดของข้อมูล
           self.data_length = len(self.data)
           self.num_batches = (self.data_length - self.window_size + self.batch_size) // self.batch_size
           
           # คำนวณขนาด batch สูงสุดสำหรับ GPU
           if self.use_gpu:
               # ประมาณการณ์หน่วยความจำที่ต้องใช้ต่อตัวอย่าง (bytes)
               # 4 bytes per float32 x จำนวนคอลัมน์ x window_size
               bytes_per_sample = 4 * self.data.shape[1] * self.window_size
               
               # คำนวณขนาด batch สูงสุดที่พอดีกับหน่วยความจำ GPU
               available_mem = self.gpu_mem * self.gpu_mem_fraction
               self.max_gpu_batch = int(available_mem / bytes_per_sample)
               
               # จำกัดขนาด batch ไม่ให้เกิน batch_size ที่กำหนด
               self.max_gpu_batch = min(self.max_gpu_batch, self.batch_size)
               
               logger.info(f"GPU Memory: {self.gpu_mem / 1e9:.2f} GB, Max GPU Batch: {self.max_gpu_batch}")
           
           logger.info(f"โหลดข้อมูลสำเร็จ: {self.data_length} แถว, {self.data.shape[1]} คอลัมน์")
           
           return True
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
           self.data_loaded = False
           return False
   
   def process_data(self) -> None:
       """
       ประมวลผลข้อมูลดิบ
       """
       if self.raw_data is None or self.raw_data.empty:
           logger.error("ไม่มีข้อมูลดิบสำหรับประมวลผล")
           return
       
       try:
           logger.info("กำลังประมวลผลข้อมูล...")
           
           # คำนวณตัวชี้วัดเพิ่มเติม (ถ้ามี)
           if self.indicators:
               self.raw_data = self.technical_indicators.calculate_indicators(self.raw_data, self.indicators)
           
           # ประมวลผลข้อมูลด้วย DataProcessor
           self.data = self.data_processor.preprocess_data(self.raw_data)
           
           # เก็บสถิติสำหรับการแปลงกลับ
           self.stats = self.data_processor.stats
           
           logger.info(f"ประมวลผลข้อมูลสำเร็จ: {self.data.shape[1]} คอลัมน์")
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")
   
   def load_detail_data(self) -> bool:
       """
       โหลดข้อมูลรายละเอียด (เช่น 1m) ที่สอดคล้องกับข้อมูลหลัก (เช่น 5m)
       
       Returns:
       bool: True ถ้าโหลดสำเร็จ, False ถ้าไม่สำเร็จ
       """
       if not self.file_path or not self.raw_data is not None:
           return False
       
       # สร้างพาธไปยังไฟล์ข้อมูลรายละเอียด
       detail_file_path = self._get_detail_file_path()
       
       if not detail_file_path or not os.path.exists(detail_file_path):
           logger.warning(f"ไม่พบไฟล์ข้อมูลรายละเอียด: {detail_file_path}")
           return False
       
       try:
           logger.info(f"กำลังโหลดข้อมูลรายละเอียด {self.detail_timeframe} จาก: {detail_file_path}")
           
           # โหลดข้อมูลตามนามสกุลไฟล์
           if detail_file_path.endswith('.csv'):
               self.detail_raw_data = pd.read_csv(detail_file_path)
           elif detail_file_path.endswith('.parquet'):
               self.detail_raw_data = pd.read_parquet(detail_file_path)
           else:
               logger.error(f"นามสกุลไฟล์ไม่รองรับ: {detail_file_path}")
               return False
           
           # จัดการข้อมูล timestamp
           if 'timestamp' in self.detail_raw_data.columns and not pd.api.types.is_datetime64_any_dtype(self.detail_raw_data['timestamp']):
               if isinstance(self.detail_raw_data['timestamp'].iloc[0], (int, np.int64)):
                   # ถ้าเป็น UNIX timestamp (มิลลิวินาที)
                   self.detail_raw_data['timestamp'] = pd.to_datetime(self.detail_raw_data['timestamp'], unit='ms')
               else:
                   # ถ้าเป็นสตริง
                   self.detail_raw_data['timestamp'] = pd.to_datetime(self.detail_raw_data['timestamp'])
           
           # เรียงลำดับข้อมูลตาม timestamp
           self.detail_raw_data = self.detail_raw_data.sort_values('timestamp').reset_index(drop=True)
           
           # กรองข้อมูลตามช่วงเวลาของข้อมูลหลัก
           min_time = self.raw_data['timestamp'].min()
           max_time = self.raw_data['timestamp'].max()
           
           self.detail_raw_data = self.detail_raw_data[
               (self.detail_raw_data['timestamp'] >= min_time) &
               (self.detail_raw_data['timestamp'] <= max_time)
           ].reset_index(drop=True)
           
           # ประมวลผลข้อมูลรายละเอียด
           self.detail_data = self.data_processor.preprocess_data(self.detail_raw_data)
           
           # ตั้งค่าสถานะการโหลดข้อมูลรายละเอียด
           self.detail_data_loaded = True
           
           logger.info(f"โหลดข้อมูลรายละเอียดสำเร็จ: {len(self.detail_data)} แถว")
           
           return True
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลรายละเอียด: {e}")
           self.detail_data_loaded = False
           return False
   
   def _get_detail_file_path(self) -> Optional[str]:
       """
       สร้างพาธไปยังไฟล์ข้อมูลรายละเอียดจากพาธของไฟล์ข้อมูลหลัก
       
       Returns:
       str หรือ None: พาธไปยังไฟล์ข้อมูลรายละเอียด หรือ None ถ้าไม่สามารถสร้างได้
       """
       if not self.file_path or not self.base_timeframe or not self.detail_timeframe:
           return None
       
       try:
           # แยกพาธ ชื่อไฟล์ และนามสกุล
           file_dir = os.path.dirname(self.file_path)
           file_name = os.path.basename(self.file_path)
           
           # แทนที่ไทม์เฟรมในชื่อไฟล์
           detail_file_name = file_name.replace(self.base_timeframe, self.detail_timeframe)
           
           # สร้างพาธใหม่
           detail_file_path = os.path.join(file_dir, detail_file_name)
           
           return detail_file_path
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการสร้างพาธไปยังไฟล์ข้อมูลรายละเอียด: {e}")
           return None
   
   def get_batch(self, batch_idx: int) -> Dict[str, Any]:
       """
       ดึงชุดข้อมูล batch ที่ต้องการ
       
       Parameters:
       batch_idx (int): ดัชนีของ batch
       
       Returns:
       dict: ข้อมูลที่ประมวลผลแล้ว
       """
       if not self.data_loaded or self.data is None:
           logger.error("ยังไม่ได้โหลดข้อมูล")
           return {}
       
       # ตรวจสอบว่า batch_idx อยู่ในช่วงที่ถูกต้อง
       if batch_idx < 0 or batch_idx >= self.num_batches:
           logger.error(f"batch_idx ไม่ถูกต้อง: {batch_idx}, ต้องอยู่ในช่วง 0-{self.num_batches-1}")
           return {}
       
       try:
           # คำนวณดัชนีเริ่มต้นและสิ้นสุดของ batch
           start_row = batch_idx * self.batch_size
           adjusted_start = max(0, start_row - (self.window_size - 1))
           end_row = min(start_row + self.batch_size, self.data_length)
           
           # ดึงข้อมูลเฉพาะที่ต้องการ
           batch_data = self.data.iloc[adjusted_start:end_row].copy()
           batch_raw = self.raw_data.iloc[adjusted_start:end_row].reset_index(drop=True)
           
           # สร้าง sliding windows
           windows = []
           timestamps = []
           
           for i in range(self.window_size - 1, len(batch_data)):
               start_idx = i - (self.window_size - 1)
               end_idx = i + 1
               window = batch_data.iloc[start_idx:end_idx].values
               timestamp = batch_data.iloc[i]['timestamp'] if 'timestamp' in batch_data.columns else None
               
               windows.append(window)
               timestamps.append(timestamp)
           
           if not windows:
               logger.warning(f"ไม่สามารถสร้าง window ได้ บทาด_ฟาะา={len(batch_data)}, window_size={self.window_size}")
               return {}
           
           # แปลงเป็น numpy array
           windows_array = np.array(windows)
           
           # แปลงเป็น tensor
           windows_tensor = torch.tensor(windows_array, dtype=torch.float32)
           
           # ย้ายไปยัง GPU ถ้าจำเป็น
           if self.use_gpu:
               windows_tensor = windows_tensor.to(self.device)
           
           # ดึงข้อมูลรายละเอียดที่สอดคล้องกัน (ถ้ามี)
           detail_batch = None
           detail_raw = None
           
           if self.detail_data_loaded and self.detail_data is not None:
               # หาช่วงเวลาที่ตรงกัน
               if 'timestamp' in batch_raw.columns:
                   min_time = batch_raw['timestamp'].min()
                   max_time = batch_raw['timestamp'].max()
                   
                   detail_mask = (
                       (self.detail_raw_data['timestamp'] >= min_time) &
                       (self.detail_raw_data['timestamp'] <= max_time)
                   )
                   
                   detail_raw = self.detail_raw_data[detail_mask].reset_index(drop=True)
                   detail_batch = self.detail_data[detail_mask].reset_index(drop=True)
                   
                   # แปลงเป็น tensor ถ้ามีข้อมูล
                   if not detail_batch.empty and self.use_gpu:
                       detail_tensor = torch.tensor(detail_batch.values, dtype=torch.float32).to(self.device)
                       detail_batch = detail_tensor
           
           return {
               'windows': windows_tensor,
               'timestamps': timestamps,
               'raw_data': batch_raw,
               'detail_data': detail_batch,
               'detail_raw': detail_raw,
               'batch_idx': batch_idx
           }
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล batch: {e}")
           return {}
   
   def data_generator(self):
       """
       Generator สำหรับการวนซ้ำผ่านชุดข้อมูลทั้งหมด
       
       Yields:
       dict: ข้อมูลสำหรับแต่ละ batch
       """
       for batch_idx in range(self.num_batches):
           yield self.get_batch(batch_idx)
   
   def inverse_transform_price(self, normalized_price: float, price_column: str = 'close') -> float:
       """
       แปลงราคาที่ normalize แล้วกลับเป็นราคาปกติ
       
       Parameters:
       normalized_price (float): ราคาที่ normalize แล้ว
       price_column (str): ชื่อคอลัมน์ราคา ('open', 'high', 'low', 'close')
       
       Returns:
       float: ราคาปกติ
       """
       if not self.data_loaded or not self.stats:
           logger.error("ยังไม่ได้โหลดข้อมูลหรือไม่มีสถิติสำหรับการแปลงกลับ")
           return normalized_price
       
       try:
           # แปลงกลับโดยใช้ DataProcessor
           return self.data_processor.inverse_transform_price(normalized_price, price_column)
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการแปลงราคากลับ: {e}")
           return normalized_price
   
   def get_data_info(self) -> Dict[str, Any]:
       """
       ดึงข้อมูลเกี่ยวกับชุดข้อมูลที่โหลดแล้ว
       
       Returns:
       dict: ข้อมูลเกี่ยวกับชุดข้อมูล
       """
       if not self.data_loaded or self.data is None:
           return {
               'loaded': False,
               'error': 'ยังไม่ได้โหลดข้อมูล'
           }
       
       # ข้อมูลทั่วไป
       info = {
           'loaded': True,
           'symbol': self.symbol,
           'base_timeframe': self.base_timeframe,
           'detail_timeframe': self.detail_timeframe,
           'data_length': self.data_length,
           'num_batches': self.num_batches,
           'window_size': self.window_size,
           'batch_size': self.batch_size,
           'columns': list(self.data.columns),
           'has_detail_data': self.detail_data_loaded
       }
       
       # ข้อมูลเวลา
       if 'timestamp' in self.raw_data.columns:
           info['start_time'] = self.raw_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
           info['end_time'] = self.raw_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
           info['duration'] = str(self.raw_data['timestamp'].max() - self.raw_data['timestamp'].min())
       
       # ข้อมูลราคา
       if 'close' in self.raw_data.columns:
           info['min_price'] = self.raw_data['close'].min()
           info['max_price'] = self.raw_data['close'].max()
           info['avg_price'] = self.raw_data['close'].mean()
           
           # คำนวณผลตอบแทนรวม
           if len(self.raw_data) > 1:
               first_price = self.raw_data['close'].iloc[0]
               last_price = self.raw_data['close'].iloc[-1]
               total_return = (last_price / first_price - 1) * 100
               info['total_return'] = f"{total_return:.2f}%"
       
       # ข้อมูล GPU
       if self.use_gpu:
           info['gpu'] = {
               'device': self.device,
               'memory': f"{self.gpu_mem / 1e9:.2f} GB",
               'max_gpu_batch': self.max_gpu_batch
           }
       
       return info

   def create_training_data(
       self,
       validation_ratio: float = 0.1,
       test_ratio: float = 0.1,
       shuffle: bool = True
   ) -> Dict[str, Any]:
       """
       สร้างชุดข้อมูลสำหรับการเทรนโมเดล
       
       Parameters:
       validation_ratio (float): สัดส่วนของข้อมูลสำหรับ validation
       test_ratio (float): สัดส่วนของข้อมูลสำหรับ test
       shuffle (bool): สุ่มข้อมูลหรือไม่
       
       Returns:
       Dict[str, Any]: Dictionary ของชุดข้อมูล
       """
       if not self.data_loaded or self.data is None:
           logger.error("ยังไม่ได้โหลดข้อมูล")
           return {}
       
       try:
           # ใช้ DataProcessor สร้างชุดข้อมูล
           training_data = self.data_processor.create_training_data(
               df=self.data,
               window_size=self.window_size,
               batch_size=self.batch_size,
               validation_ratio=validation_ratio,
               test_ratio=test_ratio,
               shuffle=shuffle
           )
           
           logger.info(f"สร้างชุดข้อมูลสำเร็จ: train={training_data['train_size']}, val={training_data['val_size']}, test={training_data['test_size']}")
           
           return training_data
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการสร้างชุดข้อมูล: {e}")
           return {}
   
   def save_stats(self, file_path: str) -> bool:
       """
       บันทึกสถิติสำหรับการแปลงกลับ
       
       Parameters:
       file_path (str): พาธสำหรับบันทึกไฟล์
       
       Returns:
       bool: True ถ้าบันทึกสำเร็จ, False ถ้าไม่สำเร็จ
       """
       if not self.stats:
           logger.error("ไม่มีสถิติสำหรับบันทึก")
           return False
       
       try:
           # สร้างโฟลเดอร์หากไม่มี
           os.makedirs(os.path.dirname(file_path), exist_ok=True)
           
           # บันทึกสถิติเป็น JSON
           with open(file_path, 'w') as f:
               json.dump(self.stats, f, indent=2)
           
           logger.info(f"บันทึกสถิติที่: {file_path}")
           return True
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการบันทึกสถิติ: {e}")
           return False
   
   def load_stats(self, file_path: str) -> bool:
       """
       โหลดสถิติสำหรับการแปลงกลับ
       
       Parameters:
       file_path (str): พาธไปยังไฟล์สถิติ
       
       Returns:
       bool: True ถ้าโหลดสำเร็จ, False ถ้าไม่สำเร็จ
       """
       try:
           # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
           if not os.path.exists(file_path):
               logger.error(f"ไม่พบไฟล์: {file_path}")
               return False
           
           # โหลดสถิติจาก JSON
           with open(file_path, 'r') as f:
               self.stats = json.load(f)
           
           # อัพเดทสถิติใน data_processor
           self.data_processor.stats = self.stats
           
           logger.info(f"โหลดสถิติจาก: {file_path}")
           return True
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการโหลดสถิติ: {e}")
           return False
   
   def find_files(self, data_dir: str, symbol: str, timeframe: str, file_pattern: str = "*") -> List[str]:
       """
       ค้นหาไฟล์ข้อมูลในไดเรกทอรี
       
       Parameters:
       data_dir (str): ไดเรกทอรีที่ต้องการค้นหา
       symbol (str): สัญลักษณ์คู่สกุลเงิน
       timeframe (str): ไทม์เฟรม
       file_pattern (str): รูปแบบไฟล์เพิ่มเติม
       
       Returns:
       List[str]: รายการพาธไปยังไฟล์ที่พบ
       """
       try:
           import glob
           
           # สร้างพาธสำหรับค้นหา
           symbol_dir = os.path.join(data_dir, symbol.replace("/", "-"))
           timeframe_dir = os.path.join(symbol_dir, timeframe)
           search_pattern = os.path.join(timeframe_dir, f"*{file_pattern}*")
           
           # ค้นหาไฟล์
           files = glob.glob(search_pattern)
           
           # กรองเฉพาะไฟล์ CSV และ Parquet
           files = [f for f in files if f.endswith('.csv') or f.endswith('.parquet')]
           
           # เรียงลำดับตามวันที่แก้ไข (ใหม่สุดก่อน)
           files.sort(key=os.path.getmtime, reverse=True)
           
           return files
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการค้นหาไฟล์: {e}")
           return []
   
   def get_timeseries_stats(self, column: str = 'close') -> Dict[str, Any]:
       """
       คำนวณสถิติของอนุกรมเวลา
       
       Parameters:
       column (str): ชื่อคอลัมน์ที่ต้องการวิเคราะห์
       
       Returns:
       Dict[str, Any]: สถิติของอนุกรมเวลา
       """
       if not self.data_loaded or self.raw_data is None or column not in self.raw_data.columns:
           logger.error(f"ไม่พบข้อมูลหรือคอลัมน์ {column}")
           return {}
       
       try:
           # ข้อมูลทั่วไป
           series = self.raw_data[column]
           
           stats = {
               'column': column,
               'count': len(series),
               'min': series.min(),
               'max': series.max(),
               'mean': series.mean(),
               'median': series.median(),
               'std': series.std(),
               'skewness': series.skew(),
               'kurtosis': series.kurtosis(),
           }
           
           # คำนวณการเปลี่ยนแปลงเป็นเปอร์เซ็นต์
           pct_change = series.pct_change().dropna()
           
           stats.update({
               'return_min': pct_change.min(),
               'return_max': pct_change.max(),
               'return_mean': pct_change.mean(),
               'return_std': pct_change.std(),
               'return_skewness': pct_change.skew(),
               'return_kurtosis': pct_change.kurtosis(),
           })
           
           # คำนวณ autocorrelation
           from statsmodels.tsa.stattools import acf
           try:
               autocorr = acf(series.dropna(), nlags=20)
               stats['autocorrelation'] = autocorr.tolist()
           except:
               stats['autocorrelation'] = []
           
           # ตรวจสอบ stationarity
           from statsmodels.tsa.stattools import adfuller
           try:
               adf_result = adfuller(series.dropna())
               stats['adf_test'] = {
                   'statistic': adf_result[0],
                   'pvalue': adf_result[1],
                   'is_stationary': adf_result[1] < 0.05
               }
           except:
               stats['adf_test'] = {'is_stationary': False}
           
           return stats
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการคำนวณสถิติ: {e}")
           return {}
   
   def get_correlation_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
       """
       คำนวณเมทริกซ์สหสัมพันธ์ระหว่างคอลัมน์
       
       Parameters:
       columns (List[str], optional): รายการคอลัมน์ที่ต้องการวิเคราะห์
       
       Returns:
       pd.DataFrame: เมทริกซ์สหสัมพันธ์
       """
       if not self.data_loaded or self.raw_data is None:
           logger.error("ไม่มีข้อมูล")
           return pd.DataFrame()
       
       try:
           # ถ้าไม่ระบุคอลัมน์ ให้ใช้เฉพาะคอลัมน์ตัวเลข
           if columns is None:
               numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
               
               # ลบคอลัมน์ที่ไม่เกี่ยวข้องออก
               exclude_cols = ['timestamp', 'date', 'time', 'index']
               columns = [col for col in numeric_cols if col not in exclude_cols]
           
           # ตรวจสอบว่ามีคอลัมน์ที่ระบุหรือไม่
           if not columns or not all(col in self.raw_data.columns for col in columns):
               missing_cols = [col for col in columns if col not in self.raw_data.columns]
               logger.error(f"ไม่พบคอลัมน์: {missing_cols}")
               return pd.DataFrame()
           
           # คำนวณเมทริกซ์สหสัมพันธ์
           correlation_matrix = self.raw_data[columns].corr()
           
           return correlation_matrix
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการคำนวณเมทริกซ์สหสัมพันธ์: {e}")
           return pd.DataFrame()
   
   def resample_data(
       self, 
       target_timeframe: str, 
       method: str = 'ohlc',
       save_to_file: Optional[str] = None
   ) -> pd.DataFrame:
       """
       สร้างข้อมูลใหม่ที่มีไทม์เฟรมต่างจากข้อมูลเดิม (resampling)
       
       Parameters:
       target_timeframe (str): ไทม์เฟรมเป้าหมาย (เช่น '5min', '1h', '1d')
       method (str): วิธีการสร้างข้อมูลใหม่ ('ohlc' หรือ 'last')
       save_to_file (str, optional): พาธสำหรับบันทึกไฟล์
       
       Returns:
       pd.DataFrame: ข้อมูลที่สร้างใหม่
       """
       if not self.data_loaded or self.raw_data is None:
           logger.error("ไม่มีข้อมูล")
           return pd.DataFrame()
       
       try:
           # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
           if 'timestamp' not in self.raw_data.columns:
               logger.error("ไม่พบคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการ resample")
               return pd.DataFrame()
           
           # ตั้งดัชนีเป็น timestamp
           df = self.raw_data.copy()
           df = df.set_index('timestamp')
           
           # แปลงรูปแบบไทม์เฟรม
           freq_map = {
               '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
               '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
               '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
           }
           
           pandas_freq = freq_map.get(target_timeframe, target_timeframe)
           
           # Resample ข้อมูลตามวิธีที่เลือก
           if method == 'ohlc':
               # ต้องมีคอลัมน์ open, high, low, close
               if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                   logger.error("ไม่พบคอลัมน์ OHLC ซึ่งจำเป็นสำหรับการ resample แบบ 'ohlc'")
                   return pd.DataFrame()
               
               # Resample แบบ OHLC
               resampled = pd.DataFrame()
               resampled['open'] = df['open'].resample(pandas_freq).first()
               resampled['high'] = df['high'].resample(pandas_freq).max()
               resampled['low'] = df['low'].resample(pandas_freq).min()
               resampled['close'] = df['close'].resample(pandas_freq).last()
               
               # ถ้ามีคอลัมน์ volume
               if 'volume' in df.columns:
                   resampled['volume'] = df['volume'].resample(pandas_freq).sum()
               
               # สำหรับคอลัมน์อื่นๆ ใช้ค่าสุดท้าย
               for col in df.columns:
                   if col not in ['open', 'high', 'low', 'close', 'volume']:
                       resampled[col] = df[col].resample(pandas_freq).last()
           
           else:  # method == 'last'
               # ใช้ค่าสุดท้ายสำหรับทุกคอลัมน์
               resampled = df.resample(pandas_freq).last()
           
           # รีเซ็ตดัชนี
           resampled = resampled.reset_index()
           
           # บันทึกไฟล์ถ้าต้องการ
           if save_to_file:
               # สร้างโฟลเดอร์หากไม่มี
               os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
               
               # บันทึกไฟล์ตามนามสกุล
               if save_to_file.endswith('.csv'):
                   resampled.to_csv(save_to_file, index=False)
               elif save_to_file.endswith('.parquet'):
                   resampled.to_parquet(save_to_file, index=False)
               else:
                   logger.error(f"นามสกุลไฟล์ไม่รองรับ: {save_to_file}")
               
               logger.info(f"บันทึกข้อมูล resample ที่: {save_to_file}")
           
           return resampled
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการ resample ข้อมูล: {e}")
           return pd.DataFrame()
   
   def get_period_stats(self, period: str = 'daily') -> pd.DataFrame:
       """
       วิเคราะห์สถิติตามช่วงเวลา (รายวัน, รายสัปดาห์, รายเดือน)
       
       Parameters:
       period (str): ช่วงเวลาที่ต้องการวิเคราะห์ ('daily', 'weekly', 'monthly', 'hourly')
       
       Returns:
       pd.DataFrame: สถิติตามช่วงเวลา
       """
       if not self.data_loaded or self.raw_data is None:
           logger.error("ไม่มีข้อมูล")
           return pd.DataFrame()
       
       try:
           # ตรวจสอบว่ามีคอลัมน์ timestamp หรือไม่
           if 'timestamp' not in self.raw_data.columns:
               logger.error("ไม่พบคอลัมน์ timestamp ซึ่งจำเป็นสำหรับการวิเคราะห์ตามช่วงเวลา")
               return pd.DataFrame()
           
           # ต้องมีคอลัมน์ close
           if 'close' not in self.raw_data.columns:
               logger.error("ไม่พบคอลัมน์ close ซึ่งจำเป็นสำหรับการวิเคราะห์")
               return pd.DataFrame()
           
           # สร้างข้อมูลสำหรับวิเคราะห์
           df = self.raw_data.copy()
           
           # สร้างคอลัมน์เพิ่มเติม
           df['return'] = df['close'].pct_change() * 100  # เปอร์เซ็นต์ผลตอบแทน
           
           # สร้างคอลัมน์ตามช่วงเวลา
           if period == 'daily':
               df['period'] = df['timestamp'].dt.date
           elif period == 'weekly':
               df['period'] = df['timestamp'].dt.strftime('%Y-%U')  # ปี-สัปดาห์
           elif period == 'monthly':
               df['period'] = df['timestamp'].dt.strftime('%Y-%m')  # ปี-เดือน
           elif period == 'hourly':
               df['period'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00')  # ปี-เดือน-วัน ชั่วโมง
           else:
               logger.error(f"ไม่รองรับช่วงเวลา: {period}")
               return pd.DataFrame()
           
           # กลุ่มตามช่วงเวลาและคำนวณสถิติ
           stats = df.groupby('period').agg({
               'open': 'first',
               'high': 'max',
               'low': 'min',
               'close': 'last',
               'volume': 'sum',
               'return': ['mean', 'std', 'min', 'max', 'sum']
           })
           
           # ปรับรูปแบบชื่อคอลัมน์
           stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
           
           # รีเซ็ตดัชนี
           stats = stats.reset_index()
           
           return stats
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์สถิติตามช่วงเวลา: {e}")
           return pd.DataFrame()
   
   def filter_data(
       self, 
       start_date: Optional[str] = None, 
       end_date: Optional[str] = None,
       min_price: Optional[float] = None,
       max_price: Optional[float] = None,
       conditions: Optional[Dict[str, Any]] = None
   ) -> bool:
       """
       กรองข้อมูลตามเงื่อนไข
       
       Parameters:
       start_date (str, optional): วันที่เริ่มต้น ("YYYY-MM-DD")
       end_date (str, optional): วันที่สิ้นสุด ("YYYY-MM-DD")
       min_price (float, optional): ราคาต่ำสุด
       max_price (float, optional): ราคาสูงสุด
       conditions (Dict[str, Any], optional): เงื่อนไขเพิ่มเติม (คอลัมน์: ค่า)
       
       Returns:
       bool: True ถ้ากรองสำเร็จ, False ถ้าไม่สำเร็จ
       """
       if not self.data_loaded or self.raw_data is None:
           logger.error("ไม่มีข้อมูล")
           return False
       
       try:
           # สร้างสำเนาของข้อมูลเดิม
           filtered_data = self.raw_data.copy()
           
           # กรองตามวันที่
           if start_date and 'timestamp' in filtered_data.columns:
               start_date = pd.to_datetime(start_date)
               filtered_data = filtered_data[filtered_data['timestamp'] >= start_date]
           
           if end_date and 'timestamp' in filtered_data.columns:
               end_date = pd.to_datetime(end_date)
               filtered_data = filtered_data[filtered_data['timestamp'] <= end_date]
           
           # กรองตามราคา
           if min_price is not None and 'close' in filtered_data.columns:
               filtered_data = filtered_data[filtered_data['close'] >= min_price]
           
           if max_price is not None and 'close' in filtered_data.columns:
               filtered_data = filtered_data[filtered_data['close'] <= max_price]
           
           # กรองตามเงื่อนไขเพิ่มเติม
           if conditions:
               for column, value in conditions.items():
                   if column in filtered_data.columns:
                       if isinstance(value, (list, tuple)) and len(value) == 2:
                           # ช่วงค่า [min, max]
                           filtered_data = filtered_data[(filtered_data[column] >= value[0]) & (filtered_data[column] <= value[1])]
                       else:
                           # ค่าเดียว
                           filtered_data = filtered_data[filtered_data[column] == value]
           
           # ตรวจสอบว่ามีข้อมูลเหลือหรือไม่
           if filtered_data.empty:
               logger.warning("ไม่มีข้อมูลที่ตรงกับเงื่อนไข")
               return False
           
           # อัพเดทข้อมูล
           self.raw_data = filtered_data.reset_index(drop=True)
           
           # ประมวลผลข้อมูลใหม่
           self.process_data()
           
           # อัพเดทขนาดของข้อมูล
           self.data_length = len(self.data)
           self.num_batches = (self.data_length - self.window_size + self.batch_size) // self.batch_size
           
           logger.info(f"กรองข้อมูลสำเร็จ: เหลือ {self.data_length} แถว")
           
           return True
       
       except Exception as e:
           logger.error(f"เกิดข้อผิดพลาดในการกรองข้อมูล: {e}")
           return False

# ฟังก์ชันสำหรับใช้งานเป็น command line tool
def main():
   """
   ฟังก์ชันหลักสำหรับใช้งานเป็น command line tool
   """
   import argparse
   
   parser = argparse.ArgumentParser(description="จัดการข้อมูลตลาดสำหรับระบบจำลองการเทรด")
   parser.add_argument("--input", type=str, help="ไฟล์ข้อมูลนำเข้า (CSV หรือ Parquet)")
   parser.add_argument("--output", type=str, help="ไฟล์ข้อมูลส่งออก (Parquet)")
   parser.add_argument("--symbol", type=str, help="สัญลักษณ์คู่สกุลเงิน (เช่น BTCUSDT)")
   parser.add_argument("--base-timeframe", type=str, help="ไทม์เฟรมหลัก (เช่น 5m)")
   parser.add_argument("--detail-timeframe", type=str, help="ไทม์เฟรมรายละเอียด (เช่น 1m)")
   parser.add_argument("--start-date", type=str, help="วันที่เริ่มต้น (YYYY-MM-DD)")
   parser.add_argument("--end-date", type=str, help="วันที่สิ้นสุด (YYYY-MM-DD)")
   parser.add_argument("--indicators", type=str, help="รายการตัวชี้วัดที่ต้องการคำนวณ (คั่นด้วยเครื่องหมายจุลภาค)")
   parser.add_argument("--batch-size", type=int, help="ขนาดของแต่ละ batch")
   parser.add_argument("--window-size", type=int, help="ขนาดของหน้าต่างสำหรับแต่ละตัวอย่าง")
   parser.add_argument("--info", action="store_true", help="แสดงข้อมูลเกี่ยวกับชุดข้อมูล")
   parser.add_argument("--resample", type=str, help="สร้างข้อมูลใหม่ที่มีไทม์เฟรมนี้")
   parser.add_argument("--correlation", action="store_true", help="คำนวณเมทริกซ์สหสัมพันธ์")
   parser.add_argument("--period-stats", type=str, choices=['daily', 'weekly', 'monthly', 'hourly'], help="วิเคราะห์สถิติตามช่วงเวลา")
   parser.add_argument("--save-stats", type=str, help="บันทึกสถิติไปยังไฟล์นี้")
   parser.add_argument("--load-stats", type=str, help="โหลดสถิติจากไฟล์นี้")
   parser.add_argument("--no-gpu", action="store_true", help="ไม่ใช้ GPU")
   
   args = parser.parse_args()
   
   # แยกรายการตัวชี้วัด
   indicators = None
   if args.indicators:
       indicators = [indicator.strip() for indicator in args.indicators.split(',')]
   
   # สร้าง data manager
   data_manager = MarketDataManager(
       file_path=args.input,
       symbol=args.symbol,
       base_timeframe=args.base_timeframe,
       detail_timeframe=args.detail_timeframe,
       start_date=args.start_date,
       end_date=args.end_date,
       batch_size=args.batch_size,
       window_size=args.window_size,
       indicators=indicators,
       use_gpu=not args.no_gpu
   )
   
   # โหลดสถิติ
   if args.load_stats:
       data_manager.load_stats(args.load_stats)
   
   # แสดงข้อมูลเกี่ยวกับชุดข้อมูล
   if args.info and data_manager.data_loaded:
       info = data_manager.get_data_info()
       print("\nข้อมูลเกี่ยวกับชุดข้อมูล:")
       for key, value in info.items():
           if key == 'columns':
               print(f"  คอลัมน์: {', '.join(value[:10])}{' และอีก ' + str(len(value) - 10) + ' คอลัมน์' if len(value) > 10 else ''}")
           elif isinstance(value, dict):
               print(f"  {key}:")
               for k, v in value.items():
                   print(f"    {k}: {v}")
           else:
               print(f"  {key}: {value}")
       print()

   # สร้างข้อมูลใหม่ที่มีไทม์เฟรมต่างจากข้อมูลเดิม
   if args.resample and data_manager.data_loaded:
       output_file = args.output if args.output else f"resampled_{args.resample}.parquet"
       resampled = data_manager.resample_data(args.resample, save_to_file=output_file)
       print(f"\nสร้างข้อมูลใหม่ที่มีไทม์เฟรม {args.resample} สำเร็จ: {len(resampled)} แถว")
       print(f"บันทึกที่: {output_file}")
   
   # คำนวณเมทริกซ์สหสัมพันธ์
   if args.correlation and data_manager.data_loaded:
       corr_matrix = data_manager.get_correlation_matrix()
       print("\nเมทริกซ์สหสัมพันธ์:")
       
       # แสดงเฉพาะคอลัมน์ที่น่าสนใจ
       interesting_cols = ['open', 'high', 'low', 'close', 'volume']
       interesting_cols = [col for col in interesting_cols if col in corr_matrix.columns]
       
       # เพิ่มคอลัมน์ตัวชี้วัดที่น่าสนใจ
       for col in corr_matrix.columns:
           if 'rsi_' in col or 'macd_' in col or 'bb_' in col or 'ema_' in col or 'sma_' in col:
               interesting_cols.append(col)
       
       # ลบคอลัมน์ที่ซ้ำกัน
       interesting_cols = list(dict.fromkeys(interesting_cols))
       
       # จำกัดจำนวนคอลัมน์ที่แสดง
       if len(interesting_cols) > 10:
           interesting_cols = interesting_cols[:10]
       
       if interesting_cols:
           print(corr_matrix.loc[interesting_cols, interesting_cols].round(3))
       else:
           print(corr_matrix.round(3))
   
   # วิเคราะห์สถิติตามช่วงเวลา
   if args.period_stats and data_manager.data_loaded:
       period_stats = data_manager.get_period_stats(args.period_stats)
       print(f"\nสถิติตามช่วงเวลา ({args.period_stats}):")
       
       # แสดงเฉพาะ 10 แถวแรกและสุดท้าย
       if len(period_stats) > 20:
           print(pd.concat([period_stats.head(10), period_stats.tail(10)]))
       else:
           print(period_stats)
       
       # บันทึกไฟล์ถ้ามีการระบุ output
       if args.output:
           if args.output.endswith('.csv'):
               period_stats.to_csv(args.output, index=False)
           elif args.output.endswith('.parquet'):
               period_stats.to_parquet(args.output, index=False)
           print(f"บันทึกสถิติตามช่วงเวลาที่: {args.output}")
   
   # บันทึกสถิติ
   if args.save_stats and data_manager.data_loaded:
       data_manager.save_stats(args.save_stats)
   
   # บันทึกข้อมูลที่ประมวลผลแล้ว
   if args.output and data_manager.data_loaded and not args.resample and not args.period_stats:
       # สร้างโฟลเดอร์หากไม่มี
       os.makedirs(os.path.dirname(args.output), exist_ok=True)
       
       # บันทึกไฟล์
       data_manager.data.to_parquet(args.output)
       print(f"บันทึกข้อมูลที่: {args.output}")

if __name__ == "__main__":
   # ทำให้สามารถใช้งานจาก command line ได้
   main()