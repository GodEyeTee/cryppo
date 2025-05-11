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

from src.data.processors.data_processor import DataProcessor
from src.data.indicators.indicator_registry import TechnicalIndicators
from src.utils.config import get_config
from src.data.utils.file_utils import ensure_directory_exists, load_dataframe, find_latest_file, save_json, load_json

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class MarketDataManager:
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
            
            # โหลดข้อมูลด้วย file_utils
            self.raw_data = load_dataframe(self.file_path)
            
            if self.raw_data.empty:
                logger.error(f"ไม่สามารถโหลดข้อมูลจาก {self.file_path} หรือข้อมูลว่างเปล่า")
                return False
            
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
            self.detail_raw_data = load_dataframe(detail_file_path)
            
            if self.detail_raw_data.empty:
                logger.error(f"ไม่สามารถโหลดข้อมูลรายละเอียดจาก {detail_file_path} หรือข้อมูลว่างเปล่า")
                return False
            
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
                logger.warning(f"ไม่สามารถสร้าง window ได้ batch_data={len(batch_data)}, window_size={self.window_size}")
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
        for batch_idx in range(self.num_batches):
            yield self.get_batch(batch_idx)
    
    def inverse_transform_price(self, normalized_price: float, price_column: str = 'close') -> float:
        """
        แปลงราคาที่ normalize แล้วกลับเป็นราคาปกติ
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
        if not self.data_loaded or self.data is None:
            logger.error("ยังไม่ได้โหลดข้อมูล")
            return {}
        
        try:
            # แยกคอลัมน์ timestamp และคอลัมน์ที่ไม่ใช่ตัวเลขออกก่อน
            data_copy = self.data.copy()
            
            # คอลัมน์ที่ต้องกำจัดออก
            drop_columns = []
            
            # กำจัดคอลัมน์ timestamp
            if 'timestamp' in data_copy.columns:
                drop_columns.append('timestamp')
            
            # กำจัดคอลัมน์ที่ไม่ใช่ตัวเลข
            for col in data_copy.columns:
                if not pd.api.types.is_numeric_dtype(data_copy[col]):
                    drop_columns.append(col)
            
            # ถ้ามีคอลัมน์ที่ต้องกำจัด ให้ลบออก
            if drop_columns:
                logger.info(f"กำจัดคอลัมน์ที่ไม่ใช่ตัวเลข: {drop_columns}")
                data_copy = data_copy.drop(columns=drop_columns)
            
            # ตรวจสอบว่าเหลือข้อมูลหรือไม่
            if data_copy.empty or data_copy.shape[1] == 0:
                logger.error("ไม่มีข้อมูลตัวเลขที่จะใช้เทรนโมเดล")
                return {}
            
            # แก้ไขข้อมูลที่เป็น NaN หรือ Inf
            data_copy = data_copy.fillna(0)
            data_copy = data_copy.replace([np.inf, -np.inf], 0)
            
            # ตรวจสอบว่าข้อมูลทั้งหมดเป็นตัวเลขและไม่มี NaN หรือ Inf
            assert np.all(np.isfinite(data_copy.values)), "ข้อมูลมีค่า NaN หรือ Inf"
            
            # แปลงเป็น NumPy array
            data = data_copy.values.astype(np.float32)  # แปลงเป็น float32 เพื่อป้องกันปัญหา
            
            # สร้าง sliding windows
            windows = []
            
            for i in range(len(data) - self.window_size + 1):
                window = data[i:i+self.window_size]
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
            
            # สร้าง Tensor และย้ายไปยัง GPU ถ้าจำเป็น
            train_tensor = torch.tensor(train_data, dtype=torch.float32)
            val_tensor = torch.tensor(val_data, dtype=torch.float32)
            test_tensor = torch.tensor(test_data, dtype=torch.float32)
            
            if self.use_gpu:
                train_tensor = train_tensor.to(self.device)
                val_tensor = val_tensor.to(self.device)
                test_tensor = test_tensor.to(self.device)
            
            # สร้าง DataLoader
            from torch.utils.data import TensorDataset, DataLoader
            
            train_dataset = TensorDataset(train_tensor)
            val_dataset = TensorDataset(val_tensor)
            test_dataset = TensorDataset(test_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            
            # ระบุขนาดของ feature
            feature_size = data_copy.shape[1]
            
            return {
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "feature_size": feature_size,  # เพิ่มขนาดของ feature
                "feature_names": data_copy.columns.tolist()  # เพิ่มชื่อของ feature
            }
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างชุดข้อมูล: {e}")
            return {}
    
    def save_stats(self, file_path: str) -> bool:
        if not self.stats:
            logger.error("ไม่มีสถิติสำหรับบันทึก")
            return False
        
        try:
            return save_json(self.stats, file_path)
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกสถิติ: {e}")
            return False
    
    def load_stats(self, file_path: str) -> bool:
        try:
            # โหลดสถิติ
            stats = load_json(file_path)
            
            if not stats:
                logger.error(f"ไม่สามารถโหลดสถิติจาก {file_path}")
                return False
            
            # อัพเดทสถิติ
            self.stats = stats
            
            # อัพเดทสถิติใน data_processor
            self.data_processor.stats = self.stats
            
            logger.info(f"โหลดสถิติจาก: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดสถิติ: {e}")
            return False
    
    def find_files(self, data_dir: str, symbol: str, timeframe: str, file_pattern: str = "*") -> List[str]:
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
    
    def filter_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
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
    
    def get_timeseries_stats(self, column_name: str) -> Dict[str, Any]:
        if not self.data_loaded or self.raw_data is None:
            logger.error("ยังไม่ได้โหลดข้อมูล")
            return {}
        
        if column_name not in self.raw_data.columns:
            logger.error(f"ไม่พบคอลัมน์: {column_name}")
            return {}
        
        try:
            # ดึงข้อมูลคอลัมน์ที่ต้องการ
            data = self.raw_data[column_name]
            
            # สถิติพื้นฐาน
            stats = {
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "25%": data.quantile(0.25),
                "50%": data.median(),
                "75%": data.quantile(0.75),
                "max": data.max(),
                "count": len(data),
                "missing": data.isnull().sum(),
                "skewness": data.skew(),
                "kurtosis": data.kurt()
            }
            
            # สถิติของการเปลี่ยนแปลง
            pct_change = data.pct_change().dropna()
            stats.update({
                "return_mean": pct_change.mean(),
                "return_std": pct_change.std(),
                "return_min": pct_change.min(),
                "return_max": pct_change.max(),
                "volatility": pct_change.std() * (252 ** 0.5)  # Annualized volatility (assuming daily data)
            })
            
            # ทดสอบ stationarity (Augmented Dickey-Fuller test)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(data.dropna())
                stats["adf_test"] = {
                    "adf_statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "is_stationary": adf_result[1] < 0.05
                }
            except ImportError:
                logger.warning("ไม่สามารถทำ ADF test ได้: ไม่พบโมดูล statsmodels")
            except Exception as e:
                logger.warning(f"ไม่สามารถทำ ADF test ได้: {e}")
            
            # คำนวณ autocorrelation
            try:
                from statsmodels.tsa.stattools import acf
                acf_result = acf(data.dropna(), nlags=10)
                stats["autocorrelation"] = {
                    "lag_1": acf_result[1] if len(acf_result) > 1 else None,
                    "lag_5": acf_result[5] if len(acf_result) > 5 else None,
                    "lag_10": acf_result[10] if len(acf_result) > 10 else None
                }
            except ImportError:
                logger.warning("ไม่สามารถคำนวณ autocorrelation ได้: ไม่พบโมดูล statsmodels")
            except Exception as e:
                logger.warning(f"ไม่สามารถคำนวณ autocorrelation ได้: {e}")
            
            return stats
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณสถิติ: {e}")
            return {}
    
    def get_correlation_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.data_loaded or self.raw_data is None:
            logger.error("ยังไม่ได้โหลดข้อมูล")
            return pd.DataFrame()
        
        try:
            # ถ้าไม่ระบุคอลัมน์ ให้ใช้คอลัมน์ตัวเลขทั้งหมด
            if columns is None:
                # เลือกคอลัมน์ตัวเลขเท่านั้น
                numeric_cols = self.raw_data.select_dtypes(include=np.number).columns
                
                # ไม่รวมคอลัมน์ timestamp และอื่นๆ ที่ไม่เกี่ยวข้อง
                exclude_cols = ['timestamp', 'date', 'time', 'close_time']
                columns = [col for col in numeric_cols if col not in exclude_cols]
            else:
                # ตรวจสอบคอลัมน์ที่ระบุ
                invalid_cols = [col for col in columns if col not in self.raw_data.columns]
                if invalid_cols:
                    logger.warning(f"ไม่พบคอลัมน์: {', '.join(invalid_cols)}")
                    columns = [col for col in columns if col in self.raw_data.columns]
            
            # คำนวณเมทริกซ์สหสัมพันธ์
            return self.raw_data[columns].corr()
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณเมทริกซ์สหสัมพันธ์: {e}")
            return pd.DataFrame()

    def get_period_stats(self, period: str = 'daily') -> pd.DataFrame:
        if not self.data_loaded or self.raw_data is None:
            logger.error("ยังไม่ได้โหลดข้อมูล")
            return pd.DataFrame()
        
        if 'timestamp' not in self.raw_data.columns:
            logger.error("ไม่พบคอลัมน์ timestamp")
            return pd.DataFrame()
        
        try:
            # สร้าง DataFrame สำหรับการวิเคราะห์
            df = self.raw_data.copy()
            
            # ตั้งช่วงเวลา
            if period == 'daily':
                df['period'] = df['timestamp'].dt.date
            elif period == 'weekly':
                df['period'] = df['timestamp'].dt.to_period('W').dt.start_time
            elif period == 'monthly':
                df['period'] = df['timestamp'].dt.to_period('M').dt.start_time
            elif period == 'hourly':
                df['period'] = df['timestamp'].dt.floor('H')
            else:
                logger.error(f"ช่วงเวลาไม่รองรับ: {period}")
                return pd.DataFrame()
            
            # สถิติที่ต้องการคำนวณ
            stats = []
            
            # วิเคราะห์แต่ละช่วงเวลา
            for period_val, group in df.groupby('period'):
                # สถิติพื้นฐาน
                open_price = group['open'].iloc[0]
                high_price = group['high'].max()
                low_price = group['low'].min()
                close_price = group['close'].iloc[-1]
                volume = group['volume'].sum()
                
                # คำนวณผลตอบแทน
                returns = (close_price - open_price) / open_price * 100
                
                # เพิ่มสถิติในช่วงเวลานี้
                stats.append({
                    'period': period_val,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'returns': returns,
                    'range': high_price - low_price,
                    'volatility': group['close'].pct_change().std() * 100,
                    'trades': group['trades'].sum() if 'trades' in group.columns else None,
                    'candles': len(group)
                })
            
            # สร้าง DataFrame จากสถิติ
            stats_df = pd.DataFrame(stats)
            
            # เรียงลำดับตามช่วงเวลา
            stats_df = stats_df.sort_values('period')
            
            return stats_df
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์สถิติตามช่วงเวลา: {e}")
            return pd.DataFrame()
