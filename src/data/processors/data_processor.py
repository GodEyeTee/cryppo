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

from src.utils.config import get_config
from src.data.transforms.data_transforms import (
    log_transform, inverse_log_transform, z_score_normalize, inverse_z_score,
    min_max_scale, inverse_min_max_scale, rolling_window
)
from src.data.transforms.data_cleaning import (
    remove_duplicates, handle_missing_values, remove_outliers,
    fill_missing_timestamps, ensure_ohlc_integrity, validate_timestamp_order
)

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class DataProcessor:    
    def __init__(self, config=None):
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
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.csv",
        additional_indicators: Optional[List[str]] = None
    ) -> List[str]:
        import glob
        import os
        from pathlib import Path
        
        # ตรวจสอบว่าไดเรกทอรีนำเข้ามีอยู่หรือไม่
        if not os.path.exists(input_dir):
            logger.error(f"ไม่พบไดเรกทอรีนำเข้า: {input_dir}")
            return []
        
        # สร้างไดเรกทอรีสำหรับไฟล์ผลลัพธ์
        os.makedirs(output_dir, exist_ok=True)
        
        # ค้นหาไฟล์ทั้งหมดที่ตรงกับรูปแบบ
        search_pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            logger.warning(f"ไม่พบไฟล์ที่ตรงกับรูปแบบ {file_pattern} ในไดเรกทอรี {input_dir}")
            return []
        
        # ประมวลผลไฟล์แต่ละไฟล์
        processed_files = []
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, file_name)
            
            # เปลี่ยนนามสกุลไฟล์เป็น .parquet ถ้าต้องการ
            if output_path.endswith('.csv') and self.config.extract_subconfig("preprocessing").get("output_format") == "parquet":
                output_path = os.path.splitext(output_path)[0] + '.parquet'
            
            # ประมวลผลไฟล์
            df = self.process_file(file_path, output_path, additional_indicators)
            
            if not df.empty:
                processed_files.append(output_path)
        
        logger.info(f"ประมวลผลไฟล์ทั้งหมด {len(processed_files)} ไฟล์")
        
        return processed_files
    
    def process_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        additional_indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        logger.info(f"กำลังประมวลผลไฟล์: {input_file}")
        
        # โหลดข้อมูล
        df = self.load_data(input_file)
        
        if df.empty:
            logger.error(f"ไม่สามารถโหลดข้อมูลจาก {input_file} หรือข้อมูลว่างเปล่า")
            return df
        
        # จัดการกับค่าที่หายไป
        if self.handle_missing:
            df = handle_missing_values(df, strategy=self.fill_missing_strategy)
        
        # คำนวณตัวชี้วัดเพิ่มเติม
        if additional_indicators:
            from src.data.indicators.indicator_registry import TechnicalIndicators
            tech_indicators = TechnicalIndicators(self.config)
            df = tech_indicators.calculate_indicators(df, additional_indicators)
        
        # กำจัด outliers
        if self.remove_outliers:
            df = remove_outliers(df, method='zscore', threshold=self.outlier_std_threshold)
        
        # ตรวจสอบความถูกต้องของข้อมูล OHLC
        df = ensure_ohlc_integrity(df)
        
        # ตรวจสอบลำดับของ timestamp
        df = validate_timestamp_order(df)
        
        # ลบข้อมูลที่ซ้ำกัน
        df = remove_duplicates(df, subset=['timestamp'])
        
        # ประมวลผลข้อมูล
        processed_df = self.preprocess_data(df)
        
        # บันทึกไฟล์
        if output_file:
            self.save_processed_data(processed_df, output_file)
            self.save_stats(os.path.splitext(output_file)[0] + "_stats.json")
        
        return processed_df
    
    def load_data(self, file_path: str) -> pd.DataFrame:
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
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ประมวลผลข้อมูลด้วย Log Transform และ Z-score Normalization
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
                    ohlcv_data = log_transform(ohlcv_data, is_volume_col)
                
                # ทำ Z-score Normalization (ถ้าเลือกใช้)
                if self.use_z_score:
                    normalized_ohlcv, ohlcv_stats = z_score_normalize(ohlcv_data)
                    
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
                    normalized_other, other_stats = z_score_normalize(other_data)
                    
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
    
    def inverse_transform(self, data: np.ndarray, stat_type: str = "ohlcv") -> np.ndarray:
        if stat_type not in self.stats:
            logger.error(f"ไม่พบสถิติประเภท: {stat_type}")
            return data
        
        stats = self.stats[stat_type]
        
        # แปลงจาก list เป็น numpy array
        means = np.array(stats["means"])
        stds = np.array(stats["stds"])
        
        # แปลงกลับจาก Z-score
        if self.use_z_score:
            data = inverse_z_score(data, means, stds)
        
        # แปลงกลับจาก Log transform
        if stats.get("log_transform", False):
            is_volume_col = np.array(stats.get("is_volume_col", [False] * data.shape[1]))
            data = inverse_log_transform(data, is_volume_col)
        
        return data
    
    def inverse_transform_price(self, price: float, column: str = "close") -> float:
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
        try:
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # บันทึกไฟล์
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
            elif output_file.endswith('.parquet'):
                df.to_parquet(output_file, index=False)
            else:
                # เพิ่มนามสกุล .parquet ถ้าไม่มีนามสกุล
                output_file = output_file + '.parquet'
                df.to_parquet(output_file, index=False)
            
            logger.info(f"บันทึกข้อมูลที่ผ่านการประมวลผลแล้วที่: {output_file}")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
    
    def save_stats(self, output_file: str) -> None:
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
    
    def create_training_data(
        self,
        df: pd.DataFrame,
        window_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Dict[str, Any]:
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
