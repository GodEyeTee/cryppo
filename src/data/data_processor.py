# src/data/data_processor.py

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

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    ประมวลผลข้อมูลสำหรับ Reinforcement Learning
    
    คลาสนี้ใช้สำหรับประมวลผลข้อมูลตลาดดิบให้เป็นรูปแบบที่เหมาะสมสำหรับการฝึกโมเดล RL
    รองรับการทำ Log Transform, Z-score Normalization, การจัดการค่าที่หายไป และการสร้างคุณลักษณะ (feature engineering)
    """
    
    def __init__(self, config=None):
        """
        กำหนดค่าเริ่มต้นสำหรับ DataProcessor
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
            from src.data.indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators(self.config)
            df = tech_indicators.calculate_indicators(df, additional_indicators)
        
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