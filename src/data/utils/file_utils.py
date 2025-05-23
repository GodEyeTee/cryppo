import os
import glob
import json
import logging
import pandas as pd
from typing import List, Optional, Dict, Union, Any

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการสร้างไดเรกทอรี {directory}: {e}")
        return False

def list_files(directory: str, pattern: str = "*") -> List[str]:
    if not os.path.exists(directory):
        logger.warning(f"ไม่พบไดเรกทอรี {directory}")
        return []
    
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    files.sort()
    
    return files

def get_file_info(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        logger.warning(f"ไม่พบไฟล์ {file_path}")
        return {}
    
    file_stats = os.stat(file_path)
    
    return {
        "path": file_path,
        "name": os.path.basename(file_path),
        "size": file_stats.st_size,
        "modified": pd.Timestamp(file_stats.st_mtime, unit='s'),
        "extension": os.path.splitext(file_path)[1]
    }

def save_dataframe(df: pd.DataFrame, file_path: str, file_format: str = "parquet") -> bool:
    if df.empty:
        logger.warning("DataFrame ว่างเปล่า ไม่บันทึกไฟล์")
        return False
    
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    
    try:
        if file_format.lower() in ["csv", "both"]:
            csv_path = f"{file_path}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"บันทึกข้อมูลเป็น CSV ที่: {csv_path}")
        
        if file_format.lower() in ["parquet", "both"]:
            parquet_path = f"{file_path}.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"บันทึกข้อมูลเป็น Parquet ที่: {parquet_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์ {file_path}: {e}")
        return False

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    try:
        directory = os.path.dirname(file_path)
        ensure_directory_exists(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"บันทึกข้อมูล JSON ที่: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์ JSON {file_path}: {e}")
        return False

def load_json(file_path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(file_path):
            logger.warning(f"ไม่พบไฟล์ {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"โหลดข้อมูล JSON จาก: {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ JSON {file_path}: {e}")
        return {}

def load_dataframe(file_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            logger.warning(f"ไม่พบไฟล์ {file_path}")
            return pd.DataFrame()
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logger.error(f"นามสกุลไฟล์ไม่รองรับ: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"โหลดข้อมูลจาก: {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ {file_path}: {e}")
        return pd.DataFrame()

def find_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    files = list_files(directory, pattern)
    
    if not files:
        return None
    
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    
    return files[0]

def get_latest_data_file(symbol: str, timeframe: str, data_dir: str = "data/raw") -> Optional[str]:
    symbol_dir = os.path.join(data_dir, symbol.replace("/", "-"))
    timeframe_dir = os.path.join(symbol_dir, timeframe)
    
    if not os.path.exists(timeframe_dir):
        logger.warning(f"ไม่พบไดเรกทอรี {timeframe_dir}")
        return None
    
    combined_files = list_files(timeframe_dir, "*combined*")
    
    if combined_files:
        parquet_files = [f for f in combined_files if f.endswith('.parquet')]
        if parquet_files:
            return parquet_files[0]
        return combined_files[0]
    
    return find_latest_file(timeframe_dir)
