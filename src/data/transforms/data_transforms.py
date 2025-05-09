"""
โมดูลสำหรับการแปลงข้อมูล (Data Transformation) ใน CRYPPO
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# ตั้งค่า logger
logger = logging.getLogger(__name__)

def log_transform(data: np.ndarray, is_volume_col: Optional[np.ndarray] = None) -> np.ndarray:
    """
    ทำ Log Transform กับข้อมูล
    
    Parameters:
    data (np.ndarray): ข้อมูลที่ต้องการแปลง
    is_volume_col (np.ndarray, optional): mask สำหรับคอลัมน์ volume
    
    Returns:
    np.ndarray: ข้อมูลที่แปลงแล้ว
    """
    # สร้าง array ใหม่สำหรับเก็บข้อมูลหลัง log transform
    log_data = np.zeros_like(data, dtype=np.float64)
    
    # ถ้าไม่ระบุ is_volume_col ให้สร้าง array ที่มีค่า False ทั้งหมด
    if is_volume_col is None:
        is_volume_col = np.zeros(data.shape[1], dtype=bool)
    
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

def inverse_log_transform(data: np.ndarray, is_volume_col: Optional[np.ndarray] = None) -> np.ndarray:
    """
    แปลงกลับจาก Log Transform
    
    Parameters:
    data (np.ndarray): ข้อมูลที่ต้องการแปลงกลับ
    is_volume_col (np.ndarray, optional): mask สำหรับคอลัมน์ volume
    
    Returns:
    np.ndarray: ข้อมูลที่แปลงกลับแล้ว
    """
    # สร้าง array ใหม่สำหรับเก็บข้อมูลหลังแปลงกลับ
    inverse_data = np.zeros_like(data, dtype=np.float64)
    
    # ถ้าไม่ระบุ is_volume_col ให้สร้าง array ที่มีค่า False ทั้งหมด
    if is_volume_col is None:
        is_volume_col = np.zeros(data.shape[1], dtype=bool)
    
    # แปลงกลับจาก Log transform
    for col in range(data.shape[1]):
        if is_volume_col[col]:
            # สำหรับปริมาณ ใช้ expm1
            inverse_data[:, col] = np.expm1(data[:, col])
        else:
            # สำหรับราคา ใช้ exp
            inverse_data[:, col] = np.exp(data[:, col])
    
    return inverse_data

def z_score_normalize(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    ทำ Z-score Normalization กับข้อมูล
    
    Parameters:
    data (np.ndarray): ข้อมูลที่ต้องการนอร์มัลไลซ์
    
    Returns:
    tuple: (ข้อมูลที่นอร์มัลไลซ์แล้ว, สถิติ)
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

def inverse_z_score(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    แปลงกลับจาก Z-score Normalization
    
    Parameters:
    data (np.ndarray): ข้อมูลที่ต้องการแปลงกลับ
    means (np.ndarray): ค่าเฉลี่ยที่ใช้ในการนอร์มัลไลซ์
    stds (np.ndarray): ส่วนเบี่ยงเบนมาตรฐานที่ใช้ในการนอร์มัลไลซ์
    
    Returns:
    np.ndarray: ข้อมูลที่แปลงกลับแล้ว
    """
    # แปลงกลับจาก Z-score
    return data * stds + means

def min_max_scale(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict]:
    """
    ทำ Min-Max Scaling กับข้อมูล
    
    Parameters:
    data (np.ndarray): ข้อมูลที่ต้องการสเกล
    feature_range (tuple): ช่วงค่าเป้าหมาย (min, max)
    
    Returns:
    tuple: (ข้อมูลที่สเกลแล้ว, สถิติ)
    """
    # หาค่าต่ำสุดและสูงสุดของแต่ละคอลัมน์
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    
    # ป้องกันการหารด้วย 0
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    # สเกลข้อมูล
    scaled_data = (data - data_min) / data_range
    
    # ปรับขนาดตามช่วงที่ต้องการ
    min_val, max_val = feature_range
    scaled_data = scaled_data * (max_val - min_val) + min_val
    
    # เก็บสถิติสำหรับการแปลงกลับ
    stats = {
        "data_min": data_min,
        "data_max": data_max,
        "feature_range": feature_range
    }
    
    return scaled_data, stats

def inverse_min_max_scale(
    data: np.ndarray, 
    data_min: np.ndarray, 
    data_max: np.ndarray, 
    feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    แปลงกลับจาก Min-Max Scaling
    
    Parameters:
    data (np.ndarray): ข้อมูลที่ต้องการแปลงกลับ
    data_min (np.ndarray): ค่าต่ำสุดของข้อมูลก่อนสเกล
    data_max (np.ndarray): ค่าสูงสุดของข้อมูลก่อนสเกล
    feature_range (tuple): ช่วงค่าเป้าหมายที่ใช้ในการสเกล
    
    Returns:
    np.ndarray: ข้อมูลที่แปลงกลับแล้ว
    """
    min_val, max_val = feature_range
    
    # แปลงกลับจากช่วงเป้าหมาย
    data_std = (data - min_val) / (max_val - min_val)
    
    # แปลงกลับเป็นช่วงเดิม
    return data_std * (data_max - data_min) + data_min

def rolling_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    สร้าง rolling window จากข้อมูล 2D
    
    Parameters:
    data (np.ndarray): ข้อมูล 2D shape (samples, features)
    window_size (int): ขนาดของ window
    
    Returns:
    np.ndarray: ข้อมูล 3D shape (samples - window_size + 1, window_size, features)
    """
    if window_size <= 0:
        raise ValueError(f"window_size ต้องมีค่ามากกว่า 0 แต่ได้รับ {window_size}")
    
    if len(data.shape) != 2:
        raise ValueError(f"data ต้องเป็น 2D array แต่ได้รับ shape {data.shape}")
    
    samples, features = data.shape
    
    if samples < window_size:
        raise ValueError(f"จำนวนตัวอย่าง ({samples}) น้อยกว่า window_size ({window_size})")
    
    # สร้าง rolling window
    windows = np.zeros((samples - window_size + 1, window_size, features))
    
    for i in range(samples - window_size + 1):
        windows[i] = data[i:i+window_size]
    
    return windows

def create_returns(data: np.ndarray, method: str = 'log') -> np.ndarray:
    """
    คำนวณผลตอบแทน (returns) จากข้อมูลราคา
    
    Parameters:
    data (np.ndarray): ข้อมูลราคา
    method (str): วิธีการคำนวณ ("log" หรือ "simple")
    
    Returns:
    np.ndarray: ผลตอบแทน
    """
    if len(data) <= 1:
        return np.zeros_like(data)
    
    if method.lower() == 'log':
        # ผลตอบแทนแบบ logarithmic
        clean_data = np.copy(data)
        clean_data[clean_data <= 0] = 1e-8  # ป้องกันการ log 0 หรือค่าลบ
        
        # log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1})
        log_data = np.log(clean_data)
        returns = np.diff(log_data, axis=0)
        
        # เพิ่ม 0 สำหรับแถวแรก
        returns = np.vstack([np.zeros((1, data.shape[1])), returns])
        
    elif method.lower() == 'simple':
        # ผลตอบแทนแบบ simple
        # (p_t - p_{t-1}) / p_{t-1}
        shifted_data = np.roll(data, 1, axis=0)
        returns = (data - shifted_data) / shifted_data
        
        # แทนค่าในแถวแรกด้วย 0
        returns[0] = 0
        
    else:
        raise ValueError(f"method ไม่รองรับ: {method}. รองรับเฉพาะ 'log' หรือ 'simple'")
    
    # แทนค่า NaN และ Inf ด้วย 0
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    return returns

def calculate_momentum(data: np.ndarray, periods: List[int] = [1, 5, 10, 20]) -> np.ndarray:
    """
    คำนวณ momentum ของข้อมูลราคา
    
    Parameters:
    data (np.ndarray): ข้อมูลราคา
    periods (list): รายการช่วงเวลาที่ต้องการคำนวณ
    
    Returns:
    np.ndarray: momentum
    """
    n_samples = len(data)
    n_features = data.shape[1]
    n_periods = len(periods)
    
    # สร้าง array สำหรับเก็บ momentum
    momentum = np.zeros((n_samples, n_features * n_periods))
    
    for i, period in enumerate(periods):
        if period >= n_samples:
            continue
            
        # คำนวณ momentum: (p_t - p_{t-period}) / p_{t-period}
        shifted_data = np.roll(data, period, axis=0)
        
        # คำนวณ momentum สำหรับช่วงเวลานี้
        mom = (data - shifted_data) / shifted_data
        
        # แทนค่าในแถวแรกๆ ด้วย 0
        mom[:period] = 0
        
        # เก็บ momentum ลงใน array ผลลัพธ์
        momentum[:, i*n_features:(i+1)*n_features] = mom
    
    # แทนค่า NaN และ Inf ด้วย 0
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    
    return momentum

def calculate_volatility(data: np.ndarray, window_sizes: List[int] = [10, 20, 50]) -> np.ndarray:
    """
    คำนวณความผันผวน (volatility) ของข้อมูลราคา
    
    Parameters:
    data (np.ndarray): ข้อมูลราคา
    window_sizes (list): รายการขนาดหน้าต่างที่ต้องการคำนวณ
    
    Returns:
    np.ndarray: ความผันผวน
    """
    n_samples = len(data)
    n_features = data.shape[1]
    n_windows = len(window_sizes)
    
    # สร้าง array สำหรับเก็บความผันผวน
    volatility = np.zeros((n_samples, n_features * n_windows))
    
    # คำนวณผลตอบแทนแบบ log
    returns = create_returns(data, method='log')
    
    for i, window in enumerate(window_sizes):
        if window >= n_samples:
            continue
            
        # คำนวณความผันผวนโดยใช้ rolling window
        vol = np.zeros_like(returns)
        
        for j in range(window, n_samples):
            # ใช้ส่วนเบี่ยงเบนมาตรฐานของผลตอบแทนในหน้าต่าง
            vol[j] = np.std(returns[j-window:j], axis=0)
        
        # เก็บความผันผวนลงใน array ผลลัพธ์
        volatility[:, i*n_features:(i+1)*n_features] = vol
    
    return volatility

def price_diff(data: np.ndarray, periods: List[int] = [1, 5, 10, 20]) -> np.ndarray:
    """
    คำนวณความแตกต่างของราคา (price difference)
    
    Parameters:
    data (np.ndarray): ข้อมูลราคา
    periods (list): รายการช่วงเวลาที่ต้องการคำนวณ
    
    Returns:
    np.ndarray: ความแตกต่างของราคา
    """
    n_samples = len(data)
    n_features = data.shape[1]
    n_periods = len(periods)
    
    # สร้าง array สำหรับเก็บความแตกต่างของราคา
    diff = np.zeros((n_samples, n_features * n_periods))
    
    for i, period in enumerate(periods):
        if period >= n_samples:
            continue
            
        # คำนวณความแตกต่างของราคา: p_t - p_{t-period}
        shifted_data = np.roll(data, period, axis=0)
        
        # คำนวณความแตกต่างสำหรับช่วงเวลานี้
        d = data - shifted_data
        
        # แทนค่าในแถวแรกๆ ด้วย 0
        d[:period] = 0
        
        # เก็บความแตกต่างลงใน array ผลลัพธ์
        diff[:, i*n_features:(i+1)*n_features] = d
    
    return diff

def normalize_price_by_reference(
    data: np.ndarray, 
    reference_idx: int = 0, 
    feature_idx: Optional[List[int]] = None
) -> np.ndarray:
    """
    นอร์มัลไลซ์ราคาด้วยค่าอ้างอิง
    
    Parameters:
    data (np.ndarray): ข้อมูลราคา 3D (samples, time_steps, features)
    reference_idx (int): ดัชนีของเวลาที่ใช้เป็นอ้างอิง
    feature_idx (list, optional): รายการดัชนีของคุณลักษณะที่ต้องการนอร์มัลไลซ์
    
    Returns:
    np.ndarray: ข้อมูลที่นอร์มัลไลซ์แล้ว
    """
    if len(data.shape) != 3:
        raise ValueError(f"data ต้องเป็น 3D array แต่ได้รับ shape {data.shape}")
    
    samples, time_steps, features = data.shape
    
    if reference_idx < 0 or reference_idx >= time_steps:
        raise ValueError(f"reference_idx ต้องอยู่ในช่วง [0, {time_steps-1}] แต่ได้รับ {reference_idx}")
    
    # สร้าง array สำหรับเก็บข้อมูลที่นอร์มัลไลซ์แล้ว
    normalized_data = np.copy(data)
    
    # ถ้าไม่ระบุ feature_idx ให้นอร์มัลไลซ์ทุกคุณลักษณะ
    if feature_idx is None:
        feature_idx = list(range(features))
    
    for i in feature_idx:
        if i < 0 or i >= features:
            continue
            
        # นอร์มัลไลซ์โดยหารด้วยค่าอ้างอิง
        ref_values = data[:, reference_idx, i].reshape(-1, 1, 1)
        
        # ป้องกันการหารด้วย 0
        ref_values[ref_values == 0] = 1.0
        
        # นอร์มัลไลซ์เฉพาะคุณลักษณะที่ระบุ
        normalized_data[:, :, i] = data[:, :, i] / ref_values
    
    return normalized_data

def extract_features(data: np.ndarray, feature_idx: List[int]) -> np.ndarray:
    """
    ดึงคุณลักษณะที่ต้องการจากข้อมูล
    
    Parameters:
    data (np.ndarray): ข้อมูล
    feature_idx (list): รายการดัชนีของคุณลักษณะที่ต้องการ
    
    Returns:
    np.ndarray: ข้อมูลที่มีเฉพาะคุณลักษณะที่ต้องการ
    """
    if len(data.shape) == 2:
        # ข้อมูล 2D (samples, features)
        return data[:, feature_idx]
    
    elif len(data.shape) == 3:
        # ข้อมูล 3D (samples, time_steps, features)
        return data[:, :, feature_idx]
    
    else:
        raise ValueError(f"data ต้องเป็น 2D หรือ 3D array แต่ได้รับ shape {data.shape}")

def create_lag_features(data: np.ndarray, lag_periods: List[int] = [1, 2, 3]) -> np.ndarray:
    """
    สร้างคุณลักษณะ lag (ข้อมูลย้อนหลัง)
    
    Parameters:
    data (np.ndarray): ข้อมูล 2D (samples, features)
    lag_periods (list): รายการช่วงเวลา lag ที่ต้องการ
    
    Returns:
    np.ndarray: ข้อมูลที่รวมคุณลักษณะ lag
    """
    if len(data.shape) != 2:
        raise ValueError(f"data ต้องเป็น 2D array แต่ได้รับ shape {data.shape}")
    
    n_samples, n_features = data.shape
    n_lags = len(lag_periods)
    
    # สร้าง array สำหรับเก็บข้อมูลที่รวมคุณลักษณะ lag
    lagged_data = np.zeros((n_samples, n_features * (1 + n_lags)))
    
    # คัดลอกข้อมูลต้นฉบับ
    lagged_data[:, :n_features] = data
    
    for i, lag in enumerate(lag_periods):
        if lag >= n_samples:
            continue
            
        # สร้างข้อมูล lag
        lagged = np.zeros_like(data)
        lagged[lag:] = data[:-lag]
        
        # เก็บข้อมูล lag ลงใน array ผลลัพธ์
        lagged_data[:, (i+1)*n_features:(i+2)*n_features] = lagged
    
    return lagged_data