import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    original_size = len(df)
    
    if subset is None and 'timestamp' in df.columns:
        subset = ['timestamp']
        
    df_clean = df.drop_duplicates(subset=subset).reset_index(drop=True)

    removed_rows = original_size - len(df_clean)
    if removed_rows > 0:
        logger.info(f"ลบข้อมูลซ้ำแล้ว {removed_rows} แถว ({removed_rows/original_size:.2%} ของข้อมูลทั้งหมด)")
    
    return df_clean

def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'ffill', 
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        return df

    df_clean = df.copy()
    if strategy == 'ffill':
        df_clean = df_clean.ffill()
        df_clean = df_clean.bfill()
    
    elif strategy == 'bfill':
        df_clean = df_clean.bfill()
        df_clean = df_clean.ffill()
    
    elif strategy == 'value':
        if fill_value is None:
            fill_value = 0
        
        df_clean = df_clean.fillna(fill_value)
    
    elif strategy == 'mean':
        for col in df_clean.select_dtypes(include=['number']).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        df_clean = df_clean.ffill().bfill()

    elif strategy == 'median':
        for col in df_clean.select_dtypes(include=['number']).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        df_clean = df_clean.ffill().bfill()
    elif strategy == 'mode':
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan)
        df_clean = df_clean.ffill().bfill()
    
    else:
        raise ValueError(f"กลยุทธ์ไม่รองรับ: {strategy}")

    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"ยังมีค่าที่หายไป {remaining_missing} ค่าหลังจากใช้กลยุทธ์ {strategy}")
    else:
        logger.info(f"เติมค่าที่หายไปทั้งหมด {missing_count} ค่าแล้ว โดยใช้กลยุทธ์ {strategy}")
    return df_clean
def remove_outliers(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    method: str = 'zscore', 
    threshold: float = 3.0
) -> pd.DataFrame:
    df_clean = df.copy()
    if columns is None:
        exclude_columns = ['timestamp', 'date', 'time', 'index']
        columns = [col for col in df_clean.select_dtypes(include=['number']).columns
                  if col not in exclude_columns]
    outlier_count = 0
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if method == 'zscore':
            # ใช้ Z-score
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            if std == 0:
                continue
            
            z_scores = (df_clean[col] - mean) / std
            outliers = (z_scores.abs() > threshold)
            
        elif method == 'iqr':
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            
        else:
            raise ValueError(f"วิธีการไม่รองรับ: {method}")

        if outliers.sum() > 0:
            outlier_count += outliers.sum()
            df_clean.loc[outliers, col] = mean if method == 'zscore' else df_clean[col].median()
    
    if outlier_count > 0:
        logger.info(f"กำจัด outliers แล้ว {outlier_count} ค่า โดยใช้วิธี {method}")
    
    return df_clean

def fill_missing_timestamps(
    df: pd.DataFrame, 
    freq: str = '1min', 
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame ไม่มีคอลัมน์ 'timestamp'")
    
    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df_indexed = df.set_index('timestamp')

    full_range = pd.date_range(start=df_indexed.index.min(), end=df_indexed.index.max(), freq=freq)

    df_filled = df_indexed.reindex(full_range)
    
    # เติมข้อมูลที่หายไป
    if fill_method == 'ffill':
        df_filled = df_filled.ffill()
        df_filled = df_filled.bfill()
    
    elif fill_method == 'bfill':
        df_filled = df_filled.bfill()
        df_filled = df_filled.ffill()
    
    elif fill_method == 'value':
        if 'open' in df_filled.columns:
            df_filled['open'] = df_filled['open'].ffill()
        
        if 'high' in df_filled.columns:
            df_filled['high'] = df_filled['high'].ffill()
        
        if 'low' in df_filled.columns:
            df_filled['low'] = df_filled['low'].ffill()
        
        if 'close' in df_filled.columns:
            df_filled['close'] = df_filled['close'].ffill()
        
        if 'volume' in df_filled.columns:
            df_filled['volume'] = df_filled['volume'].fillna(0)
        
        # สำหรับคอลัมน์อื่นๆ ให้เติมด้วย ffill
        for col in df_filled.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                df_filled[col] = df_filled[col].ffill().bfill()
    
    else:
        raise ValueError(f"วิธีการเติมข้อมูลไม่รองรับ: {fill_method}")
    
    # รีเซ็ตดัชนีและเพิ่มคอลัมน์ timestamp กลับเข้าไป
    df_filled = df_filled.reset_index().rename(columns={'index': 'timestamp'})
    
    # แสดงจำนวนแถวที่เพิ่มขึ้น
    added_rows = len(df_filled) - len(df)
    if added_rows > 0:
        logger.info(f"เติมข้อมูลที่หายไปแล้ว {added_rows} แถว")
    
    return df_filled

def detect_price_anomalies(
    df: pd.DataFrame, 
    price_column: str = 'close', 
    method: str = 'diff', 
    threshold: float = 0.1, 
    window_size: int = 5
) -> pd.DataFrame:
    if price_column not in df.columns:
        raise ValueError(f"ไม่พบคอลัมน์ราคา: {price_column}")
    df_anomaly = df.copy()
    if method == 'diff':
        price_diff = df_anomaly[price_column].diff().abs()
        anomalies = price_diff > threshold
    
    elif method == 'percent':
        price_pct_change = df_anomaly[price_column].pct_change().abs()
        anomalies = price_pct_change > threshold
    
    elif method == 'zscore':
        # ใช้ Z-score
        price = df_anomaly[price_column]
        rolling_mean = price.rolling(window=window_size).mean()
        rolling_std = price.rolling(window=window_size).std()
        
        # ป้องกันการหารด้วย 0
        rolling_std = rolling_std.replace(0, np.nan)
        
        z_scores = (price - rolling_mean) / rolling_std
        anomalies = z_scores.abs() > threshold
        
        # แทนที่ NaN ด้วย False (สำหรับช่วงต้นของข้อมูล)
        anomalies = anomalies.fillna(False)
    
    else:
        raise ValueError(f"วิธีการไม่รองรับ: {method}")
    
    # เพิ่มคอลัมน์แสดงความผิดปกติ
    df_anomaly['is_anomaly'] = anomalies
    
    # แสดงจำนวนความผิดปกติที่พบ
    anomaly_count = anomalies.sum()
    if anomaly_count > 0:
        logger.info(f"พบความผิดปกติของราคา {anomaly_count} ครั้ง โดยใช้วิธี {method}")
    
    return df_anomaly

def correct_price_anomalies(
    df: pd.DataFrame, 
    anomaly_column: str = 'is_anomaly', 
    price_columns: List[str] = ['open', 'high', 'low', 'close'], 
    method: str = 'median'
) -> pd.DataFrame:
    if anomaly_column not in df.columns:
        raise ValueError(f"ไม่พบคอลัมน์แสดงความผิดปกติ: {anomaly_column}")
    
    anomaly_count = df[anomaly_column].sum()
    if anomaly_count == 0:
        return df
    
    df_corrected = df.copy()
    for col in price_columns:
        if col not in df_corrected.columns:
            continue
        
        if method == 'median':
            # ใช้ค่ามัธยฐานของหน้าต่าง
            window_size = 5
            for idx in df_corrected.index[df_corrected[anomaly_column]]:
                # หาช่วงของหน้าต่าง
                start_idx = max(0, idx - window_size)
                end_idx = min(len(df_corrected) - 1, idx + window_size)
                
                # ดึงข้อมูลในหน้าต่าง (ไม่รวมค่าที่ผิดปกติ)
                window_data = df_corrected.loc[start_idx:end_idx, col].copy()
                window_data.loc[df_corrected.loc[start_idx:end_idx, anomaly_column]] = np.nan
                
                # แทนที่ค่าที่ผิดปกติด้วยค่ามัธยฐาน
                if not window_data.dropna().empty:
                    df_corrected.loc[idx, col] = window_data.dropna().median()
        
        elif method == 'mean':
            # ใช้ค่าเฉลี่ยของหน้าต่าง
            window_size = 5
            for idx in df_corrected.index[df_corrected[anomaly_column]]:
                # หาช่วงของหน้าต่าง
                start_idx = max(0, idx - window_size)
                end_idx = min(len(df_corrected) - 1, idx + window_size)
                
                # ดึงข้อมูลในหน้าต่าง (ไม่รวมค่าที่ผิดปกติ)
                window_data = df_corrected.loc[start_idx:end_idx, col].copy()
                window_data.loc[df_corrected.loc[start_idx:end_idx, anomaly_column]] = np.nan
                
                # แทนที่ค่าที่ผิดปกติด้วยค่าเฉลี่ย
                if not window_data.dropna().empty:
                    df_corrected.loc[idx, col] = window_data.dropna().mean()
        
        elif method == 'interp':
            # ใช้การประมาณค่าระหว่างจุด
            # แทนที่ค่าที่ผิดปกติด้วย NaN
            df_corrected.loc[df_corrected[anomaly_column], col] = np.nan
            
            # ประมาณค่าระหว่างจุด
            df_corrected[col] = df_corrected[col].interpolate(method='linear')
            
            # กรณีที่มีค่า NaN ที่ขอบ ให้เติมด้วยค่าใกล้เคียง
            df_corrected[col] = df_corrected[col].ffill().bfill()
        
        elif method == 'previous':
            # ใช้ค่าก่อนหน้า
            # แทนที่ค่าที่ผิดปกติด้วย NaN
            df_corrected.loc[df_corrected[anomaly_column], col] = np.nan
            
            # เติมด้วยค่าก่อนหน้า
            df_corrected[col] = df_corrected[col].ffill().bfill()
        
        else:
            raise ValueError(f"วิธีการแก้ไขไม่รองรับ: {method}")
    
    logger.info(f"แก้ไขความผิดปกติแล้ว {anomaly_count} ครั้ง โดยใช้วิธี {method}")
    
    return df_corrected

def ensure_ohlc_integrity(df: pd.DataFrame) -> pd.DataFrame:
    ohlc_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in ohlc_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"ไม่พบคอลัมน์ {missing_columns} จึงไม่สามารถตรวจสอบความถูกต้องของข้อมูล OHLC ได้")
        return df
    
    # คัดลอก DataFrame
    df_fixed = df.copy()
    
    # ตรวจสอบและแก้ไขความถูกต้อง
    violations = 0
    
    # กฎที่ 1: high ต้องมีค่ามากกว่าหรือเท่ากับ open, low, close
    high_violations = (
        (df_fixed['high'] < df_fixed['open']) |
        (df_fixed['high'] < df_fixed['low']) |
        (df_fixed['high'] < df_fixed['close'])
    )
    
    if high_violations.any():
        # แก้ไขโดยตั้ง high ให้เป็นค่าสูงสุดของ open, high, low, close
        for idx in df_fixed.index[high_violations]:
            df_fixed.loc[idx, 'high'] = max(
                df_fixed.loc[idx, 'open'],
                df_fixed.loc[idx, 'high'],
                df_fixed.loc[idx, 'low'],
                df_fixed.loc[idx, 'close']
            )
        
        violations += high_violations.sum()
    
    # กฎที่ 2: low ต้องมีค่าน้อยกว่าหรือเท่ากับ open, high, close
    low_violations = (
        (df_fixed['low'] > df_fixed['open']) |
        (df_fixed['low'] > df_fixed['high']) |
        (df_fixed['low'] > df_fixed['close'])
    )
    
    if low_violations.any():
        # แก้ไขโดยตั้ง low ให้เป็นค่าต่ำสุดของ open, high, low, close
        for idx in df_fixed.index[low_violations]:
            df_fixed.loc[idx, 'low'] = min(
                df_fixed.loc[idx, 'open'],
                df_fixed.loc[idx, 'high'],
                df_fixed.loc[idx, 'low'],
                df_fixed.loc[idx, 'close']
            )
        
        violations += low_violations.sum()
    
    if violations > 0:
        logger.info(f"แก้ไขความถูกต้องของข้อมูล OHLC แล้ว {violations} ครั้ง")
    
    return df_fixed

def validate_timestamp_order(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
    if timestamp_column not in df.columns:
        logger.warning(f"ไม่พบคอลัมน์ {timestamp_column} จึงไม่สามารถตรวจสอบลำดับ timestamp ได้")
        return df
    
    # ตรวจสอบประเภทข้อมูลของคอลัมน์ timestamp
    if not pd.api.types.is_datetime64_dtype(df[timestamp_column]):
        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # ตรวจสอบว่าเรียงลำดับถูกต้องหรือไม่
    if df[timestamp_column].is_monotonic_increasing:
        return df
    
    # เรียงลำดับตาม timestamp
    df_sorted = df.sort_values(timestamp_column).reset_index(drop=True)
    
    logger.info("เรียงลำดับ timestamp ใหม่แล้ว")
    
    return df_sorted

def consolidate_duplicate_timestamps(
    df: pd.DataFrame, 
    timestamp_column: str = 'timestamp', 
    method: str = 'last'
) -> pd.DataFrame:
    if timestamp_column not in df.columns:
        logger.warning(f"ไม่พบคอลัมน์ {timestamp_column} จึงไม่สามารถรวมแถวที่มี timestamp ซ้ำกันได้")
        return df
    
    # ตรวจสอบว่ามี timestamp ซ้ำกันหรือไม่
    duplicate_count = df.duplicated(subset=[timestamp_column], keep=False).sum()
    
    if duplicate_count == 0:
        return df
    
    # รวมแถวที่มี timestamp ซ้ำกัน
    if method == 'last':
        df_consolidated = df.drop_duplicates(subset=[timestamp_column], keep='last')
    
    elif method == 'first':
        df_consolidated = df.drop_duplicates(subset=[timestamp_column], keep='first')
    
    else:
        # กำหนดวิธีการรวมสำหรับแต่ละคอลัมน์
        agg_dict = {}
        
        # สำหรับคอลัมน์ OHLCV
        if 'open' in df.columns:
            agg_dict['open'] = 'first'
        
        if 'high' in df.columns:
            agg_dict['high'] = 'max'
        
        if 'low' in df.columns:
            agg_dict['low'] = 'min'
        
        if 'close' in df.columns:
            agg_dict['close'] = 'last'
        
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        
        # สำหรับคอลัมน์อื่นๆ
        for col in df.columns:
            if col not in [timestamp_column, 'open', 'high', 'low', 'close', 'volume']:
                agg_dict[col] = method
        
        # รวมแถวตามวิธีการที่กำหนด
        df_consolidated = df.groupby(timestamp_column).agg(agg_dict).reset_index()
    
    logger.info(f"รวมแถวที่มี timestamp ซ้ำกันแล้ว {duplicate_count} แถว โดยใช้วิธี {method}")
    
    return df_consolidated
