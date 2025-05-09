"""
ระบบทะเบียนตัวชี้วัดสำหรับ CRYPPO (Cryptocurrency Position Optimization)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# นำเข้าตัวชี้วัดพื้นฐานและขั้นสูง
from src.data.indicators.basic_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_sma, calculate_ema, calculate_atr
)

from src.data.indicators.advanced_indicators import (
    calculate_stochastic, calculate_obv, calculate_vwap,
    calculate_fibonacci_retracement, calculate_ichimoku_cloud,
    calculate_parabolic_sar, calculate_adx, calculate_volume_profile,
    calculate_moving_average_crossover, calculate_pivot_points,
    calculate_chaikin_money_flow, calculate_momentum
)

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    คลาสสำหรับการจัดการและคำนวณตัวชี้วัดทางเทคนิคต่างๆ
    
    รองรับตัวชี้วัดพื้นฐานและตัวชี้วัดขั้นสูงหลายประเภท
    สามารถเรียกใช้แบบเป็น singleton หรือสร้างอินสแตนซ์ใหม่ก็ได้
    """
    
    # singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, config=None):
        """
        ดึง singleton instance ของ TechnicalIndicators
        
        Parameters:
        config: อ็อบเจ็กต์การตั้งค่า (optional)
        
        Returns:
        TechnicalIndicators: อินสแตนซ์ของ TechnicalIndicators
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config=None):
        """
        กำหนดค่าเริ่มต้นสำหรับ TechnicalIndicators
        
        Parameters:
        config: อ็อบเจ็กต์การตั้งค่า (optional)
        """
        self.config = config
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        self.indicator_config = {}
        if self.config is not None:
            self.indicator_config = self.config.extract_subconfig("indicators", {})
        
        # ลงทะเบียนตัวชี้วัด
        self.register_indicators()
    
    def register_indicators(self):
        """
        ลงทะเบียนตัวชี้วัดที่รองรับทั้งหมด
        """
        # ฟังก์ชันสำหรับคำนวณตัวชี้วัดแต่ละประเภท
        self.indicator_functions = {
            # ตัวชี้วัดพื้นฐาน
            "rsi": calculate_rsi,
            "macd": calculate_macd,
            "bollinger_bands": calculate_bollinger_bands,
            "ema": calculate_ema,
            "sma": calculate_sma,
            "atr": calculate_atr,
            
            # ตัวชี้วัดขั้นสูง
            "stochastic": calculate_stochastic,
            "obv": calculate_obv,
            "vwap": calculate_vwap,
            "fibonacci_retracement": calculate_fibonacci_retracement,
            "ichimoku_cloud": calculate_ichimoku_cloud,
            "parabolic_sar": calculate_parabolic_sar,
            "adx": calculate_adx,
            "volume_profile": calculate_volume_profile,
            "moving_average_crossover": calculate_moving_average_crossover,
            "pivot_points": calculate_pivot_points,
            "chaikin_money_flow": calculate_chaikin_money_flow,
            "momentum": calculate_momentum,
            
            # ตัวชี้วัดที่ใช้ค่าเริ่มต้น
            "hull_moving_average": lambda df: df,
            "keltner_channels": lambda df: df,
            "price_rate_of_change": lambda df: df,
            "bears_vs_bulls": lambda df: df,
            "williams_r": lambda df: df,
            "standard_deviation": lambda df: df,
            "trix": lambda df: df,
            "average_directional_index": lambda df: df,
            "money_flow_index": lambda df: df,
            "accumulation_distribution_line": lambda df: df,
            "time_segmented_volume": lambda df: df,
            "relative_volume": lambda df: df
        }
        
        # การตั้งค่าเริ่มต้นของตัวชี้วัด
        self.indicator_default_params = {
            "rsi": {"period": 14},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "bollinger_bands": {"period": 20, "std_dev": 2.0},
            "ema": {"periods": [9, 21, 50, 200]},
            "sma": {"periods": [10, 50, 200]},
            "atr": {"period": 14},
            "stochastic": {"k_period": 14, "d_period": 3, "slowing": 3},
            "vwap": {},
            "obv": {},
            # ตัวชี้วัดขั้นสูงอื่นๆ...
        }
        
        # อัพเดตการตั้งค่าจาก config ถ้ามี
        if self.indicator_config:
            for key, value in self.indicator_config.items():
                if key in self.indicator_default_params:
                    self.indicator_default_params[key].update(value)
    
    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """
        คำนวณตัวชี้วัดที่ระบุ
        
        Parameters:
        df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
        indicator_name (str): ชื่อตัวชี้วัดที่ต้องการคำนวณ
        **kwargs: พารามิเตอร์เพิ่มเติมสำหรับการคำนวณตัวชี้วัด
        
        Returns:
        pd.DataFrame: DataFrame ที่มีคอลัมน์ของตัวชี้วัดเพิ่มเติม
        """
        indicator_name = indicator_name.lower()
        
        if indicator_name not in self.indicator_functions:
            logger.warning(f"ไม่รองรับตัวชี้วัด: {indicator_name}")
            return df
        
        try:
            # รวมพารามิเตอร์เริ่มต้นกับพารามิเตอร์ที่ระบุ
            params = self.indicator_default_params.get(indicator_name, {}).copy()
            params.update(kwargs)
            
            # เรียกใช้ฟังก์ชันคำนวณตัวชี้วัด
            result_df = self.indicator_functions[indicator_name](df, **params)
            
            # ถ้าผลลัพธ์ไม่ใช่ DataFrame ให้สร้าง DataFrame ใหม่
            if not isinstance(result_df, pd.DataFrame):
                logger.warning(f"ฟังก์ชันคำนวณตัวชี้วัด {indicator_name} ไม่คืนค่าเป็น DataFrame")
                return df
            
            # ตรวจสอบว่าตัวชี้วัดได้เพิ่มคอลัมน์ใหม่หรือไม่
            new_columns = [col for col in result_df.columns if col not in df.columns]
            if not new_columns:
                logger.warning(f"ตัวชี้วัด {indicator_name} ไม่ได้เพิ่มคอลัมน์ใหม่")
                return df
            
            # สร้าง DataFrame ผลลัพธ์โดยรวมคอลัมน์ของ df และคอลัมน์ใหม่จาก result_df
            result = df.copy()
            for col in new_columns:
                result[col] = result_df[col]
            
            logger.info(f"คำนวณตัวชี้วัด {indicator_name} แล้ว, เพิ่ม {len(new_columns)} คอลัมน์: {new_columns}")
            
            return result
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณตัวชี้วัด {indicator_name}: {e}")
            return df
    
    def calculate_indicators(self, df: pd.DataFrame, indicators: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        คำนวณตัวชี้วัดหลายตัวพร้อมกัน
        
        Parameters:
        df (pd.DataFrame): DataFrame ที่มีข้อมูลตลาด
        indicators (list): รายการชื่อตัวชี้วัดที่ต้องการคำนวณ
        **kwargs: พารามิเตอร์เพิ่มเติมสำหรับการคำนวณตัวชี้วัด
        
        Returns:
        pd.DataFrame: DataFrame ที่มีคอลัมน์ของตัวชี้วัดเพิ่มเติม
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
        
        # คำนวณตัวชี้วัดทีละตัว
        result_df = df.copy()
        
        for indicator in indicators:
            result_df = self.calculate_indicator(result_df, indicator, **kwargs)
        
        return result_df

    def get_supported_indicators(self) -> List[str]:
        """
        ดึงรายการตัวชี้วัดที่รองรับทั้งหมด
        
        Returns:
        list: รายการชื่อตัวชี้วัดที่รองรับ
        """
        return list(self.indicator_functions.keys())
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """
        ดึงพารามิเตอร์เริ่มต้นของตัวชี้วัด
        
        Parameters:
        indicator_name (str): ชื่อตัวชี้วัด
        
        Returns:
        dict: พารามิเตอร์เริ่มต้นของตัวชี้วัด
        """
        indicator_name = indicator_name.lower()
        
        if indicator_name not in self.indicator_default_params:
            logger.warning(f"ไม่พบพารามิเตอร์เริ่มต้นสำหรับตัวชี้วัด: {indicator_name}")
            return {}
        
        return self.indicator_default_params[indicator_name].copy()