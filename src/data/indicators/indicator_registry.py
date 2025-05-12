import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from src.data.indicators.basic_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_sma, calculate_ema, calculate_atr
)

from src.data.indicators.advanced_indicators import (
    calculate_stochastic, calculate_obv, calculate_vwap,
    calculate_fibonacci_retracement, calculate_ichimoku_cloud,
    calculate_parabolic_sar, calculate_adx, calculate_momentum,
    calculate_chaikin_money_flow
)

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    _instance = None
    
    @classmethod
    def get_instance(cls, config=None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config=None):
        self.config = config
        self.indicator_config = {}
        if self.config is not None:
            self.indicator_config = self.config.extract_subconfig("indicators", {})
        self.register_indicators()
    
    def register_indicators(self):
        self.indicator_functions = {
            "rsi": calculate_rsi,
            "macd": calculate_macd,
            "bollinger_bands": calculate_bollinger_bands,
            "ema": calculate_ema,
            "sma": calculate_sma,
            "atr": calculate_atr,
            "stochastic": calculate_stochastic,
            "obv": calculate_obv,
            "vwap": calculate_vwap,
            "fibonacci_retracement": calculate_fibonacci_retracement,
            "ichimoku_cloud": calculate_ichimoku_cloud,
            "parabolic_sar": calculate_parabolic_sar,
            "adx": calculate_adx,
            "momentum": calculate_momentum,
            "chaikin_money_flow": calculate_chaikin_money_flow,
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
        }
        
        if self.indicator_config:
            for key, value in self.indicator_config.items():
                if key in self.indicator_default_params:
                    self.indicator_default_params[key].update(value)
    
    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        indicator_name = indicator_name.lower()
        
        if indicator_name not in self.indicator_functions:
            logger.warning(f"ไม่รองรับตัวชี้วัด: {indicator_name}")
            return df
        
        try:
            params = self.indicator_default_params.get(indicator_name, {}).copy()
            params.update(kwargs)
            
            result_df = self.indicator_functions[indicator_name](df, **params)
            
            if not isinstance(result_df, pd.DataFrame):
                logger.warning(f"ฟังก์ชันคำนวณตัวชี้วัด {indicator_name} ไม่คืนค่าเป็น DataFrame")
                return df
            
            new_columns = [col for col in result_df.columns if col not in df.columns]
            if not new_columns:
                logger.warning(f"ตัวชี้วัด {indicator_name} ไม่ได้เพิ่มคอลัมน์ใหม่")
                return df
            
            result = df.copy()
            for col in new_columns:
                result[col] = result_df[col]
            
            logger.info(f"คำนวณตัวชี้วัด {indicator_name} แล้ว, เพิ่ม {len(new_columns)} คอลัมน์: {new_columns}")
            
            return result
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณตัวชี้วัด {indicator_name}: {e}")
            return df
    
    def calculate_indicators(self, df: pd.DataFrame, indicators: List[str] = None, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df
        
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"DataFrame ขาดคอลัมน์ที่จำเป็น: {missing_columns}")
            return df
        
        if indicators is None:
            indicators = self.indicator_config.get("default_indicators", [])
        
        result_df = df.copy()
        
        for indicator in indicators:
            result_df = self.calculate_indicator(result_df, indicator, **kwargs)
        
        return result_df

    def get_supported_indicators(self) -> List[str]:
        return list(self.indicator_functions.keys())
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        indicator_name = indicator_name.lower()
        
        if indicator_name not in self.indicator_default_params:
            logger.warning(f"ไม่พบพารามิเตอร์เริ่มต้นสำหรับตัวชี้วัด: {indicator_name}")
            return {}
        
        return self.indicator_default_params[indicator_name].copy()
