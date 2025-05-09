"""
โมดูลข้อมูลสำหรับ CRYPPO (Cryptocurrency Position Optimization)

โมดูลนี้ประกอบด้วยคลาสและฟังก์ชันสำหรับการดาวน์โหลด ประมวลผล และจัดการข้อมูลตลาดคริปโต
"""

# API
from src.data.api.binance_api import BinanceAPI

# Downloaders
from src.data.downloaders.base_downloader import BaseDownloader
from src.data.downloaders.binance_downloader import BinanceDownloader

# Processors
from src.data.processors.data_processor import DataProcessor

# Managers
from src.data.managers.data_manager import MarketDataManager

# Indicators
from src.data.indicators.indicator_registry import TechnicalIndicators
from src.data.indicators.basic_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_sma, calculate_ema, calculate_atr
)
from src.data.indicators.advanced_indicators import (
    calculate_stochastic, calculate_obv, calculate_vwap,
    calculate_fibonacci_retracement, calculate_ichimoku_cloud
)

# Transforms
from src.data.transforms.data_transforms import (
    log_transform, inverse_log_transform, z_score_normalize, inverse_z_score,
    min_max_scale, inverse_min_max_scale, rolling_window, create_returns,
    calculate_momentum, calculate_volatility, price_diff, normalize_price_by_reference,
    extract_features, create_lag_features
)
from src.data.transforms.data_cleaning import (
    remove_duplicates, handle_missing_values, remove_outliers,
    fill_missing_timestamps, detect_price_anomalies, correct_price_anomalies,
    ensure_ohlc_integrity, validate_timestamp_order, consolidate_duplicate_timestamps
)

# Utils
from src.data.utils.time_utils import (
    parse_timeframe, get_timeframe_delta, get_timeframe_in_minutes,
    get_timeframe_in_milliseconds, get_pandas_freq, is_timeframe_valid,
    align_timestamp_to_timeframe, convert_to_datetime
)
from src.data.utils.file_utils import (
    ensure_directory_exists, list_files, get_file_info, save_dataframe,
    save_json, load_json, load_dataframe, find_latest_file, get_latest_data_file
)

# รายการรวมที่สามารถ import โดยตรง
__all__ = [
    # API
    'BinanceAPI',
    
    # Downloaders
    'BaseDownloader',
    'BinanceDownloader',
    
    # Processors
    'DataProcessor',
    
    # Managers
    'MarketDataManager',
    
    # Indicators
    'TechnicalIndicators',
    'calculate_rsi', 'calculate_macd', 'calculate_bollinger_bands',
    'calculate_sma', 'calculate_ema', 'calculate_atr',
    'calculate_stochastic', 'calculate_obv', 'calculate_vwap',
    'calculate_fibonacci_retracement', 'calculate_ichimoku_cloud',
    
    # Transforms
    'log_transform', 'inverse_log_transform', 'z_score_normalize', 'inverse_z_score',
    'min_max_scale', 'inverse_min_max_scale', 'rolling_window', 'create_returns',
    'calculate_momentum', 'calculate_volatility', 'price_diff', 'normalize_price_by_reference',
    'extract_features', 'create_lag_features',
    
    # Data Cleaning
    'remove_duplicates', 'handle_missing_values', 'remove_outliers',
    'fill_missing_timestamps', 'detect_price_anomalies', 'correct_price_anomalies',
    'ensure_ohlc_integrity', 'validate_timestamp_order', 'consolidate_duplicate_timestamps',
    
    # Utils
    'parse_timeframe', 'get_timeframe_delta', 'get_timeframe_in_minutes',
    'get_timeframe_in_milliseconds', 'get_pandas_freq', 'is_timeframe_valid',
    'align_timestamp_to_timeframe', 'convert_to_datetime',
    'ensure_directory_exists', 'list_files', 'get_file_info', 'save_dataframe',
    'save_json', 'load_json', 'load_dataframe', 'find_latest_file', 'get_latest_data_file'
]

# Singleton instance ของ TechnicalIndicators
technical_indicators = TechnicalIndicators.get_instance()

def get_binance_downloader(api_key=None, api_secret=None):
    """
    สร้างอินสแตนซ์ของ BinanceDownloader
    
    Parameters:
    api_key (str, optional): Binance API key
    api_secret (str, optional): Binance API secret
    
    Returns:
    BinanceDownloader: อินสแตนซ์ของ BinanceDownloader
    """
    return BinanceDownloader(api_key=api_key, api_secret=api_secret)

def get_data_processor(config=None):
    """
    สร้างอินสแตนซ์ของ DataProcessor
    
    Parameters:
    config (Config, optional): อ็อบเจ็กต์การตั้งค่า
    
    Returns:
    DataProcessor: อินสแตนซ์ของ DataProcessor
    """
    return DataProcessor(config=config)

def get_market_data_manager(
    file_path=None,
    symbol=None,
    base_timeframe=None,
    detail_timeframe=None,
    start_date=None,
    end_date=None,
    batch_size=None,
    window_size=None,
    indicators=None,
    use_gpu=None,
    config=None
):
    """
    สร้างอินสแตนซ์ของ MarketDataManager
    
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
    config (Config, optional): อ็อบเจ็กต์การตั้งค่า
    
    Returns:
    MarketDataManager: อินสแตนซ์ของ MarketDataManager
    """
    return MarketDataManager(
        file_path=file_path,
        symbol=symbol,
        base_timeframe=base_timeframe,
        detail_timeframe=detail_timeframe,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        window_size=window_size,
        indicators=indicators,
        use_gpu=use_gpu,
        config=config
    )