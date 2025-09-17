from src.data.processors.data_processor import DataProcessor

from src.data.managers.data_manager import MarketDataManager

from src.data.indicators.indicator_registry import TechnicalIndicators
from src.data.indicators.basic_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_sma, calculate_ema, calculate_atr
)
from src.data.indicators.advanced_indicators import (
    calculate_stochastic, calculate_obv, calculate_vwap,
    calculate_fibonacci_retracement, calculate_ichimoku_cloud
)

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

from src.data.utils.file_utils import (
    ensure_directory_exists, list_files, get_file_info, save_dataframe,
    save_json, load_json, load_dataframe, find_latest_file, get_latest_data_file
)

__all__ = [
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
    'ensure_directory_exists', 'list_files', 'get_file_info', 'save_dataframe',
    'save_json', 'load_json', 'load_dataframe', 'find_latest_file', 'get_latest_data_file'
]

# Singleton instance ของ TechnicalIndicators
technical_indicators = TechnicalIndicators.get_instance()

def get_data_processor(config=None):
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
