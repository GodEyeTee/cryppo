"""
โมดูลข้อมูลสำหรับ CRYPPO (Cryptocurrency Position Optimization)

โมดูลนี้ประกอบด้วยคลาสและฟังก์ชันสำหรับการดาวน์โหลด ประมวลผล และจัดการข้อมูลตลาดคริปโต
"""

from src.data.binance_downloader import BinanceDownloader
from src.data.data_processor import DataProcessor
from src.data.data_manager import MarketDataManager
from src.data.indicators import TechnicalIndicators, get_technical_indicators

__all__ = [
    'BinanceDownloader',
    'DataProcessor',
    'MarketDataManager',
    'TechnicalIndicators',
    'get_technical_indicators'
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