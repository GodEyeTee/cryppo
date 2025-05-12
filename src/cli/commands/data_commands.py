import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.data.downloaders.binance_downloader import BinanceDownloader
from src.data.processors.data_processor import DataProcessor
from src.data.managers.data_manager import MarketDataManager
from src.utils.config_manager import get_config

logger = logging.getLogger('cli.data')

def setup_download_parser(parser):
    parser.add_argument("--symbol", type=str, required=True, help="คู่สกุลเงิน (เช่น BTCUSDT)")
    parser.add_argument("--timeframes", type=str, required=True, help="ไทม์เฟรม (เช่น 1m,5m,1h คั่นด้วยเครื่องหมายจุลภาค)")
    parser.add_argument("--start", type=str, default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                      help="วันเริ่มต้น (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                      help="วันสิ้นสุด (YYYY-MM-DD, ค่าเริ่มต้น = วันปัจจุบัน)")
    parser.add_argument("--output", type=str, default="data/raw", help="ไดเรกทอรีที่จะบันทึกข้อมูล")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet", "both"],
                      help="รูปแบบไฟล์ที่จะบันทึก")
    parser.add_argument("--api-key", type=str, default=None, help="Binance API key")
    parser.add_argument("--api-secret", type=str, default=None, help="Binance API secret")

def setup_update_parser(parser):
    parser.add_argument("--symbol", type=str, required=True, help="คู่สกุลเงิน (เช่น BTCUSDT)")
    parser.add_argument("--timeframes", type=str, required=True, help="ไทม์เฟรม (เช่น 1m,5m,1h คั่นด้วยเครื่องหมายจุลภาค)")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="ไดเรกทอรีที่เก็บข้อมูล")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet", "both"],
                      help="รูปแบบไฟล์ที่จะบันทึก")
    parser.add_argument("--api-key", type=str, default=None, help="Binance API key")
    parser.add_argument("--api-secret", type=str, default=None, help="Binance API secret")

def setup_process_parser(parser):
    parser.add_argument("--input", type=str, required=True, help="ไฟล์หรือไดเรกทอรีนำเข้า")
    parser.add_argument("--output", type=str, required=True, help="ไฟล์หรือไดเรกทอรีส่งออก")
    parser.add_argument("--indicators", type=str, default=None,
                      help="รายการตัวชี้วัดที่ต้องการคำนวณ (คั่นด้วยเครื่องหมายจุลภาค)")
    parser.add_argument("--no-log-transform", action="store_true", help="ไม่ใช้ Log Transform")
    parser.add_argument("--no-z-score", action="store_true", help="ไม่ใช้ Z-score Normalization")
    parser.add_argument("--no-handle-missing", action="store_true", help="ไม่จัดการกับค่าที่หายไป")
    parser.add_argument("--window-size", type=int, default=None,
                      help="ขนาดของหน้าต่างสำหรับแต่ละตัวอย่าง")
    parser.add_argument("--file-pattern", type=str, default="*.csv",
                      help="รูปแบบไฟล์ที่ต้องการประมวลผล (สำหรับโหมดไดเรกทอรี)")

def setup_analyze_parser(parser):
    parser.add_argument("--input", type=str, required=True, help="ไฟล์ข้อมูลที่ต้องการวิเคราะห์")
    parser.add_argument("--output", type=str, default=None, help="ไฟล์สำหรับบันทึกผลการวิเคราะห์")
    parser.add_argument("--stats", action="store_true", help="แสดงสถิติของข้อมูล")
    parser.add_argument("--correlation", action="store_true", help="แสดงเมทริกซ์สหสัมพันธ์")
    parser.add_argument("--plot", choices=["price", "volume", "returns", "indicators", "all"], 
                      default=None, help="สร้างกราฟ")
    parser.add_argument("--period-stats", choices=["daily", "weekly", "monthly", "hourly"],
                      default=None, help="วิเคราะห์สถิติตามช่วงเวลา")
    parser.add_argument("--columns", type=str, default=None,
                      help="รายการคอลัมน์ที่ต้องการวิเคราะห์ (คั่นด้วยเครื่องหมายจุลภาค)")

def handle_download(args):
    downloader = BinanceDownloader(api_key=args.api_key, api_secret=args.api_secret)
    
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    results = downloader.download_multiple_timeframes(
        symbol=args.symbol,
        timeframes=timeframes,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        file_format=args.format
    )
    
    for timeframe, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"ดาวน์โหลดข้อมูล {args.symbol} ไทม์เฟรม {timeframe} สำเร็จ: {len(df)} แท่งเทียน")
        else:
            logger.warning(f"ไม่พบข้อมูลสำหรับ {args.symbol} ไทม์เฟรม {timeframe}")

def handle_update(args):
    downloader = BinanceDownloader(api_key=args.api_key, api_secret=args.api_secret)
    
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    results = downloader.update_multiple_timeframes(
        symbol=args.symbol,
        timeframes=timeframes,
        data_dir=args.data_dir,
        file_format=args.format
    )
    
    for timeframe, df in results.items():
        if df is not None and not df.empty:
            logger.info(f"อัพเดตข้อมูล {args.symbol} ไทม์เฟรม {timeframe} สำเร็จ: {len(df)} แท่งเทียน")
        else:
            logger.warning(f"ไม่สามารถอัพเดตข้อมูลสำหรับ {args.symbol} ไทม์เฟรม {timeframe} ได้")

def handle_process(args):
    config = get_config()
    processor = DataProcessor(config)
    
    if args.no_log_transform:
        processor.use_log_transform = False
    
    if args.no_z_score:
        processor.use_z_score = False
    
    if args.no_handle_missing:
        processor.handle_missing = False
    
    if args.window_size:
        processor.window_size = args.window_size
    
    indicators = None
    if args.indicators:
        indicators = [indicator.strip() for indicator in args.indicators.split(',')]
    
    path = Path(args.input)
    if path.is_file():
        # ประมวลผลไฟล์เดียว
        df = processor.process_file(str(path), args.output, indicators)
        if df is not None and not df.empty:
            logger.info(f"ประมวลผลไฟล์ {args.input} สำเร็จ: {len(df)} แถว, {len(df.columns)} คอลัมน์")
        else:
            logger.error(f"ไม่สามารถประมวลผลไฟล์ {args.input} ได้")
    elif path.is_dir():
        processed_files = processor.process_directory(
            str(path),
            args.output,
            args.file_pattern,
            indicators
        )
        logger.info(f"ประมวลผลไฟล์ทั้งหมด {len(processed_files)} ไฟล์")
    else:
        logger.error(f"ไม่พบไฟล์หรือไดเรกทอรี: {args.input}")

def handle_analyze(args):
    if not os.path.exists(args.input):
        logger.error(f"ไม่พบไฟล์: {args.input}")
        return
        
    data_manager = MarketDataManager(file_path=args.input)
    
    if not data_manager.data_loaded:
        logger.error(f"ไม่สามารถโหลดข้อมูลจาก {args.input} ได้")
        return
    
    info = data_manager.get_data_info()
    print("\nข้อมูลเกี่ยวกับชุดข้อมูล:")
    for key, value in info.items():
        if key == 'columns':
            print(f"  คอลัมน์: {', '.join(value[:10])}{' และอีก ' + str(len(value) - 10) + ' คอลัมน์' if len(value) > 10 else ''}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # แสดงสถิติ
    if args.stats:
        price_stats = data_manager.get_timeseries_stats('close')
        print("\nสถิติของราคา:")
        for key, value in price_stats.items():
            if key != 'autocorrelation' and key != 'adf_test':
                print(f"  {key}: {value}")
        
        volume_stats = data_manager.get_timeseries_stats('volume')
        print("\nสถิติของปริมาณการซื้อขาย:")
        for key, value in volume_stats.items():
            if key != 'autocorrelation' and key != 'adf_test':
                print(f"  {key}: {value}")
    
    if args.correlation:
        # แยกคอลัมน์
        columns = None
        if args.columns:
            columns = [col.strip() for col in args.columns.split(',')]
        
        # คำนวณเมทริกซ์สหสัมพันธ์
        corr_matrix = data_manager.get_correlation_matrix(columns)
        print("\nเมทริกซ์สหสัมพันธ์:")
        print(corr_matrix.round(3))
    
    # วิเคราะห์สถิติตามช่วงเวลา
    if args.period_stats:
        period_stats = data_manager.get_period_stats(args.period_stats)
        print(f"\nสถิติตามช่วงเวลา ({args.period_stats}):")
        
        # แสดงเฉพาะ 10 แถวแรกและสุดท้าย
        if len(period_stats) > 20:
            import pandas as pd
            print(pd.concat([period_stats.head(10), period_stats.tail(10)]))
        else:
            print(period_stats)
        
        # บันทึกไฟล์ถ้ามีการระบุ output
        if args.output:
            if args.output.endswith('.csv'):
                period_stats.to_csv(args.output, index=False)
            elif args.output.endswith('.parquet'):
                period_stats.to_parquet(args.output, index=False)
            print(f"\nบันทึกสถิติตามช่วงเวลาที่: {args.output}")
    
    # สร้างกราฟ
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from src.utils.visualization import plot_data
            
            # แยกคอลัมน์
            columns = None
            if args.columns:
                columns = [col.strip() for col in args.columns.split(',')]
            
            # สร้างกราฟ
            if args.plot == 'all' or args.plot == 'price':
                plt.figure(figsize=(12, 6))
                plot_data(data_manager.raw_data, 'price', columns)
                plt.show()
            
            if args.plot == 'all' or args.plot == 'volume':
                plt.figure(figsize=(12, 6))
                plot_data(data_manager.raw_data, 'volume', columns)
                plt.show()
            
            if args.plot == 'all' or args.plot == 'returns':
                plt.figure(figsize=(12, 6))
                plot_data(data_manager.raw_data, 'returns', columns)
                plt.show()
            
            if args.plot == 'all' or args.plot == 'indicators':
                plt.figure(figsize=(12, 6))
                plot_data(data_manager.raw_data, 'indicators', columns)
                plt.show()
        
        except ImportError:
            logger.error("ไม่สามารถสร้างกราฟได้: ไม่พบโมดูล matplotlib")
