import os
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
from tqdm import tqdm

from src.data.downloaders.base_downloader import BaseDownloader
from src.data.api.binance_api import BinanceAPI
from src.data.utils.time_utils import get_timeframe_delta

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class BinanceDownloader(BaseDownloader):
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__()
        self.api = BinanceAPI(api_key, api_secret)
    
    def download_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime, None] = None,
        output_dir: str = "data/raw",
        file_format: str = "both",
        include_current_candle: bool = False
    ) -> pd.DataFrame:
        # ตรวจสอบและแปลงพารามิเตอร์
        symbol = symbol.upper()
        timeframe = self.api.validate_timeframe(timeframe)
        start_ts = self._parse_date(start_date)
        
        # ถ้าไม่ระบุ end_date ให้ใช้วันปัจจุบัน
        if end_date is None:
            end_ts = int(datetime.now().timestamp() * 1000)
        else:
            end_ts = self._parse_date(end_date)
        
        # สร้างไดเรกทอรีสำหรับบันทึกข้อมูล
        symbol_dir = os.path.join(output_dir, symbol.replace("/", "-"))
        timeframe_dir = os.path.join(symbol_dir, timeframe)
        os.makedirs(timeframe_dir, exist_ok=True)
        
        logger.info(f"เริ่มดาวน์โหลดข้อมูล {symbol} ไทม์เฟรม {timeframe} ตั้งแต่ {datetime.fromtimestamp(start_ts/1000)} ถึง {datetime.fromtimestamp(end_ts/1000)}")
        
        # คำนวณจำนวนแท่งเทียนทั้งหมดที่ต้องดาวน์โหลด
        interval_ms = self.api.get_interval_ms(timeframe)
        total_candles = (end_ts - start_ts) // interval_ms
        
        if total_candles <= 0:
            logger.warning(f"ไม่มีข้อมูลในช่วงเวลาที่กำหนด: {start_date} ถึง {end_date}")
            return pd.DataFrame()
        
        logger.info(f"ประมาณการจำนวนแท่งเทียนที่ต้องดาวน์โหลด: {total_candles}")
        
        # ดาวน์โหลดข้อมูลเป็นชุดๆ
        all_candles = []
        current_start_ts = start_ts
        
        with tqdm(total=total_candles, desc=f"ดาวน์โหลด {symbol} {timeframe}") as pbar:
            while current_start_ts < end_ts:
                # คำนวณเวลาสิ้นสุดสำหรับแต่ละชุด
                current_end_ts = min(
                    current_start_ts + (self.api.MAX_CANDLES_PER_REQUEST * interval_ms),
                    end_ts
                )
                
                # ดาวน์โหลดแท่งเทียนสำหรับช่วงเวลานี้
                candles = self.api.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=current_start_ts,
                    end_time=current_end_ts,
                    limit=self.api.MAX_CANDLES_PER_REQUEST
                )
                
                if not candles:
                    logger.warning(f"ไม่พบข้อมูลในช่วง {datetime.fromtimestamp(current_start_ts/1000)} ถึง {datetime.fromtimestamp(current_end_ts/1000)}")
                    current_start_ts = current_end_ts
                    continue
                
                # เก็บข้อมูลที่ดาวน์โหลดได้
                all_candles.extend(candles)
                
                # อัพเดต progress bar
                pbar.update(len(candles))
                
                # เตรียมสำหรับการดาวน์โหลดชุดต่อไป
                if len(candles) < self.api.MAX_CANDLES_PER_REQUEST:
                    # ถ้าได้รับข้อมูลน้อยกว่าที่ขอ แสดงว่าไม่มีข้อมูลเพิ่มเติมแล้ว
                    break
                
                # เลื่อนเวลาเริ่มต้นสำหรับชุดต่อไป
                current_start_ts = candles[-1][0] + interval_ms
                
                # หน่วงเวลาเพื่อไม่ให้เกิน rate limit
                time.sleep(0.5)
        
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if not all_candles:
            logger.warning(f"ไม่พบข้อมูลสำหรับ {symbol} ไทม์เฟรม {timeframe} ในช่วงเวลาที่กำหนด")
            return pd.DataFrame()
        
        # แปลงข้อมูลเป็น DataFrame
        df = self._candles_to_dataframe(all_candles)
        
        # ลบแท่งเทียนสุดท้ายถ้ายังไม่ปิดตามขอบเขตปลายทาง (now หรือ end_date)
        if not include_current_candle and len(df) > 0:
            interval_ms = self.api.get_interval_ms(timeframe)
            # คำนวณเวลาปิดของแท่งสุดท้ายใน DataFrame
            last_open_ms = int(df['timestamp'].iloc[-1].value // 1_000_000)
            last_close_ms = last_open_ms + interval_ms - 1
            # ปลายทางที่ใช้ตรวจสอบ: ถ้า user ระบุ end_date ให้ใช้ค่านั้น ไม่เช่นนั้นให้ใชเวลา "ตอนนี้"
            effective_end_ms = end_ts if end_date is not None else int(datetime.now().timestamp() * 1000)
            # ถ้าแท่งสุดท้ายยังไม่ปิดเมื่อเทียบกับปลายทาง ให้ตัดออก
            if last_close_ms >= effective_end_ms:
                df = df.iloc[:-1]
        
        # บันทึกข้อมูล
        file_prefix = f"{symbol.lower()}_{timeframe}_{datetime.fromtimestamp(start_ts/1000).strftime('%Y%m%d')}_{datetime.fromtimestamp(end_ts/1000).strftime('%Y%m%d')}"
        
        if file_format.lower() in ["csv", "both"]:
            csv_path = os.path.join(timeframe_dir, f"{file_prefix}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"บันทึกข้อมูลเป็น CSV ที่: {csv_path}")
        
        if file_format.lower() in ["parquet", "both"]:
            parquet_path = os.path.join(timeframe_dir, f"{file_prefix}.parquet")
            df.to_parquet(parquet_path, index=False)
            logger.info(f"บันทึกข้อมูลเป็น Parquet ที่: {parquet_path}")
        
        logger.info(f"ดาวน์โหลดข้อมูลสำเร็จ: {len(df)} แท่งเทียน")
        return df
    
    def update_data(
        self,
        symbol: str,
        timeframe: str,
        data_dir: str = "data/raw",
        file_format: str = "both"
    ) -> Optional[pd.DataFrame]:
        symbol = symbol.upper()
        timeframe = self.api.validate_timeframe(timeframe)
        
        # ค้นหาไฟล์ล่าสุด
        symbol_dir = os.path.join(data_dir, symbol.replace("/", "-"))
        timeframe_dir = os.path.join(symbol_dir, timeframe)
        
        if not os.path.exists(timeframe_dir):
            logger.warning(f"ไม่พบไดเรกทอรี {timeframe_dir}")
            return None
        
        # ค้นหาไฟล์ Parquet หรือ CSV ล่าสุด
        latest_file = self._find_latest_file(timeframe_dir)
        
        if not latest_file:
            logger.warning(f"ไม่พบไฟล์ข้อมูลใน {timeframe_dir}")
            return None
        
        # โหลดข้อมูลจากไฟล์ล่าสุด
        if latest_file.endswith('.parquet'):
            df = pd.read_parquet(latest_file)
        else:  # .csv
            df = pd.read_csv(latest_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # หาวันที่ล่าสุดในข้อมูล
        last_timestamp = df['timestamp'].max()
        
        # เพิ่ม 1 ช่วงเวลาเพื่อหลีกเลี่ยงการดาวน์โหลดซ้ำ
        interval_ms = self.api.get_interval_ms(timeframe)
        start_ts = int(last_timestamp.timestamp() * 1000) + interval_ms
        
        # ถ้าเวลาเริ่มต้นอยู่ในอนาคต ให้ถือว่าข้อมูลเป็นปัจจุบันแล้ว
        if start_ts > int(datetime.now().timestamp() * 1000):
            logger.info(f"ข้อมูล {symbol} {timeframe} เป็นปัจจุบันแล้ว")
            return df
        
        # ดาวน์โหลดข้อมูลใหม่
        logger.info(f"กำลังอัพเดตข้อมูล {symbol} {timeframe} ตั้งแต่ {datetime.fromtimestamp(start_ts/1000)}")
        
        # ดาวน์โหลดข้อมูลใหม่
        new_df = self.download_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.fromtimestamp(start_ts/1000),
            output_dir=data_dir,
            file_format=file_format,
            include_current_candle=False
        )
        
        if new_df.empty:
            logger.info(f"ไม่มีข้อมูลใหม่สำหรับ {symbol} {timeframe}")
            return df
        
        # รวมข้อมูลเก่าและใหม่
        combined_df = pd.concat([df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        # บันทึกข้อมูลที่รวมแล้ว
        file_prefix = f"{symbol.lower()}_{timeframe}_combined"
        
        if file_format.lower() in ["csv", "both"]:
            csv_path = os.path.join(timeframe_dir, f"{file_prefix}.csv")
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"บันทึกข้อมูลรวมเป็น CSV ที่: {csv_path}")
        
        if file_format.lower() in ["parquet", "both"]:
            parquet_path = os.path.join(timeframe_dir, f"{file_prefix}.parquet")
            combined_df.to_parquet(parquet_path, index=False)
            logger.info(f"บันทึกข้อมูลรวมเป็น Parquet ที่: {parquet_path}")
        
        logger.info(f"อัพเดตข้อมูลสำเร็จ: เพิ่ม {len(new_df)} แท่งเทียน รวมทั้งหมด {len(combined_df)} แท่งเทียน")
        return combined_df
    
    def validate_data(
        self,
        df: pd.DataFrame,
        timeframe: str,
        fill_missing: bool = True,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        if df.empty:
            logger.warning("DataFrame ว่างเปล่า")
            return df
        
        # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นครบหรือไม่
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"คอลัมน์ที่จำเป็นหายไป: {missing_columns}")
        
        # ตรวจสอบว่า timestamp เป็น datetime หรือไม่
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        original_length = len(df)
        
        # ลบข้อมูลที่ซ้ำกัน
        if remove_duplicates:
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            if len(df) < original_length:
                logger.info(f"ลบข้อมูลที่ซ้ำกัน {original_length - len(df)} แถว")
        
        # เรียงข้อมูลตาม timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # ตรวจสอบและเติมข้อมูลที่หายไป
        if fill_missing and len(df) > 1:
            # คำนวณช่วงเวลาระหว่างแท่งเทียน
            interval_ms = self.api.get_interval_ms(timeframe)
            expected_diff = pd.Timedelta(milliseconds=interval_ms)
            
            # ตรวจสอบว่ามีช่องว่างในข้อมูลหรือไม่
            timestamps = df['timestamp'].copy()
            diffs = timestamps.diff()
            
            # หาช่องว่างในข้อมูล
            gaps = diffs[diffs > expected_diff]
            
            if not gaps.empty:
                logger.info(f"พบช่องว่างในข้อมูล {len(gaps)} ช่อง")
                
                if fill_missing:
                    # สร้างช่วงเวลาที่สมบูรณ์
                    start_time = df['timestamp'].min()
                    end_time = df['timestamp'].max()
                    
                    # สร้าง timestamp ที่สมบูรณ์
                    complete_timestamps = pd.date_range(
                        start=start_time,
                        end=end_time,
                        freq=pd.Timedelta(milliseconds=interval_ms)
                    )
                    
                    # สร้าง DataFrame ใหม่ด้วย timestamp ที่สมบูรณ์
                    complete_df = pd.DataFrame({'timestamp': complete_timestamps})
                    
                    # รวมกับข้อมูลเดิม
                    merged_df = pd.merge(complete_df, df, on='timestamp', how='left')
                    
                    # เติมข้อมูลที่หายไป
                    # ใช้ราคาปิดล่าสุดเป็นราคาเปิด สูง ต่ำ และปิดของแท่งเทียนที่หายไป
                    # ตั้งปริมาณเป็น 0
                    merged_df['close'] = merged_df['close'].fillna(method='ffill')
                    merged_df['open'] = merged_df['open'].fillna(merged_df['close'])
                    merged_df['high'] = merged_df['high'].fillna(merged_df['close'])
                    merged_df['low'] = merged_df['low'].fillna(merged_df['close'])
                    merged_df['volume'] = merged_df['volume'].fillna(0)
                    
                    # เติมคอลัมน์อื่นๆ
                    for col in df.columns:
                        if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                            if col in ['quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']:
                                merged_df[col] = merged_df[col].fillna(0)
                            elif col in ['trades']:
                                merged_df[col] = merged_df[col].fillna(0).astype(int)
                            elif col == 'close_time':
                                # คำนวณ close_time จาก timestamp (timestamp + interval - 1ms)
                                if 'close_time' not in merged_df.columns:
                                    merged_df['close_time'] = merged_df['timestamp'] + pd.Timedelta(milliseconds=interval_ms-1)
                                else:
                                    merged_df['close_time'] = merged_df['close_time'].fillna(
                                        merged_df['timestamp'] + pd.Timedelta(milliseconds=interval_ms-1)
                                    )
                            else:
                                merged_df[col] = merged_df[col].fillna(method='ffill')
                    
                    logger.info(f"เติมข้อมูลที่หายไป {len(merged_df) - len(df)} แถว")
                    return merged_df
        
        return df
    
    def download_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime, None] = None,
        output_dir: str = "data/raw",
        file_format: str = "both"
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        
        for timeframe in timeframes:
            logger.info(f"กำลังดาวน์โหลดข้อมูล {symbol} ไทม์เฟรม {timeframe}")
            df = self.download_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                file_format=file_format
            )
            results[timeframe] = df
        
        return results
    
    def update_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        data_dir: str = "data/raw",
        file_format: str = "both"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        results = {}
        
        for timeframe in timeframes:
            logger.info(f"กำลังอัพเดตข้อมูล {symbol} ไทม์เฟรม {timeframe}")
            df = self.update_data(
                symbol=symbol,
                timeframe=timeframe,
                data_dir=data_dir,
                file_format=file_format
            )
            results[timeframe] = df
        
        return results
    
    def convert_timeframe(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str
    ) -> pd.DataFrame:
        if df.empty:
            logger.warning("DataFrame ว่างเปล่า")
            return df
        
        # ตรวจสอบว่า timestamp เป็น datetime หรือไม่
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ตรวจสอบว่าไทม์เฟรมถูกต้องหรือไม่
        source_timeframe = self.api.validate_timeframe(source_timeframe)
        target_timeframe = self.api.validate_timeframe(target_timeframe)
        
        # ตรวจสอบว่าสามารถแปลงไทม์เฟรมได้หรือไม่
        source_interval_ms = self.api.get_interval_ms(source_timeframe)
        target_interval_ms = self.api.get_interval_ms(target_timeframe)
        
        if target_interval_ms < source_interval_ms:
            raise ValueError(f"ไม่สามารถแปลงจากไทม์เฟรมใหญ่ ({source_timeframe}) ไปเป็นไทม์เฟรมเล็ก ({target_timeframe}) ได้")
        
        if target_interval_ms % source_interval_ms != 0:
            raise ValueError(f"ไม่สามารถแปลงจากไทม์เฟรม {source_timeframe} ไปเป็น {target_timeframe} ได้ (ไม่เป็นสัดส่วนกัน)")
        
        # กำหนดรูปแบบการรวมข้อมูล
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        
        # เพิ่มคอลัมน์อื่นๆ ถ้ามี
        for col in df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time']:
                if col in ['quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']:
                    ohlc_dict[col] = 'sum'
                elif col in ['trades']:
                    ohlc_dict[col] = 'sum'
                else:
                    ohlc_dict[col] = 'last'
        
        # กำหนดขอบเขตเวลาสำหรับการรวมข้อมูล
        # ตัดเวลาให้ตรงกับขอบเขตของไทม์เฟรมเป้าหมาย
        
        # หาจุดเริ่มต้นที่พอดีกับไทม์เฟรมเป้าหมาย
        first_timestamp = df['timestamp'].min()
        
        # คำนวณ offset จาก timestamp แรกไปยังเวลาที่พอดีกับไทม์เฟรมเป้าหมาย
        if target_timeframe.endswith('m'):
            # ไทม์เฟรมนาที
            target_start = first_timestamp.replace(
                second=0, microsecond=0,
                minute=(first_timestamp.minute // int(target_timeframe[:-1])) * int(target_timeframe[:-1])
            )
        elif target_timeframe.endswith('h'):
            # ไทม์เฟรมชั่วโมง
            target_start = first_timestamp.replace(
                second=0, microsecond=0, minute=0,
                hour=(first_timestamp.hour // int(target_timeframe[:-1])) * int(target_timeframe[:-1])
            )
        elif target_timeframe.endswith('d'):
            # ไทม์เฟรมวัน
            target_start = first_timestamp.replace(
                second=0, microsecond=0, minute=0, hour=0
            )
        elif target_timeframe.endswith('w'):
            # ไทม์เฟรมสัปดาห์
            # วันจันทร์เป็นวันแรกของสัปดาห์ (weekday=0 คือวันจันทร์)
            days_since_monday = first_timestamp.weekday()
            target_start = (first_timestamp - pd.Timedelta(days=days_since_monday)).replace(
                second=0, microsecond=0, minute=0, hour=0
            )
        elif target_timeframe.endswith('M'):
            # ไทม์เฟรมเดือน
            target_start = first_timestamp.replace(
                second=0, microsecond=0, minute=0, hour=0, day=1
            )
        else:
            target_start = first_timestamp
        
        # กรองข้อมูลที่อยู่ก่อนจุดเริ่มต้นที่พอดีกับไทม์เฟรมเป้าหมาย
        df = df[df['timestamp'] >= target_start].copy()
        
        # สร้างคอลัมน์สำหรับการรวมข้อมูล
        if target_timeframe.endswith('m'):
            # ไทม์เฟรมนาที
            df['group'] = df['timestamp'].dt.floor(f"{target_timeframe[:-1]}min")
        elif target_timeframe.endswith('h'):
            # ไทม์เฟรมชั่วโมง
            df['group'] = df['timestamp'].dt.floor(f"{target_timeframe[:-1]}h")
        elif target_timeframe.endswith('d'):
            # ไทม์เฟรมวัน
            df['group'] = df['timestamp'].dt.floor('d')
        elif target_timeframe.endswith('w'):
            # ไทม์เฟรมสัปดาห์
            df['group'] = df['timestamp'].dt.to_period('W').dt.start_time
        elif target_timeframe.endswith('M'):
            # ไทม์เฟรมเดือน
            df['group'] = df['timestamp'].dt.to_period('M').dt.start_time
        else:
            raise ValueError(f"ไม่รองรับไทม์เฟรม: {target_timeframe}")
        
        # รวมข้อมูลตามกลุ่ม
        df_ohlc = df.groupby('group').agg(ohlc_dict)
        
        # รีเซ็ตดัชนีและเปลี่ยนชื่อคอลัมน์
        df_ohlc = df_ohlc.reset_index()
        df_ohlc = df_ohlc.rename(columns={'group': 'timestamp'})
        
        # เพิ่มคอลัมน์ close_time ถ้ายังไม่มี
        if 'close_time' not in df_ohlc.columns:
            df_ohlc['close_time'] = df_ohlc['timestamp'] + pd.Timedelta(milliseconds=target_interval_ms-1)
        
        return df_ohlc
