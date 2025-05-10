import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import io
from PIL import Image
from datetime import datetime

# ใช้ mplfinance แทน mpl_finance ที่ไม่ได้ใช้งานแล้ว
import mplfinance as mpf

# ลองนำเข้า get_config จาก src.utils.config_manager แทน src.utils.config
try:
    from src.utils.config_manager import get_config
except ImportError:
    try:
        from src.utils.config import get_config
    except ImportError:
        # ถ้าไม่พบทั้งสองที่ ให้สร้างฟังก์ชันปลอมขึ้นมา
        def get_config():
            return {}

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class Renderer:
    """
    ตัวแสดงผลสำหรับสภาพแวดล้อมการเทรด
    
    คลาสนี้รับผิดชอบการแสดงผลในรูปแบบต่างๆ เช่น กราฟ, ข้อความ, และรูปภาพ
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = 'console',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        config = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับ Renderer
        
        Parameters:
        render_mode (str, optional): โหมดการแสดงผล ('human', 'rgb_array', 'console', 'none')
        figure_size (Tuple[int, int]): ขนาดของรูปภาพ
        dpi (int): ความละเอียดของรูปภาพ
        config (Config, optional): อ็อบเจ็กต์การตั้งค่า
        """
        # โหลดการตั้งค่า
        self.config = config if config is not None else get_config()
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        env_config = {}
        if hasattr(self.config, 'extract_subconfig'):
            env_config = self.config.extract_subconfig("environment")
        
        # กำหนดค่าพารามิเตอร์
        self.render_mode = render_mode
        if isinstance(env_config, dict) and "render_mode" in env_config:
            self.render_mode = env_config.get("render_mode")
        
        self.figure_size = figure_size
        self.dpi = dpi
        
        # ตัวแปรสำหรับการแสดงผล
        self.fig = None
        self.ax = None
        
        # ตัวแปรสำหรับเก็บประวัติราคาและการกระทำ
        self.price_history = []
        self.action_history = []
        self.balance_history = []
        self.position_history = []
        self.profit_history = []
        
        logger.info(f"สร้าง Renderer (mode={render_mode}, figure_size={figure_size}, dpi={dpi})")
    
    def render(self, data: Dict[str, Any]) -> Optional[Union[np.ndarray, str]]:
        """
        แสดงผลตามข้อมูลที่กำหนด
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลสำหรับการแสดงผล
        
        Returns:
        np.ndarray or str or None: การแสดงผลในรูปแบบต่างๆ ขึ้นอยู่กับ render_mode
        """
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if not data:
            logger.warning("ไม่มีข้อมูลสำหรับการแสดงผล")
            return None
        
        # อัพเดตประวัติ
        self._update_history(data)
        
        # แสดงผลตาม render_mode
        if self.render_mode == 'none':
            return None
        elif self.render_mode == 'human':
            return self._render_human(data)
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array(data)
        elif self.render_mode == 'console':
            return self._render_console(data)
        else:
            logger.warning(f"โหมดการแสดงผล '{self.render_mode}' ไม่ถูกต้อง, ใช้ 'console' แทน")
            return self._render_console(data)
    
    def _update_history(self, data: Dict[str, Any]) -> None:
        """
        อัพเดตประวัติสำหรับการแสดงผล
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลใหม่
        """
        # เก็บประวัติราคา
        if 'price' in data:
            self.price_history.append(data['price'])
        
        # เก็บประวัติการกระทำ
        if 'action' in data:
            self.action_history.append(data['action'])
        
        # เก็บประวัติยอดคงเหลือ
        if 'balance' in data:
            self.balance_history.append(data['balance'])
        elif 'equity' in data:
            self.balance_history.append(data['equity'])
        
        # เก็บประวัติตำแหน่ง
        if 'position' in data:
            self.position_history.append(data['position'])
        
        # เก็บประวัติกำไร/ขาดทุน
        if 'profit' in data:
            self.profit_history.append(data['profit'])
    
    def _render_human(self, data: Dict[str, Any]) -> None:
        """
        แสดงผลในโหมด 'human' (แสดงผลบนหน้าต่าง)
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลสำหรับการแสดงผล
        
        Returns:
        None
        """
        # สร้างรูปภาพ
        self._create_figure(data)
        
        # แสดงผล
        plt.show()
        
        return None
    
    def _render_rgb_array(self, data: Dict[str, Any]) -> np.ndarray:
        """
        แสดงผลในโหมด 'rgb_array' (แปลงเป็น numpy array)
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลสำหรับการแสดงผล
        
        Returns:
        np.ndarray: รูปภาพในรูปแบบ RGB array
        """
        # สร้างรูปภาพ
        self._create_figure(data)
        
        # แปลงเป็น RGB array
        plt.tight_layout()
        fig = plt.gcf()
        
        # บันทึกรูปภาพไปยัง buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='rgba', dpi=self.dpi)
        buf.seek(0)
        
        # แปลงเป็น numpy array
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        
        img = img_arr.reshape(self.dpi * self.figure_size[1], self.dpi * self.figure_size[0], -1)
        
        # ปิดรูปภาพ
        plt.close(fig)
        
        return img
    
    def _render_console(self, data: Dict[str, Any]) -> str:
        """
        แสดงผลในโหมด 'console' (แสดงผลเป็นข้อความ)
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลสำหรับการแสดงผล
        
        Returns:
        str: ข้อความแสดงผล
        """
        output = []
        
        # เพิ่มข้อมูลพื้นฐาน
        output.append("=== Trading Environment ===")
        
        # เพิ่มข้อมูลขั้นตอน
        if 'step' in data:
            output.append(f"Step: {data['step']}")
        
        # เพิ่มข้อมูลวันที่
        if 'date' in data and data['date'] is not None:
            if isinstance(data['date'], np.datetime64) or isinstance(data['date'], pd.Timestamp):
                date_str = pd.to_datetime(data['date']).strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_str = str(data['date'])
            output.append(f"Date: {date_str}")
        
        # เพิ่มข้อมูลราคา
        if 'price' in data:
            output.append(f"Price: {data['price']:.6f}")
        
        # เพิ่มข้อมูลการกระทำ
        if 'action' in data:
            action_map = {0: 'NONE', 1: 'LONG', 2: 'SHORT', 3: 'EXIT'}
            action = action_map.get(data['action'], str(data['action']))
            output.append(f"Action: {action}")
        
        # เพิ่มข้อมูลยอดคงเหลือและทรัพย์สิน
        if 'balance' in data:
            output.append(f"Balance: {data['balance']:.2f}")
        
        if 'equity' in data:
            output.append(f"Equity: {data['equity']:.2f}")
        
        # เพิ่มข้อมูลตำแหน่ง
        if 'position' in data:
            output.append(f"Position: {data['position']}")
        
        if 'units' in data:
            output.append(f"Units: {data['units']:.6f}")
        
        # เพิ่มข้อมูลกำไร/ขาดทุน
        if 'profit' in data:
            output.append(f"Profit: {data['profit']:.2f}")
        
        if 'return' in data:
            output.append(f"Return: {data['return']:.2f}%")
        
        # เชื่อมต่อข้อความด้วยการขึ้นบรรทัดใหม่
        result = "\n".join(output)
        
        # แสดงผลถ้าจำเป็น
        if 'print' in data and data['print']:
            print(result)
        
        return result
    
    def _create_figure(self, data: Dict[str, Any]) -> plt.Figure:
        """
        สร้างรูปภาพสำหรับการแสดงผล
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลสำหรับการแสดงผล
        
        Returns:
        plt.Figure: รูปภาพที่สร้าง
        """
        # ตรวจสอบว่ามีข้อมูลประวัติหรือไม่
        if not self.price_history:
            logger.warning("ไม่มีข้อมูลประวัติสำหรับการสร้างรูปภาพ")
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=16)
            plt.tight_layout()
            return fig
        
        # สร้างรูปภาพ
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figure_size, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # พลอตราคา
        steps = list(range(len(self.price_history)))
        
        ax1.plot(steps, self.price_history, label='Price', color='blue')
        ax1.set_ylabel('Price')
        ax1.set_title('Trading Environment')
        ax1.grid(True)
        
        # พลอตยอดคงเหลือ
        if self.balance_history:
            ax1.plot(steps[:len(self.balance_history)], self.balance_history, label='Balance', color='green', alpha=0.7)
        
        # พลอตตำแหน่ง
        colors = {'none': 'black', 'long': 'green', 'short': 'red'}
        
        for i, pos in enumerate(self.position_history):
            if i >= len(steps):
                break
            
            if pos != 'none':
                color = colors.get(pos, 'black')
                marker = '^' if pos == 'long' else 'v'
                
                if i < len(self.price_history):
                    ax1.scatter(steps[i], self.price_history[i], marker=marker, color=color, s=100)
        
        ax1.legend()
        
        # พลอตการกระทำ
        if self.action_history:
            ax2.bar(steps[:len(self.action_history)], self.action_history, color='purple', alpha=0.7)
            ax2.set_ylabel('Action')
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(['NONE', 'LONG', 'SHORT', 'EXIT'])
            ax2.grid(True)
        
        # พลอตกำไร/ขาดทุน
        if self.profit_history:
            colors = ['green' if p >= 0 else 'red' for p in self.profit_history]
            ax3.bar(steps[:len(self.profit_history)], self.profit_history, color=colors, alpha=0.7)
            ax3.set_ylabel('Profit/Loss')
            ax3.set_xlabel('Step')
            ax3.grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def render_candlestick(self, ohlc_data: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        แสดงผลกราฟแท่งเทียน (ใช้ mplfinance)
        
        Parameters:
        ohlc_data (pd.DataFrame): ข้อมูล OHLC
        **kwargs: พารามิเตอร์เพิ่มเติม
        
        Returns:
        plt.Figure: รูปภาพที่สร้าง
        """
        # คัดลอกข้อมูล
        df = ohlc_data.copy()
        
        # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นหรือไม่
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = []
        for col in required_columns:
            if not any(c.lower() == col for c in df.columns):
                missing_columns.append(col)
        
        if missing_columns:
            logger.error(f"ข้อมูล OHLC ขาดคอลัมน์ที่จำเป็น: {missing_columns}")
            return None
        
        # แปลงชื่อคอลัมน์ให้ตรงกับรูปแบบที่ mplfinance ต้องการ
        rename_dict = {}
        for col in df.columns:
            if col.lower() == 'open':
                rename_dict[col] = 'Open'
            elif col.lower() == 'high':
                rename_dict[col] = 'High'
            elif col.lower() == 'low':
                rename_dict[col] = 'Low'
            elif col.lower() == 'close':
                rename_dict[col] = 'Close'
            elif col.lower() == 'volume':
                rename_dict[col] = 'Volume'
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # ตั้งดัชนีเป็นวันที่ถ้ามี
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        
        # ตรวจสอบว่าดัชนีเป็นประเภทวันที่หรือไม่
        if not isinstance(df.index, pd.DatetimeIndex) and ('timestamp' in df.columns or 'date' in df.columns):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                # ถ้าแปลงเป็นวันที่ไม่ได้ ให้ใช้ดัชนีตัวเลขแทน
                df = df.reset_index(drop=True)
        
        # ตั้งค่าพารามิเตอร์สำหรับ mplfinance
        title = kwargs.get('title', 'Candlestick Chart')
        style = kwargs.get('style', 'yahoo')
        figsize = kwargs.get('figsize', self.figure_size)
        
        # สร้างรูปภาพ
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=style,
            title=title,
            figsize=figsize,
            volume=True if 'Volume' in df.columns else False,
            panel_ratios=(4, 1) if 'Volume' in df.columns else None,
            returnfig=True
        )
        
        return fig[0]  # คืน Figure object
    
    def close(self) -> None:
        """
        ปิดตัวแสดงผลและทรัพยากรที่ใช้
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        
        # รีเซ็ตประวัติ
        self.price_history = []
        self.action_history = []
        self.balance_history = []
        self.position_history = []
        self.profit_history = []