import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import io
from PIL import Image
from datetime import datetime
import mplfinance as mpf

try:
    from src.utils.config_manager import get_config
except ImportError:
    try:
        from src.utils.config import get_config
    except ImportError:
        def get_config():
            return {}

logger = logging.getLogger(__name__)

class Renderer:
    def __init__(
        self,
        render_mode: Optional[str] = 'console',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        config = None
    ):
        self.config = config if config is not None else get_config()
        
        env_config = {}
        if hasattr(self.config, 'extract_subconfig'):
            env_config = self.config.extract_subconfig("environment")
        
        self.render_mode = render_mode
        if isinstance(env_config, dict) and "render_mode" in env_config:
            self.render_mode = env_config.get("render_mode")
        
        self.figure_size = figure_size
        self.dpi = dpi
        self.fig = None
        self.ax = None
        
        self.price_history = []
        self.action_history = []
        self.balance_history = []
        self.position_history = []
        self.profit_history = []
        
        logger.info(f"Created Renderer (mode={render_mode}, figure_size={figure_size}, dpi={dpi})")
    
    def render(self, data: Dict[str, Any]) -> Optional[Union[np.ndarray, str]]:
        if not data:
            logger.warning("No data for rendering")
            return None
        
        self._update_history(data)
        
        if self.render_mode == 'none':
            return None
        elif self.render_mode == 'human':
            return self._render_human(data)
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array(data)
        elif self.render_mode == 'console':
            return self._render_console(data)
        else:
            logger.warning(f"Invalid render mode '{self.render_mode}', using 'console'")
            return self._render_console(data)
    
    def _update_history(self, data: Dict[str, Any]) -> None:
        if 'price' in data: self.price_history.append(data['price'])
        if 'action' in data: self.action_history.append(data['action'])
        if 'balance' in data: self.balance_history.append(data['balance'])
        elif 'equity' in data: self.balance_history.append(data['equity'])
        if 'position' in data: self.position_history.append(data['position'])
        if 'profit' in data: self.profit_history.append(data['profit'])
    
    def _render_human(self, data: Dict[str, Any]) -> None:
        self._create_figure(data)
        plt.show()
        return None
    
    def _render_rgb_array(self, data: Dict[str, Any]) -> np.ndarray:
        self._create_figure(data)
        plt.tight_layout()
        fig = plt.gcf()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='rgba', dpi=self.dpi)
        buf.seek(0)
        
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        
        img = img_arr.reshape(self.dpi * self.figure_size[1], self.dpi * self.figure_size[0], -1)
        plt.close(fig)
        
        return img
    
    def _render_console(self, data: Dict[str, Any]) -> str:
        output = ["=== Trading Environment ==="]
        
        if 'step' in data:
            output.append(f"Step: {data['step']}")
        
        if 'date' in data and data['date'] is not None:
            if isinstance(data['date'], np.datetime64) or isinstance(data['date'], pd.Timestamp):
                date_str = pd.to_datetime(data['date']).strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_str = str(data['date'])
            output.append(f"Date: {date_str}")
        
        if 'price' in data:
            output.append(f"Price: {data['price']:.6f}")
        
        if 'action' in data:
            action_map = {0: 'NONE', 1: 'LONG', 2: 'SHORT', 3: 'EXIT'}
            action = action_map.get(data['action'], str(data['action']))
            output.append(f"Action: {action}")
        
        if 'balance' in data:
            output.append(f"Balance: {data['balance']:.2f}")
        
        if 'equity' in data:
            output.append(f"Equity: {data['equity']:.2f}")
        
        if 'position' in data:
            output.append(f"Position: {data['position']}")
        
        if 'units' in data:
            output.append(f"Units: {data['units']:.6f}")
        
        if 'profit' in data:
            output.append(f"Profit: {data['profit']:.2f}")
        
        if 'return' in data:
            output.append(f"Return: {data['return']:.2f}%")
        
        result = "\n".join(output)
        
        if 'print' in data and data['print']:
            print(result)
        
        return result
    
    def _create_figure(self, data: Dict[str, Any]) -> plt.Figure:
        if not self.price_history:
            logger.warning("No history data for figure creation")
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=16)
            plt.tight_layout()
            return fig
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figure_size, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        steps = list(range(len(self.price_history)))
        
        ax1.plot(steps, self.price_history, label='Price', color='blue')
        ax1.set_ylabel('Price')
        ax1.set_title('Trading Environment')
        ax1.grid(True)
        
        if self.balance_history:
            ax1.plot(steps[:len(self.balance_history)], self.balance_history, label='Balance', color='green', alpha=0.7)
        
        colors = {'none': 'black', 'long': 'green', 'short': 'red'}
        
        for i, pos in enumerate(self.position_history):
            if i >= len(steps): break
            
            if pos != 'none':
                color = colors.get(pos, 'black')
                marker = '^' if pos == 'long' else 'v'
                
                if i < len(self.price_history):
                    ax1.scatter(steps[i], self.price_history[i], marker=marker, color=color, s=100)
        
        ax1.legend()
        
        if self.action_history:
            ax2.bar(steps[:len(self.action_history)], self.action_history, color='purple', alpha=0.7)
            ax2.set_ylabel('Action')
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(['NONE', 'LONG', 'SHORT', 'EXIT'])
            ax2.grid(True)
        
        if self.profit_history:
            colors = ['green' if p >= 0 else 'red' for p in self.profit_history]
            ax3.bar(steps[:len(self.profit_history)], self.profit_history, color=colors, alpha=0.7)
            ax3.set_ylabel('Profit/Loss')
            ax3.set_xlabel('Step')
            ax3.grid(True)
        
        plt.tight_layout()
        return fig
    
    def render_candlestick(self, ohlc_data: pd.DataFrame, **kwargs) -> plt.Figure:
        df = ohlc_data.copy()
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if not any(c.lower() == col for c in df.columns)]
        
        if missing_columns:
            logger.error(f"OHLC data missing required columns: {missing_columns}")
            return None
        
        rename_dict = {}
        for col in df.columns:
            lower_col = col.lower()
            if lower_col == 'open': rename_dict[col] = 'Open'
            elif lower_col == 'high': rename_dict[col] = 'High'
            elif lower_col == 'low': rename_dict[col] = 'Low'
            elif lower_col == 'close': rename_dict[col] = 'Close'
            elif lower_col == 'volume': rename_dict[col] = 'Volume'
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        
        if not isinstance(df.index, pd.DatetimeIndex) and ('timestamp' in df.columns or 'date' in df.columns):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                df = df.reset_index(drop=True)
        
        title = kwargs.get('title', 'Candlestick Chart')
        style = kwargs.get('style', 'yahoo')
        figsize = kwargs.get('figsize', self.figure_size)
        
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
        
        return fig[0]
    
    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        
        self.price_history = []
        self.action_history = []
        self.balance_history = []
        self.position_history = []
        self.profit_history = []
