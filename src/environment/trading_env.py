import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from src.environment.base_env import BaseEnv
from src.environment.simulators.trading_simulator import TradingSimulator
from src.environment.renderers.renderer import Renderer
from src.data.managers.data_manager import MarketDataManager
from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class TradingEnv(BaseEnv):
    ACTIONS = {'NONE': 0, 'LONG': 1, 'SHORT': 2, 'EXIT': 3}

    def __init__(
        self,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        window_size: int = 60,
        initial_balance: float = 1e4,
        transaction_fee: float = 0.0025,
        reward_fn: Union[str, Callable] = 'sharpe',
        use_position_info: bool = True,
        normalize_obs: bool = True,
        render_mode: Optional[str] = None,
        config=None
    ):
        self.config = config or get_config()
        env_cfg = self.config.extract_subconfig("environment")
        self.window_size = window_size or env_cfg.get("window_size", 60)
        self.initial_balance = initial_balance or env_cfg.get("initial_balance", 1e4)
        self.transaction_fee = transaction_fee or env_cfg.get("fee_rate", 0.0025)
        self.use_position_info = use_position_info
        self.normalize_obs = normalize_obs

        # Data manager
        if file_path:
            self.data_manager = MarketDataManager(
                file_path=file_path,
                symbol=symbol or env_cfg.get("symbol"),
                window_size=self.window_size,
                config=self.config
            )
            if not self.data_manager.data_loaded:
                raise ValueError(f"Failed to load data from {file_path}")
        else:
            self.data_manager = None
            logger.warning("No data source provided; load_data() must be called before use.")

        # Simulator and renderer
        self.simulator = TradingSimulator(self.initial_balance, self.transaction_fee, self.config)
        self.renderer = Renderer(render_mode or env_cfg.get("render_mode"), self.config)
        self.history: list = []

        # Spaces
        action_space = spaces.Discrete(len(self.ACTIONS))
        obs_space = spaces.Box(-np.inf, np.inf, shape=(self.window_size, self._feature_dim()), dtype=np.float32)
        super().__init__(obs_space, action_space, render_mode, self.config)

        # Reward
        self.reward_function = self._get_reward_fn(reward_fn)
    
    def _feature_dim(self):
        # Count features excluding timestamp
        cols = self.data_manager.data.columns.tolist()
        dim = len(cols) - (1 if 'timestamp' in cols else 0)
        return dim + (3 if self.use_position_info else 0)
    
    def load_data(self, file_path: str) -> None:
        if not self.data_manager:
            self.data_manager = MarketDataManager(file_path=file_path, window_size=self.window_size, config=self.config)
        else:
            self.data_manager.load_data(file_path)
        if not self.data_manager.data_loaded:
            raise ValueError(f"Failed to load data from {file_path}")
    
    def _setup_observation_space(self) -> None:
        """
        กำหนดพื้นที่สังเกตการณ์ (observation space) ตามข้อมูลที่โหลด
        """
        if self.data_manager.data is None:
            logger.error("ไม่สามารถกำหนด observation_space เนื่องจากยังไม่ได้โหลดข้อมูล")
            return
        
        # จำนวนคุณลักษณะ (features) ในข้อมูล
        num_features = self.data_manager.data.shape[1]
        
        # ลบคอลัมน์ timestamp ถ้ามี
        if 'timestamp' in self.data_manager.data.columns:
            num_features -= 1
        
        # เพิ่มคุณลักษณะสำหรับข้อมูลตำแหน่ง
        if self.use_position_info:
            num_features += 3  # position_type, units, profit
        
        # เก็บ obs_shape
        self.obs_shape = (self.window_size, num_features)
        
        # กำหนด observation_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_shape,
            dtype=np.float32
        )
        
        logger.info(f"กำหนด observation_space: {self.observation_space}")
    
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed, options)
        opts = options or {}
        idx = opts.get('start_index', 0)
        if opts.get('start_date') and 'timestamp' in self.data_manager.raw_data:
            dt = pd.to_datetime(opts['start_date'])
            idxs = np.searchsorted(self.data_manager.raw_data['timestamp'], dt)
            idx = idxs[0] if len(idxs) else idx
        self.current_step = max(0, min(idx, len(self.data_manager.data) - self.window_size))
        self.prices = self.data_manager.raw_data['close'].to_numpy()
        self.dates = self.data_manager.raw_data['timestamp'].to_numpy()
        self.simulator.reset(self.initial_balance)
        self.history.clear()
        obs = self._get_observation()
        info.update({ 'step': self.current_step, 'price': self.prices[self.current_step] })
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        ดำเนินการหนึ่งขั้นตอนในสภาพแวดล้อมตามการกระทำที่กำหนด
        
        Parameters:
        action (int): การกระทำที่เลือก (0=NONE, 1=LONG, 2=SHORT, 3=EXIT)
        
        Returns:
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: สถานะใหม่, รางวัล, done, truncated, ข้อมูลเพิ่มเติม
        """
        # ตรวจสอบว่าการกระทำถูกต้องหรือไม่
        if not self.action_space.contains(action):
            logger.warning(f"การกระทำไม่ถูกต้อง: {action}, ใช้ NONE แทน")
            action = self.ACTIONS['NONE']
        
        # ตรวจสอบว่า episode จบแล้วหรือไม่
        if self.done:
            next_obs = self._get_observation()
            return next_obs, 0.0, self.done, False, self.info
        
        # ดำเนินการตามการกระทำและคำนวณผลลัพธ์
        next_state, reward, done, info = self._process_action(action)
        
        # อัพเดตตัวแปรต่างๆ
        self.state = next_state
        self.done = done
        self.info = info
        self.total_rewards += reward
        
        # เพิ่มข้อมูลลงในประวัติ
        self.history.append({
            'step': self.current_step,
            'price': self.prices[self.current_step] if self.current_step < len(self.prices) else 0,
            'date': self.dates[self.current_step] if self.current_step < len(self.dates) else None,
            'action': action,
            'reward': reward,
            'balance': self.simulator.balance,
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit
        })
        
        # ตรวจสอบ truncated (จำกัดจำนวนขั้นตอน)
        truncated = False
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True
        
        # แสดงผลถ้าจำเป็น
        if self.render_mode != "none":
            self.render()
        
        return next_state, reward, done, truncated, info
    
    def _process_action(self, action: int):
        # Validate action
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action {action}, defaulting to NONE.")
            action = self.ACTIONS['NONE']
        # Execute trade
        price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1] if self.current_step > 0 else price
        if action == self.ACTIONS['LONG']:
            self.simulator.open_long_position(price)
        elif action == self.ACTIONS['SHORT']:
            self.simulator.open_short_position(price)
        elif action == self.ACTIONS['EXIT']:
            self.simulator.close_position(price)
        self.simulator.update(price)
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        if done and self.simulator.has_position():
            self.simulator.close_position(price)
        reward = self.reward_function(self.simulator, price, prev_price)
        obs = self._get_observation()
        info = self._make_info(price)
        self.history.append(info.copy())
        return obs, reward, done, info
    
    def _get_observation(self):
        df = self.data_manager.data
        start = max(0, self.current_step - self.window_size + 1)
        window = df.iloc[start:self.current_step+1].copy()
        if 'timestamp' in window:
            window.drop('timestamp', axis=1, inplace=True)
        arr = window.reindex(list(range(self.current_step - self.window_size + 1, self.current_step+1))).fillna(0).to_numpy()
        if self.use_position_info:
            pos = np.full((self.window_size, 3), [
                1 if self.simulator.position_type=='long' else -1 if self.simulator.position_type=='short' else 0,
                self.simulator.units,
                self.simulator.profit
            ])
            arr = np.hstack([arr, pos])
        return arr.astype(np.float32)
    
    def _make_info(self, price):
        return {
            'step': self.current_step,
            'price': price,
            'balance': self.simulator.balance,
            'equity': self.simulator.get_equity(price),
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit,
            'action': None
        }
    
    def _get_initial_state(self, options: Dict[str, Any] = None) -> np.ndarray:
        """
        เตรียมสถานะเริ่มต้นของสภาพแวดล้อม
        
        Parameters:
        options (Dict[str, Any], optional): ตัวเลือกเพิ่มเติมสำหรับการรีเซ็ต
        
        Returns:
        np.ndarray: สถานะเริ่มต้น
        """
        return self._get_observation()
    
    def _render_frame(self) -> Optional[Union[np.ndarray, str]]:
        """
        สร้างเฟรมสำหรับการแสดงผล
        
        Returns:
        np.ndarray or str or None: เฟรมสำหรับการแสดงผล
        """
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if self.data_manager is None or self.current_step >= len(self.prices):
            return None
        
        # เตรียมข้อมูลสำหรับการแสดงผล
        render_data = {
            'step': self.current_step,
            'price': self.prices[self.current_step - 1] if self.current_step > 0 else 0,
            'date': self.dates[self.current_step - 1] if self.current_step > 0 else None,
            'balance': self.simulator.balance,
            'equity': self.simulator.get_equity(self.prices[self.current_step - 1] if self.current_step > 0 else 0),
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit,
            'return': self.simulator.total_return,
            'history': self.history
        }
        
        # แสดงผลด้วย renderer
        return self.renderer.render(render_data)
    
    def render(self):
        if self.render_mode=='none': return None
        data = self.history
        return self.renderer.render({ 'history': data })
    
    def close(self):
        super().close()
        if self.renderer:
            self.renderer.close()
    
    def _get_reward_function(self, rt: Union[str, Callable]) -> Callable:
        funcs = {
            'profit': lambda sim, c, p: sim.profit,
            'return': self._return_reward,
            'sharpe': self._sharpe_reward,
            'sortino': self._sortino_reward,
            'calmar': self._calmar_reward,
            'custom': self._custom_reward
        }
        return funcs.get(rt, funcs['profit'])
    
    def _profit_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        """
        คำนวณรางวัลตามกำไร/ขาดทุน
        
        Parameters:
        simulator (TradingSimulator): อินสแตนซ์ของ TradingSimulator
        current_price (float): ราคาปัจจุบัน
        prev_price (float): ราคาก่อนหน้า
        
        Returns:
        float: รางวัล
        """
        return simulator.profit
    
    def _return_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        """
        คำนวณรางวัลตามผลตอบแทน
        
        Parameters:
        simulator (TradingSimulator): อินสแตนซ์ของ TradingSimulator
        current_price (float): ราคาปัจจุบัน
        prev_price (float): ราคาก่อนหน้า
        
        Returns:
        float: รางวัล
        """
        if simulator.position_type == 'none':
            return 0.0
        
        # คำนวณผลตอบแทนจากการเปลี่ยนแปลงของราคา
        price_change = (current_price - prev_price) / prev_price
        
        # ปรับตามประเภทของตำแหน่ง
        if simulator.position_type == 'long':
            return price_change * 100  # เปอร์เซ็นต์
        elif simulator.position_type == 'short':
            return -price_change * 100  # เปอร์เซ็นต์
        
        return 0.0
    
    def _sharpe_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        """
        คำนวณรางวัลตาม Sharpe Ratio
        
        Parameters:
        simulator (TradingSimulator): อินสแตนซ์ของ TradingSimulator
        current_price (float): ราคาปัจจุบัน
        prev_price (float): ราคาก่อนหน้า
        
        Returns:
        float: รางวัล
        """
        # คำนวณผลตอบแทน
        returns = self._return_reward(simulator, current_price, prev_price)
        
        # ถ้ายังไม่มีข้อมูลพอสำหรับคำนวณ Sharpe Ratio
        if len(self.history) < 10:
            return returns
        
        # คำนวณ Sharpe Ratio
        returns_history = [h['reward'] for h in self.history[-10:]]
        returns_history.append(returns)
        
        mean_return = np.mean(returns_history)
        std_return = np.std(returns_history)
        
        if std_return == 0:
            return 0.0
        
        # Sharpe Ratio = (Mean Return - Risk Free Rate) / Standard Deviation
        # Risk Free Rate = 0 (สมมติ)
        sharpe_ratio = mean_return / std_return
        
        return sharpe_ratio
    
    def _sortino_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        """
        คำนวณรางวัลตาม Sortino Ratio
        
        Parameters:
        simulator (TradingSimulator): อินสแตนซ์ของ TradingSimulator
        current_price (float): ราคาปัจจุบัน
        prev_price (float): ราคาก่อนหน้า
        
        Returns:
        float: รางวัล
        """
        # คำนวณผลตอบแทน
        returns = self._return_reward(simulator, current_price, prev_price)
        
        # ถ้ายังไม่มีข้อมูลพอสำหรับคำนวณ Sortino Ratio
        if len(self.history) < 10:
            return returns
        
        # คำนวณ Sortino Ratio
        returns_history = [h['reward'] for h in self.history[-10:]]
        returns_history.append(returns)
        
        mean_return = np.mean(returns_history)
        
        # เฉพาะผลตอบแทนที่ติดลบ
        negative_returns = [r for r in returns_history if r < 0]
        
        if not negative_returns:
            return mean_return  # ไม่มีผลตอบแทนติดลบ
        
        # Downside Deviation
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))
        
        if downside_deviation == 0:
            return 0.0
        
        # Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation
        # Risk Free Rate = 0 (สมมติ)
        sortino_ratio = mean_return / downside_deviation
        
        return sortino_ratio
    
    def _calmar_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        """
        คำนวณรางวัลตาม Calmar Ratio
        
        Parameters:
        simulator (TradingSimulator): อินสแตนซ์ของ TradingSimulator
        current_price (float): ราคาปัจจุบัน
        prev_price (float): ราคาก่อนหน้า
        
        Returns:
        float: รางวัล
        """
        # คำนวณผลตอบแทน
        returns = self._return_reward(simulator, current_price, prev_price)
        
        # ถ้ายังไม่มีข้อมูลพอสำหรับคำนวณ Calmar Ratio
        if len(self.history) < 10:
            return returns
        
        # คำนวณผลตอบแทนเฉลี่ยรายปี (สมมติว่าเป็นรายวัน)
        returns_history = [h['reward'] for h in self.history[-10:]]
        returns_history.append(returns)
        
        mean_return = np.mean(returns_history) * 252  # Annualized
        
        # คำนวณ Maximum Drawdown
        equity_curve = [h['balance'] + h['profit'] for h in self.history]
        equity_curve.append(simulator.balance + simulator.profit)
        
        max_drawdown = 0.0
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        if max_drawdown == 0:
            return mean_return
        
        # Calmar Ratio = Annualized Return / Maximum Drawdown
        calmar_ratio = mean_return / (max_drawdown * 100)  # แปลง max_drawdown เป็นเปอร์เซ็นต์
        
        return calmar_ratio
    
    def _custom_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        """
        คำนวณรางวัลแบบกำหนดเอง
        
        Parameters:
        simulator (TradingSimulator): อินสแตนซ์ของ TradingSimulator
        current_price (float): ราคาปัจจุบัน
        prev_price (float): ราคาก่อนหน้า
        
        Returns:
        float: รางวัล
        """
        # ผสมผสานระหว่างผลตอบแทนและ Sharpe Ratio
        profit_reward = self._profit_reward(simulator, current_price, prev_price)
        sharpe_reward = self._sharpe_reward(simulator, current_price, prev_price)
        
        # ถ่วงน้ำหนัก 70% ผลตอบแทน, 30% Sharpe Ratio
        return 0.7 * profit_reward + 0.3 * sharpe_reward
    
    def save_history(self, file_path: str) -> bool:
        """
        บันทึกประวัติการเทรดไปยังไฟล์
        
        Parameters:
        file_path (str): พาธไปยังไฟล์ที่จะบันทึก
        
        Returns:
        bool: True ถ้าบันทึกสำเร็จ, False ถ้าไม่สำเร็จ
        """
        try:
            # สร้างโฟลเดอร์หากไม่มีอยู่
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # แปลงวันที่เป็นสตริง
            history = []
            for item in self.history:
                item_copy = item.copy()
                if 'date' in item_copy and item_copy['date'] is not None:
                    if isinstance(item_copy['date'], np.datetime64) or isinstance(item_copy['date'], pd.Timestamp):
                        item_copy['date'] = item_copy['date'].strftime('%Y-%m-%d %H:%M:%S')
                history.append(item_copy)
            
            # บันทึกเป็น JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"บันทึกประวัติการเทรดที่: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกประวัติการเทรด: {e}")
            return False
    
    def load_history(self, file_path: str) -> bool:
        """
        โหลดประวัติการเทรดจากไฟล์
        
        Parameters:
        file_path (str): พาธไปยังไฟล์ที่จะโหลด
        
        Returns:
        bool: True ถ้าโหลดสำเร็จ, False ถ้าไม่สำเร็จ
        """
        try:
            # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
            if not os.path.exists(file_path):
                logger.error(f"ไม่พบไฟล์: {file_path}")
                return False
            
            # โหลด JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # แปลงวันที่เป็น datetime
            for item in history:
                if 'date' in item and item['date'] is not None:
                    if isinstance(item['date'], str):
                        item['date'] = pd.to_datetime(item['date'])
            
            self.history = history
            
            logger.info(f"โหลดประวัติการเทรดจาก: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดประวัติการเทรด: {e}")
            return False
    
    def plot_performance(self, output_file: Optional[str] = None, show: bool = True) -> Optional[plt.Figure]:
        """
        สร้างกราฟแสดงประสิทธิภาพการเทรด
        
        Parameters:
        output_file (str, optional): พาธไปยังไฟล์ที่จะบันทึกกราฟ
        show (bool): แสดงกราฟหรือไม่
        
        Returns:
        plt.Figure or None: กราฟหรือ None ถ้าไม่มีข้อมูล
        """
        if not self.history:
            logger.warning("ไม่มีประวัติการเทรด")
            return None
        
        # สร้างกราฟด้วย matplotlib
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # ข้อมูลสำหรับกราฟ
        steps = [h['step'] for h in self.history]
        dates = [h['date'] for h in self.history]
        prices = [h['price'] for h in self.history]
        balances = [h['balance'] + h['profit'] for h in self.history]
        positions = [h['position'] for h in self.history]
        actions = [h['action'] for h in self.history]
        rewards = [h['reward'] for h in self.history]
        
        # กราฟราคา
        ax1.plot(steps, prices, label='Price', color='blue')
        ax1.set_ylabel('Price')
        ax1.set_title('Trading Performance')
        ax1.grid(True)
        
        # กราฟเงินทุน
        ax1.plot(steps, balances, label='Equity', color='green', alpha=0.7)
        ax1.legend()
        
        # ตำแหน่ง Long และ Short
        for i, pos in enumerate(positions):
            if pos == 'long':
                ax1.scatter(steps[i], prices[i], marker='^', color='green', s=100)
            elif pos == 'short':
                ax1.scatter(steps[i], prices[i], marker='v', color='red', s=100)
        
        # กราฟการกระทำ
        ax2.bar(steps, actions, color='purple', alpha=0.7)
        ax2.set_ylabel('Action')
        ax2.set_yticks(list(self.ACTIONS.values()))
        ax2.set_yticklabels(list(self.ACTIONS.keys()))
        ax2.grid(True)
        
        # กราฟรางวัล
        ax3.bar(steps, rewards, color='orange', alpha=0.7)
        ax3.set_ylabel('Reward')
        ax3.set_xlabel('Step')
        ax3.grid(True)
        
        # ปรับระยะห่างระหว่างกราฟ
        plt.tight_layout()
        
        # บันทึกกราฟ
        if output_file:
            # สร้างโฟลเดอร์หากไม่มีอยู่
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"บันทึกกราฟที่: {output_file}")
        
        # แสดงกราฟ
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig