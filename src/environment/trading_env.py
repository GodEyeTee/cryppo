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
from src.data.data_manager import MarketDataManager
from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class TradingEnv(BaseEnv):
    """
    สภาพแวดล้อมการเทรดสำหรับ Reinforcement Learning
    
    คลาสนี้รับผิดชอบการจำลองสภาพแวดล้อมการเทรดที่เอเจนต์ใช้ในการเทรนและทดสอบอัลกอริทึม RL
    มีการเชื่อมต่อกับ MarketDataManager เพื่อรับข้อมูลตลาด และ TradingSimulator เพื่อจำลองการเทรด
    """
    
    # Action คือ NONE=0, LONG=1, SHORT=2, EXIT=3
    ACTIONS = {
        'NONE': 0,
        'LONG': 1,
        'SHORT': 2,
        'EXIT': 3
    }
    
    def __init__(
        self,
        data_manager: Optional[MarketDataManager] = None,
        window_size: int = 60,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.0025,
        reward_function: Union[str, Callable] = 'sharpe',
        use_position_info: bool = True,
        normalize_obs: bool = True,
        render_mode: Optional[str] = None,
        config = None,
        **kwargs
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับสภาพแวดล้อมการเทรด
        
        Parameters:
        data_manager (MarketDataManager, optional): อินสแตนซ์ของ MarketDataManager
        window_size (int): ขนาดของหน้าต่างข้อมูลในแต่ละการสังเกต
        initial_balance (float): เงินทุนเริ่มต้น
        transaction_fee (float): ค่าธรรมเนียมการทำธุรกรรม (เปอร์เซ็นต์)
        reward_function (str or Callable): ฟังก์ชันสำหรับคำนวณรางวัล
        use_position_info (bool): ใช้ข้อมูลตำแหน่งในการสังเกตหรือไม่
        normalize_obs (bool): ทำ normalization กับข้อมูลการสังเกตหรือไม่
        render_mode (str, optional): โหมดการแสดงผล
        config (Config, optional): อ็อบเจ็กต์การตั้งค่า
        **kwargs: พารามิเตอร์เพิ่มเติม
        """
        # โหลดการตั้งค่า
        self.config = config if config is not None else get_config()
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        env_config = self.config.extract_subconfig("environment")
        
        # กำหนดค่าพารามิเตอร์
        self.window_size = window_size or env_config.get("window_size", 60)
        self.initial_balance = initial_balance or env_config.get("initial_balance", 10000.0)
        self.transaction_fee = transaction_fee or env_config.get("fee_rate", 0.0025)
        self.use_position_info = use_position_info
        self.normalize_obs = normalize_obs
        
        # กำหนดฟังก์ชันรางวัล
        if isinstance(reward_function, str):
            reward_function = reward_function or env_config.get("reward_function", "sharpe")
            self.reward_function = self._get_reward_function(reward_function)
        else:
            self.reward_function = reward_function
        
        # ตั้งค่า data_manager
        if data_manager is None:
            # สร้าง data_manager ใหม่ถ้าไม่ได้ระบุ
            file_path = kwargs.get('file_path')
            symbol = kwargs.get('symbol') or env_config.get("symbol", "BTCUSDT")
            base_timeframe = kwargs.get('base_timeframe') or env_config.get("base_timeframe", "5m")
            detail_timeframe = kwargs.get('detail_timeframe') or env_config.get("detail_timeframe", "1m")
            
            if file_path:
                self.data_manager = MarketDataManager(
                    file_path=file_path,
                    symbol=symbol,
                    base_timeframe=base_timeframe,
                    detail_timeframe=detail_timeframe,
                    window_size=self.window_size,
                    config=self.config
                )
            else:
                # ถ้าไม่มี file_path จะต้องโหลดข้อมูลในภายหลัง
                self.data_manager = None
                logger.warning("ไม่ได้ระบุ data_manager หรือ file_path, จะต้องโหลดข้อมูลก่อนใช้งาน")
        else:
            self.data_manager = data_manager
        
        # ตัวแปรสำหรับเก็บข้อมูลตลาด
        self.current_step = 0
        self.prices = []
        self.dates = []
        self.features = None
        self.obs_shape = None
        
        # สร้าง trading simulator
        self.simulator = TradingSimulator(
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            config=self.config
        )
        
        # สร้าง renderer
        self.renderer = Renderer(render_mode=render_mode, config=self.config)
        
        # ตัวแปรสำหรับเก็บประวัติ
        self.history = []
        
        # กำหนดพื้นที่สังเกตการณ์และพื้นที่การกระทำ
        action_space = spaces.Discrete(len(self.ACTIONS))
        
        # รอการกำหนด observation_space จนกว่าจะโหลดข้อมูล
        observation_space = None
        
        # เรียกคอนสตรักเตอร์ของคลาสแม่
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            render_mode=render_mode,
            config=self.config
        )
        
        logger.info(f"สร้างสภาพแวดล้อมการเทรด (window_size={self.window_size}, balance={self.initial_balance})")
    
    def load_data(self, file_path: str) -> bool:
        """
        โหลดข้อมูลตลาดจากไฟล์
        
        Parameters:
        file_path (str): พาธไปยังไฟล์ข้อมูล
        
        Returns:
        bool: True ถ้าโหลดสำเร็จ, False ถ้าไม่สำเร็จ
        """
        if self.data_manager is None:
            self.data_manager = MarketDataManager(
                file_path=file_path,
                window_size=self.window_size,
                config=self.config
            )
        else:
            self.data_manager.load_data(file_path)
        
        # ตรวจสอบว่าโหลดข้อมูลสำเร็จหรือไม่
        if not self.data_manager.data_loaded:
            logger.error(f"ไม่สามารถโหลดข้อมูลจาก {file_path}")
            return False
        
        # กำหนด observation_space เมื่อโหลดข้อมูลแล้ว
        if self.data_manager.data is not None:
            self._setup_observation_space()
        
        return True
    
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
    
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        รีเซ็ตสภาพแวดล้อมให้กลับสู่สถานะเริ่มต้น
        
        Parameters:
        seed (int, optional): ค่า seed สำหรับการสุ่ม
        options (Dict[str, Any], optional): ตัวเลือกเพิ่มเติมสำหรับการรีเซ็ต
        
        Returns:
        Tuple[np.ndarray, Dict[str, Any]]: สถานะเริ่มต้นและข้อมูลเพิ่มเติม
        """
        # เรียกเมธอด reset ของคลาสแม่
        super().reset(seed, options)
        
        # ตรวจสอบว่าโหลดข้อมูลแล้วหรือไม่
        if self.data_manager is None or not self.data_manager.data_loaded:
            raise ValueError("ต้องโหลดข้อมูลก่อนใช้งานสภาพแวดล้อม")
        
        # ตรวจสอบข้อมูลเพิ่มเติมจาก options
        if options is None:
            options = {}
        
        start_index = options.get('start_index', 0)
        start_date = options.get('start_date')
        
        # ถ้าระบุ start_date ให้หา index ที่ตรงกับวันที่
        if start_date is not None and 'timestamp' in self.data_manager.raw_data.columns:
            start_date = pd.to_datetime(start_date)
            # หา index แรกที่มีวันที่มากกว่าหรือเท่ากับ start_date
            candidate_indices = np.where(self.data_manager.raw_data['timestamp'] >= start_date)[0]
            if len(candidate_indices) > 0:
                start_index = candidate_indices[0]
        
        # ตรวจสอบว่า start_index ถูกต้องหรือไม่
        if start_index < 0:
            start_index = 0
        
        if start_index + self.window_size >= len(self.data_manager.data):
            start_index = len(self.data_manager.data) - self.window_size - 1
        
        # ตั้งค่า current_step
        self.current_step = start_index
        
        # ดึงข้อมูลราคาและวันที่
        if 'close' in self.data_manager.raw_data.columns:
            self.prices = self.data_manager.raw_data['close'].values
        else:
            self.prices = np.zeros(len(self.data_manager.raw_data))
        
        if 'timestamp' in self.data_manager.raw_data.columns:
            self.dates = self.data_manager.raw_data['timestamp'].values
        else:
            self.dates = np.arange(len(self.data_manager.raw_data))
        
        # รีเซ็ต simulator
        self.simulator.reset(initial_balance=self.initial_balance)
        
        # ล้างประวัติ
        self.history = []
        
        # เตรียมการสังเกตเริ่มต้น
        obs = self._get_observation()
        
        # เตรียมข้อมูลเพิ่มเติม
        info = {
            'step': self.current_step,
            'price': self.prices[self.current_step] if self.current_step < len(self.prices) else 0,
            'date': self.dates[self.current_step] if self.current_step < len(self.dates) else None,
            'balance': self.simulator.balance,
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit
        }
        
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
    
    def _process_action(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        ดำเนินการตามการกระทำและคำนวณผลลัพธ์
        
        Parameters:
        action (int): การกระทำที่เลือก (0=NONE, 1=LONG, 2=SHORT, 3=EXIT)
        
        Returns:
        Tuple[np.ndarray, float, bool, Dict[str, Any]]: สถานะใหม่, รางวัล, done, ข้อมูลเพิ่มเติม
        """
        # ราคาปัจจุบันและราคาก่อนหน้า
        current_price = self.prices[self.current_step] if self.current_step < len(self.prices) else 0
        prev_price = self.prices[self.current_step - 1] if self.current_step > 0 and self.current_step - 1 < len(self.prices) else current_price
        
        # ดำเนินการตามการกระทำ
        if action == self.ACTIONS['LONG']:
            # เปิดสถานะ Long
            self.simulator.open_long_position(current_price)
        elif action == self.ACTIONS['SHORT']:
            # เปิดสถานะ Short
            self.simulator.open_short_position(current_price)
        elif action == self.ACTIONS['EXIT']:
            # ปิดสถานะ
            self.simulator.close_position(current_price)
        
        # ปรับสถานะตามการเปลี่ยนแปลงของราคา (สำหรับสถานะที่เปิดอยู่แล้ว)
        self.simulator.update(current_price)
        
        # เลื่อนไปขั้นตอนถัดไป
        self.current_step += 1
        
        # ตรวจสอบว่าจบ episode แล้วหรือไม่
        done = self.current_step >= len(self.data_manager.data) - 1
        
        # ถ้าจบ episode และยังมีสถานะที่เปิดอยู่ ให้ปิดสถานะทั้งหมด
        if done and self.simulator.has_position():
            self.simulator.close_position(current_price)
        
        # คำนวณรางวัล
        reward = self.reward_function(self.simulator, current_price, prev_price)
        
        # เตรียมการสังเกตใหม่
        next_obs = self._get_observation()
        
        # เตรียมข้อมูลเพิ่มเติม
        info = {
            'step': self.current_step,
            'price': current_price,
            'date': self.dates[self.current_step - 1] if self.current_step - 1 < len(self.dates) else None,
            'balance': self.simulator.balance,
            'equity': self.simulator.get_equity(current_price),
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit,
            'return': self.simulator.total_return,
            'action': action
        }
        
        return next_obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        เตรียมข้อมูลการสังเกต (observation) จากสถานะปัจจุบัน
        
        Returns:
        np.ndarray: ข้อมูลการสังเกต
        """
        # ตรวจสอบว่ามีข้อมูลหรือไม่
        if self.data_manager is None or self.data_manager.data is None:
            raise ValueError("ไม่มีข้อมูลสำหรับการสังเกต")
        
        # ขอบเขตของหน้าต่างข้อมูล
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.window_size + 1)
        
        # ดึงข้อมูลตามหน้าต่าง
        if start_idx >= 0 and end_idx < len(self.data_manager.data):
            data = self.data_manager.data.iloc[start_idx:end_idx+1].copy()
        else:
            # ถ้าดัชนีไม่ถูกต้อง ให้ใช้ข้อมูลว่าง
            data = pd.DataFrame(np.zeros((self.window_size, self.data_manager.data.shape[1])), 
                                columns=self.data_manager.data.columns)
        
        # ลบคอลัมน์ timestamp ถ้ามี
        if 'timestamp' in data.columns:
            data = data.drop('timestamp', axis=1)
        
        # แปลงเป็น numpy array
        obs_data = data.values
        
        # ปรับขนาดถ้าจำเป็น
        if len(obs_data) < self.window_size:
            # เติมด้วยค่าศูนย์
            padding = np.zeros((self.window_size - len(obs_data), obs_data.shape[1]))
            obs_data = np.vstack([padding, obs_data])
        
        # เพิ่มข้อมูลตำแหน่ง
        if self.use_position_info:
            # ข้อมูลตำแหน่ง (position_type, units, profit)
            position_info = np.zeros((len(obs_data), 3))
            
            # แปลง position_type เป็น one-hot
            if self.simulator.position_type == 'long':
                position_info[:, 0] = 1
            elif self.simulator.position_type == 'short':
                position_info[:, 0] = -1
            
            # จำนวนหน่วย
            position_info[:, 1] = self.simulator.units
            
            # กำไร/ขาดทุน
            position_info[:, 2] = self.simulator.profit
            
            # รวมข้อมูลตำแหน่งกับข้อมูลตลาด
            obs_data = np.hstack([obs_data, position_info])
        
        return obs_data.astype(np.float32)
    
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
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        แสดงผลสภาพแวดล้อมปัจจุบัน
        
        Returns:
        np.ndarray or str or None: การแสดงผลในรูปแบบต่างๆ ขึ้นอยู่กับ render_mode
        """
        if self.render_mode == "none":
            return None
        
        return self._render_frame()
    
    def close(self) -> None:
        """
        ปิดสภาพแวดล้อมและทรัพยากรที่ใช้
        """
        if self.renderer:
            self.renderer.close()
    
    def _get_reward_function(self, reward_type: str) -> Callable:
        """
        ดึงฟังก์ชันรางวัลตามประเภทที่ระบุ
        
        Parameters:
        reward_type (str): ประเภทของฟังก์ชันรางวัล
        
        Returns:
        Callable: ฟังก์ชันรางวัล
        """
        reward_functions = {
            'profit': self._profit_reward,
            'return': self._return_reward,
            'sharpe': self._sharpe_reward,
            'sortino': self._sortino_reward,
            'calmar': self._calmar_reward,
            'custom': self._custom_reward
        }
        
        if reward_type not in reward_functions:
            logger.warning(f"ไม่พบฟังก์ชันรางวัลประเภท '{reward_type}', ใช้ 'profit' แทน")
            reward_type = 'profit'
        
        logger.info(f"ใช้ฟังก์ชันรางวัลประเภท: {reward_type}")
        return reward_functions[reward_type]
    
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