import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class BaseEnv(gym.Env, ABC):
    """
    คลาสพื้นฐานสำหรับสภาพแวดล้อมการเรียนรู้แบบเสริมกำลัง (Reinforcement Learning)
    
    คลาสนี้รับผิดชอบการกำหนดโครงสร้างพื้นฐานของสภาพแวดล้อมที่ใช้ใน Reinforcement Learning
    โดยสืบทอดจาก gym.Env ของ Gymnasium และเป็นคลาสแม่ของสภาพแวดล้อมอื่นๆ
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'console', 'none']}
    
    def __init__(
        self,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        reward_range: Tuple[float, float] = (-float('inf'), float('inf')),
        render_mode: Optional[str] = None,
        config = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับ BaseEnv
        
        Parameters:
        observation_space (spaces.Space, optional): พื้นที่สังเกตการณ์ (observation space)
        action_space (spaces.Space, optional): พื้นที่การกระทำ (action space)
        reward_range (Tuple[float, float]): ช่วงของรางวัล
        render_mode (str, optional): โหมดการแสดงผล ('human', 'rgb_array', 'console', 'none')
        config (Config, optional): อ็อบเจ็กต์การตั้งค่า
        """
        # โหลดการตั้งค่า
        self.config = config if config is not None else get_config()
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        env_config = self.config.extract_subconfig("environment")
        
        # กำหนดค่าพื้นฐาน
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        
        # ตรวจสอบว่า render_mode ถูกต้องหรือไม่
        if render_mode is None:
            render_mode = env_config.get("render_mode", "none")
            
        if render_mode not in self.metadata['render_modes']:
            logger.warning(f"โหมดการแสดงผล '{render_mode}' ไม่ถูกต้อง, ใช้ 'none' แทน")
            render_mode = "none"
            
        self.render_mode = render_mode
        
        # ตัวแปรสำหรับเก็บข้อมูลสถานะของสภาพแวดล้อม
        self.state = None
        self.done = False
        self.info = {}
        self.steps = 0
        self.episode = 0
        self.total_rewards = 0.0
        
        # ตัวแปรสำหรับการจำกัดจำนวนขั้นตอนสูงสุดในแต่ละ episode
        self.max_episode_steps = env_config.get("max_episode_steps", None)
        
        # ตัวแปรสำหรับ seed
        self.np_random = None
        self.seed()
        
        logger.info(f"สร้างสภาพแวดล้อมพื้นฐาน (render_mode={render_mode})")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        กำหนดค่า seed สำหรับการสุ่ม
        
        Parameters:
        seed (int, optional): ค่า seed สำหรับการสุ่ม
        
        Returns:
        List[int]: รายการของ seed ที่ใช้
        """
        if seed is None:
            seed = self.config.get("general.random_seed", None)
            
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        รีเซ็ตสภาพแวดล้อมให้กลับสู่สถานะเริ่มต้น
        
        Parameters:
        seed (int, optional): ค่า seed สำหรับการสุ่ม
        options (Dict[str, Any], optional): ตัวเลือกเพิ่มเติมสำหรับการรีเซ็ต
        
        Returns:
        Tuple[Any, Dict[str, Any]]: สถานะเริ่มต้นและข้อมูลเพิ่มเติม
        """
        # รีเซ็ตตัวแปรต่างๆ
        self.done = False
        self.steps = 0
        self.episode += 1
        self.total_rewards = 0.0
        
        # กำหนดค่า seed ถ้าระบุ
        if seed is not None:
            self.seed(seed)
        
        # เตรียมสถานะเริ่มต้น (จะถูก override ในคลาสลูก)
        self.state = self._get_initial_state(options)
        
        # เตรียมข้อมูลเพิ่มเติม
        self.info = {}
        
        return self.state, self.info
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        ดำเนินการหนึ่งขั้นตอนในสภาพแวดล้อมตามการกระทำที่กำหนด
        
        Parameters:
        action (Any): การกระทำที่เลือก
        
        Returns:
        Tuple[Any, float, bool, bool, Dict[str, Any]]: สถานะใหม่, รางวัล, done, truncated, ข้อมูลเพิ่มเติม
        """
        # เพิ่มจำนวนขั้นตอน
        self.steps += 1
        
        # ตรวจสอบว่าถึงจำนวนขั้นตอนสูงสุดหรือไม่
        truncated = False
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True
        
        # ถ้า episode จบแล้ว (self.done == True) แต่ยังเรียก step อีก
        if self.done:
            logger.warning("เรียก step() ในขณะที่ episode จบแล้ว, รีเซ็ตสภาพแวดล้อมก่อน")
            return self.state, 0.0, self.done, truncated, self.info
        
        # ดำเนินการตามการกระทำ (จะถูก override ในคลาสลูก)
        next_state, reward, done, info = self._process_action(action)
        
        # อัพเดตตัวแปรต่างๆ
        self.state = next_state
        self.done = done
        self.info = info
        self.total_rewards += reward
        
        # เพิ่มข้อมูลเกี่ยวกับสถานะของสภาพแวดล้อมลงใน info
        info.update({
            'steps': self.steps,
            'episode': self.episode,
            'total_rewards': self.total_rewards
        })
        
        return self.state, reward, done, truncated, info
    
    @abstractmethod
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        แสดงผลสภาพแวดล้อมปัจจุบัน
        
        Returns:
        np.ndarray or str or None: การแสดงผลในรูปแบบต่างๆ ขึ้นอยู่กับ render_mode
        """
        if self.render_mode == "none":
            return None
        
        # สร้างการแสดงผล (จะถูก override ในคลาสลูก)
        return self._render_frame()
    
    @abstractmethod
    def close(self) -> None:
        """
        ปิดสภาพแวดล้อมและทรัพยากรที่ใช้
        """
        pass
    
    @abstractmethod
    def _get_initial_state(self, options: Dict[str, Any] = None) -> Any:
        """
        เตรียมสถานะเริ่มต้นของสภาพแวดล้อม
        
        Parameters:
        options (Dict[str, Any], optional): ตัวเลือกเพิ่มเติมสำหรับการรีเซ็ต
        
        Returns:
        Any: สถานะเริ่มต้น
        """
        pass
    
    @abstractmethod
    def _process_action(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        ดำเนินการตามการกระทำและคำนวณผลลัพธ์
        
        Parameters:
        action (Any): การกระทำที่เลือก
        
        Returns:
        Tuple[Any, float, bool, Dict[str, Any]]: สถานะใหม่, รางวัล, done, ข้อมูลเพิ่มเติม
        """
        pass
    
    @abstractmethod
    def _render_frame(self) -> Optional[Union[np.ndarray, str]]:
        """
        สร้างเฟรมสำหรับการแสดงผล
        
        Returns:
        np.ndarray or str or None: เฟรมสำหรับการแสดงผล
        """
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        ดึงข้อมูลสถานะของสภาพแวดล้อมในรูปแบบ dict
        
        Returns:
        Dict[str, Any]: ข้อมูลสถานะของสภาพแวดล้อม
        """
        return {
            'state': self.state,
            'done': self.done,
            'info': self.info,
            'steps': self.steps,
            'episode': self.episode,
            'total_rewards': self.total_rewards
        }
    
    def set_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        ตั้งค่าสถานะของสภาพแวดล้อมจาก dict
        
        Parameters:
        state_dict (Dict[str, Any]): ข้อมูลสถานะของสภาพแวดล้อม
        """
        self.state = state_dict.get('state', self.state)
        self.done = state_dict.get('done', self.done)
        self.info = state_dict.get('info', self.info)
        self.steps = state_dict.get('steps', self.steps)
        self.episode = state_dict.get('episode', self.episode)
        self.total_rewards = state_dict.get('total_rewards', self.total_rewards)
    
    def __str__(self) -> str:
        """
        แปลงสภาพแวดล้อมเป็นสตริง
        
        Returns:
        str: ข้อมูลสภาพแวดล้อมในรูปแบบสตริง
        """
        return f"{self.__class__.__name__}(obs_space={self.observation_space}, action_space={self.action_space}, render_mode={self.render_mode})"