import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import random
import math

class EpsilonGreedyPolicy:
    """
    นโยบาย Epsilon-Greedy สำหรับการสำรวจในการเรียนรู้แบบเสริมกำลัง
    
    นโยบายนี้จะสุ่มทำการกระทำแบบสุ่มด้วยความน่าจะเป็น epsilon
    และทำการกระทำที่ดีที่สุดด้วยความน่าจะเป็น 1-epsilon
    """
    
    def __init__(
        self, 
        epsilon_start: float = 1.0, 
        epsilon_end: float = 0.01, 
        epsilon_decay: float = 0.995, 
        warmup_steps: int = 0
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับนโยบาย Epsilon-Greedy
        
        Parameters:
        epsilon_start (float): ค่า epsilon เริ่มต้น (ความน่าจะเป็นในการสุ่ม)
        epsilon_end (float): ค่า epsilon สุดท้าย
        epsilon_decay (float): อัตราการลดลงของ epsilon
        warmup_steps (int): จำนวนขั้นตอนที่ให้ทำการสุ่มแบบเต็มที่ (epsilon = 1.0)
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.warmup_steps = warmup_steps
        
        self.epsilon = epsilon_start
        self.steps = 0
    
    def select_action(
        self, 
        q_values: Union[np.ndarray, torch.Tensor], 
        evaluate: bool = False
    ) -> int:
        """
        เลือกการกระทำตามนโยบาย Epsilon-Greedy
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        evaluate (bool): โหมดประเมิน (ไม่มีการสำรวจ)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ไม่มีการสำรวจ
        if evaluate:
            return np.argmax(q_values)
        
        # ในระหว่าง warmup ทำการสุ่มแบบเต็มที่
        if self.steps < self.warmup_steps:
            self.steps += 1
            return random.randint(0, len(q_values) - 1)
        
        # สุ่มตามค่า epsilon
        if random.random() < self.epsilon:
            # สำรวจ
            return random.randint(0, len(q_values) - 1)
        else:
            # ใช้ประโยชน์
            return np.argmax(q_values)
    
    def update(self, steps: Optional[int] = None):
        """
        อัพเดตค่า epsilon
        
        Parameters:
        steps (int, optional): จำนวนขั้นตอนที่ผ่านไป (ถ้าไม่ระบุจะใช้ค่าภายใน)
        """
        if steps is not None:
            self.steps = steps
        else:
            self.steps += 1
        
        # ลดค่า epsilon หลังจาก warmup
        if self.steps >= self.warmup_steps:
            self.epsilon = max(
                self.epsilon_end, 
                self.epsilon_start * (self.epsilon_decay ** (self.steps - self.warmup_steps))
            )
    
    def get_epsilon(self) -> float:
        """
        ดึงค่า epsilon ปัจจุบัน
        
        Returns:
        float: ค่า epsilon ปัจจุบัน
        """
        return self.epsilon

class SoftmaxPolicy:
    """
    นโยบาย Softmax (Boltzmann) สำหรับการสำรวจในการเรียนรู้แบบเสริมกำลัง
    
    นโยบายนี้จะแปลงค่า Q-values เป็นความน่าจะเป็นโดยใช้ฟังก์ชัน softmax
    """
    
    def __init__(self, temperature_start: float = 1.0, temperature_end: float = 0.1, temperature_decay: float = 0.995):
        """
        กำหนดค่าเริ่มต้นสำหรับนโยบาย Softmax
        
        Parameters:
        temperature_start (float): ค่า temperature เริ่มต้น (ค่าสูงจะทำให้การเลือกมีความสุ่มมากขึ้น)
        temperature_end (float): ค่า temperature สุดท้าย
        temperature_decay (float): อัตราการลดลงของ temperature
        """
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        
        self.temperature = temperature_start
        self.steps = 0
    
    def select_action(
        self, 
        q_values: Union[np.ndarray, torch.Tensor], 
        evaluate: bool = False
    ) -> int:
        """
        เลือกการกระทำตามนโยบาย Softmax
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        evaluate (bool): โหมดประเมิน (ใช้ temperature ต่ำ)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ใช้ temperature ต่ำ
        if evaluate:
            return np.argmax(q_values)
        
        # คำนวณความน่าจะเป็นด้วย softmax
        probs = self._softmax(q_values, self.temperature)
        
        # สุ่มการกระทำตามความน่าจะเป็น
        return np.random.choice(len(probs), p=probs)
    
    def _softmax(self, x: np.ndarray, temperature: float) -> np.ndarray:
        """
        คำนวณฟังก์ชัน softmax ด้วย temperature
        
        Parameters:
        x (np.ndarray): อินพุต
        temperature (float): temperature
        
        Returns:
        np.ndarray: ความน่าจะเป็น
        """
        # หลีกเลี่ยงการหารด้วย zero temperature
        temperature = max(temperature, 1e-8)
        
        # คำนวณ softmax
        x_temp = x / temperature
        exp_x = np.exp(x_temp - np.max(x_temp))  # ลบค่าสูงสุดเพื่อความเสถียรภาพทางตัวเลข
        return exp_x / np.sum(exp_x)
    
    def update(self, steps: Optional[int] = None):
        """
        อัพเดตค่า temperature
        
        Parameters:
        steps (int, optional): จำนวนขั้นตอนที่ผ่านไป (ถ้าไม่ระบุจะใช้ค่าภายใน)
        """
        if steps is not None:
            self.steps = steps
        else:
            self.steps += 1
        
        # ลดค่า temperature
        self.temperature = max(
            self.temperature_end, 
            self.temperature_start * (self.temperature_decay ** self.steps)
        )
    
    def get_temperature(self) -> float:
        """
        ดึงค่า temperature ปัจจุบัน
        
        Returns:
        float: ค่า temperature ปัจจุบัน
        """
        return self.temperature

class UCBPolicy:
    """
    นโยบาย Upper Confidence Bound (UCB) สำหรับการสำรวจในการเรียนรู้แบบเสริมกำลัง
    
    นโยบายนี้จะเลือกการกระทำที่มีค่า Q-value บวกกับค่าความไม่แน่นอนสูงสุด
    """
    
    def __init__(self, c: float = 1.0):
        """
        กำหนดค่าเริ่มต้นสำหรับนโยบาย UCB
        
        Parameters:
        c (float): พารามิเตอร์การสำรวจ (c สูงจะให้ความสำคัญกับการสำรวจมากขึ้น)
        """
        self.c = c
        self.action_counts = None
        self.total_count = 0
    
    def select_action(
        self, 
        q_values: Union[np.ndarray, torch.Tensor], 
        evaluate: bool = False
    ) -> int:
        """
        เลือกการกระทำตามนโยบาย UCB
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        evaluate (bool): โหมดประเมิน (ไม่ใช้ UCB)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ไม่ใช้ UCB
        if evaluate:
            return np.argmax(q_values)
        
        # เริ่มต้น action_counts ถ้ายังไม่ได้กำหนด
        if self.action_counts is None:
            self.action_counts = np.zeros(len(q_values))
        
        # คำนวณ UCB
        self.total_count += 1
        exploration = np.zeros_like(q_values)
        
        for i in range(len(q_values)):
            if self.action_counts[i] > 0:
                exploration[i] = self.c * np.sqrt(np.log(self.total_count) / self.action_counts[i])
            else:
                exploration[i] = float('inf')  # ทำให้การกระทำที่ยังไม่เคยทำมีค่าสูงมาก
        
        # เลือกการกระทำที่มีค่า UCB สูงสุด
        ucb_values = q_values + exploration
        action = np.argmax(ucb_values)
        
        # อัพเดตจำนวนครั้งของการกระทำ
        self.action_counts[action] += 1
        
        return action
    
    def update(self, action: int):
        """
        อัพเดตจำนวนครั้งของการกระทำ
        
        Parameters:
        action (int): การกระทำที่เลือก
        """
        if self.action_counts is not None:
            self.action_counts[action] += 1
            self.total_count += 1
    
    def reset(self):
        """
        รีเซ็ตนโยบาย
        """
        self.action_counts = None
        self.total_count = 0

class BoltzmannPolicy:
    """
    นโยบาย Boltzmann Exploration สำหรับการสำรวจในการเรียนรู้แบบเสริมกำลัง
    
    นโยบายนี้จะเลือกการกระทำตามความน่าจะเป็นที่คำนวณจาก Boltzmann distribution
    """
    
    def __init__(
        self, 
        temperature_start: float = 1.0, 
        temperature_end: float = 0.1, 
        temperature_decay: float = 0.995, 
        state_dependent: bool = False, 
        network_fn: Optional[Callable] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับนโยบาย Boltzmann
        
        Parameters:
        temperature_start (float): ค่า temperature เริ่มต้น
        temperature_end (float): ค่า temperature สุดท้าย
        temperature_decay (float): อัตราการลดลงของ temperature
        state_dependent (bool): ใช้ temperature ที่ขึ้นกับสถานะหรือไม่
        network_fn (Callable, optional): ฟังก์ชันสำหรับคำนวณ temperature จากสถานะ
        """
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.state_dependent = state_dependent
        self.network_fn = network_fn
        
        self.temperature = temperature_start
        self.steps = 0
    
    def select_action(
        self, 
        q_values: Union[np.ndarray, torch.Tensor], 
        state: Optional[Union[np.ndarray, torch.Tensor]] = None,
        evaluate: bool = False
    ) -> int:
        """
        เลือกการกระทำตามนโยบาย Boltzmann
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        state (np.ndarray or torch.Tensor, optional): สถานะปัจจุบัน (สำหรับ state-dependent temperature)
        evaluate (bool): โหมดประเมิน (ใช้ temperature ต่ำ)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ใช้ temperature ต่ำ
        if evaluate:
            return np.argmax(q_values)
        
        # คำนวณ temperature
        if self.state_dependent and state is not None and self.network_fn is not None:
            temperature = self.network_fn(state)
            temperature = max(self.temperature_end, min(self.temperature_start, temperature))
        else:
            temperature = self.temperature
        
        # คำนวณความน่าจะเป็นด้วย Boltzmann distribution
        probs = self._boltzmann(q_values, temperature)
        
        # สุ่มการกระทำตามความน่าจะเป็น
        return np.random.choice(len(probs), p=probs)
    
    def _boltzmann(self, q_values: np.ndarray, temperature: float) -> np.ndarray:
        """
        คำนวณความน่าจะเป็นด้วย Boltzmann distribution
        
        Parameters:
        q_values (np.ndarray): ค่า Q-values
        temperature (float): temperature
        
        Returns:
        np.ndarray: ความน่าจะเป็น
        """
        # หลีกเลี่ยงการหารด้วย zero temperature
        temperature = max(temperature, 1e-8)
        
        # คำนวณ Boltzmann distribution
        q_temp = q_values / temperature
        exp_q = np.exp(q_temp - np.max(q_temp))  # ลบค่าสูงสุดเพื่อความเสถียรภาพทางตัวเลข
        probs = exp_q / np.sum(exp_q)
        
        return probs
    
    def update(self, steps: Optional[int] = None):
        """
        อัพเดตค่า temperature
        
        Parameters:
        steps (int, optional): จำนวนขั้นตอนที่ผ่านไป (ถ้าไม่ระบุจะใช้ค่าภายใน)
        """
        if steps is not None:
            self.steps = steps
        else:
            self.steps += 1
        
        # ลดค่า temperature
        self.temperature = max(
            self.temperature_end, 
            self.temperature_start * (self.temperature_decay ** self.steps)
        )
    
    def get_temperature(self) -> float:
        """
        ดึงค่า temperature ปัจจุบัน
        
        Returns:
        float: ค่า temperature ปัจจุบัน
        """
        return self.temperature