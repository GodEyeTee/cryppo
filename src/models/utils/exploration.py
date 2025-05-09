import numpy as np
import torch
import random
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger('models.utils.exploration')

class ExplorationStrategy:
    """
    คลาสพื้นฐานสำหรับกลยุทธ์การสำรวจ (exploration strategy)
    
    กลยุทธ์การสำรวจใช้ในการตัดสินใจระหว่างการใช้ประโยชน์จากความรู้ที่มีอยู่ (exploitation)
    กับการสำรวจเพื่อหาข้อมูลใหม่ (exploration) ในการเรียนรู้แบบเสริมกำลัง
    """
    
    def __init__(self):
        """
        กำหนดค่าเริ่มต้นสำหรับกลยุทธ์การสำรวจ
        """
        self.step_count = 0
    
    def select_action(self, q_values: Union[np.ndarray, torch.Tensor], state: Optional[np.ndarray] = None, 
                      evaluation: bool = False) -> int:
        """
        เลือกการกระทำตามกลยุทธ์การสำรวจ
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        state (np.ndarray, optional): สถานะปัจจุบัน (ใช้กับบางกลยุทธ์)
        evaluation (bool): โหมดประเมิน (ไม่มีการสำรวจสุ่ม)
        
        Returns:
        int: การกระทำที่เลือก
        """
        raise NotImplementedError("คลาสลูกต้องทำการ implement เมธอดนี้")
    
    def update(self, step: Optional[int] = None):
        """
        อัพเดตพารามิเตอร์ของกลยุทธ์การสำรวจ
        
        Parameters:
        step (int, optional): ขั้นตอนปัจจุบัน
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
    
    def reset(self):
        """
        รีเซ็ตกลยุทธ์การสำรวจ
        """
        self.step_count = 0

class EpsilonGreedyExploration(ExplorationStrategy):
    """
    กลยุทธ์การสำรวจแบบ Epsilon-Greedy
    
    กลยุทธ์นี้จะสุ่มทำการกระทำแบบสุ่มด้วยความน่าจะเป็น epsilon 
    และทำการกระทำที่ดีที่สุดด้วยความน่าจะเป็น 1-epsilon
    โดย epsilon จะลดลงเรื่อยๆ ตามเวลา (epsilon decay)
    """
    
    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 epsilon_decay: float = 0.995, warmup_steps: int = 0, 
                 decay_type: str = 'exponential'):
        """
        กำหนดค่าเริ่มต้นสำหรับกลยุทธ์ Epsilon-Greedy
        
        Parameters:
        epsilon_start (float): ค่า epsilon เริ่มต้น (ความน่าจะเป็นในการสุ่ม)
        epsilon_end (float): ค่า epsilon สุดท้าย
        epsilon_decay (float): อัตราการลดลงของ epsilon
        warmup_steps (int): จำนวนขั้นตอนที่ให้ทำการสุ่มแบบเต็มที่ (epsilon = 1.0)
        decay_type (str): ประเภทของการลดลง ('exponential', 'linear')
        """
        super().__init__()
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.warmup_steps = warmup_steps
        self.decay_type = decay_type.lower()
        
        self.epsilon = epsilon_start
        
        # คำนวณพารามิเตอร์สำหรับการลดลงแบบเชิงเส้น
        if self.decay_type == 'linear':
            # คำนวณระยะเวลาในการลดลงจาก epsilon_start ไปถึง epsilon_end
            # โดยใช้แทนสูตร: epsilon = epsilon_start - (step - warmup_steps) * decay_rate
            # เมื่อ decay_rate = (epsilon_start - epsilon_end) / decay_length
            # และต้องการให้ epsilon = epsilon_end เมื่อ step = decay_length + warmup_steps
            # จึงได้ decay_rate = (epsilon_start - epsilon_end) / decay_length
            # เมื่อ decay_length = 1 / (1 - epsilon_decay) โดยประมาณ
            
            # จำนวนขั้นตอนที่ใช้ในการลดลงแบบ exponential เพื่อให้ epsilon ลดลงจาก epsilon_start เป็น epsilon_end
            decay_length = int(math.log(epsilon_end / epsilon_start) / math.log(epsilon_decay))
            
            # อัตราการลดลงแบบเชิงเส้น
            self.linear_decay_rate = (epsilon_start - epsilon_end) / decay_length
        
        logger.info(f"สร้างกลยุทธ์ Epsilon-Greedy (start={epsilon_start}, end={epsilon_end}, type={decay_type})")
    
    def select_action(self, q_values: Union[np.ndarray, torch.Tensor], state: Optional[np.ndarray] = None,
                      evaluation: bool = False) -> int:
        """
        เลือกการกระทำตามกลยุทธ์ Epsilon-Greedy
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        state (np.ndarray, optional): สถานะปัจจุบัน (ไม่ได้ใช้)
        evaluation (bool): โหมดประเมิน (ไม่มีการสำรวจสุ่ม)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ไม่มีการสำรวจสุ่ม
        if evaluation:
            return np.argmax(q_values)
        
        # ในระหว่าง warmup ทำการสุ่มแบบเต็มที่
        if self.step_count < self.warmup_steps:
            return random.randint(0, len(q_values) - 1)
        
        # สุ่มตามค่า epsilon
        if random.random() < self.epsilon:
            # สำรวจ
            return random.randint(0, len(q_values) - 1)
        else:
            # ใช้ประโยชน์
            return np.argmax(q_values)
    
    def update(self, step: Optional[int] = None):
        """
        อัพเดตค่า epsilon
        
        Parameters:
        step (int, optional): ขั้นตอนปัจจุบัน
        """
        super().update(step)
        
        # ลดค่า epsilon หลังจาก warmup
        if self.step_count >= self.warmup_steps:
            if self.decay_type == 'exponential':
                # การลดลงแบบ exponential
                self.epsilon = max(
                    self.epsilon_end, 
                    self.epsilon_start * (self.epsilon_decay ** (self.step_count - self.warmup_steps))
                )
            elif self.decay_type == 'linear':
                # การลดลงแบบเชิงเส้น
                self.epsilon = max(
                    self.epsilon_end,
                    self.epsilon_start - self.linear_decay_rate * (self.step_count - self.warmup_steps)
                )
    
    def get_epsilon(self) -> float:
        """
        ดึงค่า epsilon ปัจจุบัน
        
        Returns:
        float: ค่า epsilon ปัจจุบัน
        """
        return self.epsilon

class BoltzmannExploration(ExplorationStrategy):
    """
    กลยุทธ์การสำรวจแบบ Boltzmann (Softmax)
    
    กลยุทธ์นี้แปลงค่า Q-values เป็นความน่าจะเป็นโดยใช้ฟังก์ชัน softmax กับพารามิเตอร์ temperature
    โดย temperature สูงจะทำให้การเลือกมีความสุ่มมากขึ้น และ temperature ต่ำจะทำให้การเลือกเป็นแบบ greedy มากขึ้น
    """
    
    def __init__(self, temperature_start: float = 1.0, temperature_end: float = 0.1,
                 temperature_decay: float = 0.995, decay_type: str = 'exponential'):
        """
        กำหนดค่าเริ่มต้นสำหรับกลยุทธ์ Boltzmann
        
        Parameters:
        temperature_start (float): ค่า temperature เริ่มต้น (ค่าสูงจะทำให้การเลือกมีความสุ่มมากขึ้น)
        temperature_end (float): ค่า temperature สุดท้าย
        temperature_decay (float): อัตราการลดลงของ temperature
        decay_type (str): ประเภทของการลดลง ('exponential', 'linear')
        """
        super().__init__()
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.decay_type = decay_type.lower()
        
        self.temperature = temperature_start
        
        # คำนวณพารามิเตอร์สำหรับการลดลงแบบเชิงเส้น
        if self.decay_type == 'linear':
            # คำนวณระยะเวลาในการลดลงแบบเดียวกับ EpsilonGreedyExploration
            decay_length = int(math.log(temperature_end / temperature_start) / math.log(temperature_decay))
            self.linear_decay_rate = (temperature_start - temperature_end) / decay_length
        
        logger.info(f"สร้างกลยุทธ์ Boltzmann (start={temperature_start}, end={temperature_end}, type={decay_type})")
    
    def select_action(self, q_values: Union[np.ndarray, torch.Tensor], state: Optional[np.ndarray] = None,
                      evaluation: bool = False) -> int:
        """
        เลือกการกระทำตามกลยุทธ์ Boltzmann
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        state (np.ndarray, optional): สถานะปัจจุบัน (ไม่ได้ใช้)
        evaluation (bool): โหมดประเมิน (ใช้ temperature ต่ำ)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ใช้ temperature ต่ำ (greedy)
        if evaluation:
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
    
    def update(self, step: Optional[int] = None):
        """
        อัพเดตค่า temperature
        
        Parameters:
        step (int, optional): ขั้นตอนปัจจุบัน
        """
        super().update(step)
        
        # ลดค่า temperature
        if self.decay_type == 'exponential':
            # การลดลงแบบ exponential
            self.temperature = max(
                self.temperature_end, 
                self.temperature_start * (self.temperature_decay ** self.step_count)
            )
        elif self.decay_type == 'linear':
            # การลดลงแบบเชิงเส้น
            self.temperature = max(
                self.temperature_end,
                self.temperature_start - self.linear_decay_rate * self.step_count
            )
    
    def get_temperature(self) -> float:
        """
        ดึงค่า temperature ปัจจุบัน
        
        Returns:
        float: ค่า temperature ปัจจุบัน
        """
        return self.temperature

class UCBExploration(ExplorationStrategy):
    """
    กลยุทธ์การสำรวจแบบ Upper Confidence Bound (UCB)
    
    กลยุทธ์นี้จะเลือกการกระทำที่มีค่า Q-value บวกกับค่าความไม่แน่นอนสูงสุด
    โดยค่าความไม่แน่นอนจะสูงสำหรับการกระทำที่ถูกเลือกน้อย
    """
    
    def __init__(self, c: float = 1.0, decay: float = 0.9999):
        """
        กำหนดค่าเริ่มต้นสำหรับกลยุทธ์ UCB
        
        Parameters:
        c (float): พารามิเตอร์การสำรวจ (c สูงจะให้ความสำคัญกับการสำรวจมากขึ้น)
        decay (float): อัตราการลดลงของพารามิเตอร์การสำรวจ
        """
        super().__init__()
        self.c_start = c
        self.c = c
        self.decay = decay
        self.action_counts = None
        self.total_count = 0
        
        logger.info(f"สร้างกลยุทธ์ UCB (c={c}, decay={decay})")
    
    def select_action(self, q_values: Union[np.ndarray, torch.Tensor], state: Optional[np.ndarray] = None,
                      evaluation: bool = False) -> int:
        """
        เลือกการกระทำตามกลยุทธ์ UCB
        
        Parameters:
        q_values (np.ndarray or torch.Tensor): ค่า Q-values ของแต่ละการกระทำ
        state (np.ndarray, optional): สถานะปัจจุบัน (ไม่ได้ใช้)
        evaluation (bool): โหมดประเมิน (ไม่ใช้ UCB)
        
        Returns:
        int: การกระทำที่เลือก
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(q_values, torch.Tensor):
            q_values = q_values.cpu().numpy()
        
        # ในโหมดประเมิน ไม่ใช้ UCB
        if evaluation:
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
    
    def update(self, step: Optional[int] = None, action: Optional[int] = None):
        """
        อัพเดตพารามิเตอร์ของกลยุทธ์ UCB
        
        Parameters:
        step (int, optional): ขั้นตอนปัจจุบัน
        action (int, optional): การกระทำที่เลือก
        """
        super().update(step)
        
        # อัพเดตจำนวนครั้งของการกระทำ
        if action is not None and self.action_counts is not None:
            self.action_counts[action] += 1
            self.total_count += 1
        
        # ลดค่า c
        self.c = self.c_start * (self.decay ** self.step_count)
    
    def get_c(self) -> float:
        """
        ดึงค่า c ปัจจุบัน
        
        Returns:
        float: ค่า c ปัจจุบัน
        """
        return self.c
    
    def reset(self):
        """
        รีเซ็ตกลยุทธ์ UCB
        """
        super().reset()
        self.action_counts = None
        self.total_count = 0
        self.c = self.c_start

class NoiseBasedExploration(ExplorationStrategy):
    """
    กลยุทธ์การสำรวจโดยใช้การเพิ่มสัญญาณรบกวน (noise) เข้าไปในการกระทำ
    
    เหมาะสำหรับ action space แบบต่อเนื่อง (continuous) เช่น DDPG, TD3, SAC
    """
    
    def __init__(self, noise_type: str = 'gaussian', sigma_start: float = 0.2, 
                 sigma_end: float = 0.01, sigma_decay: float = 0.995,
                 noise_clip: Optional[float] = None):
        """
        กำหนดค่าเริ่มต้นสำหรับกลยุทธ์ NoiseBasedExploration
        
        Parameters:
        noise_type (str): ประเภทของสัญญาณรบกวน ('gaussian', 'ou')
        sigma_start (float): ค่า sigma (ความแปรปรวน) เริ่มต้น
        sigma_end (float): ค่า sigma สุดท้าย
        sigma_decay (float): อัตราการลดลงของ sigma
        noise_clip (float, optional): ค่าสูงสุดของสัญญาณรบกวน
        """
        super().__init__()
        self.noise_type = noise_type.lower()
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_decay = sigma_decay
        self.noise_clip = noise_clip
        
        self.sigma = sigma_start
        
        # พารามิเตอร์สำหรับ Ornstein-Uhlenbeck process
        if self.noise_type == 'ou':
            self.ou_theta = 0.15  # พารามิเตอร์ความแรงในการดึงกลับค่าเฉลี่ย
            self.ou_mu = 0.0      # ค่าเฉลี่ย (mean)
            self.ou_state = None  # สถานะของ OU process
        
        logger.info(f"สร้างกลยุทธ์ NoiseBasedExploration (type={noise_type}, sigma={sigma_start}, clip={noise_clip})")
    
    def select_action(self, action: Union[np.ndarray, torch.Tensor], state: Optional[np.ndarray] = None,
                      evaluation: bool = False) -> np.ndarray:
        """
        เพิ่มสัญญาณรบกวนเข้าไปในการกระทำ
        
        Parameters:
        action (np.ndarray or torch.Tensor): การกระทำที่เลือก (ผลลัพธ์จากโมเดล)
        state (np.ndarray, optional): สถานะปัจจุบัน (ไม่ได้ใช้)
        evaluation (bool): โหมดประเมิน (ไม่เพิ่มสัญญาณรบกวน)
        
        Returns:
        np.ndarray: การกระทำที่เพิ่มสัญญาณรบกวนแล้ว
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # ในโหมดประเมิน ไม่เพิ่มสัญญาณรบกวน
        if evaluation:
            return action
        
        # สร้างสัญญาณรบกวน
        if self.noise_type == 'gaussian':
            # สัญญาณรบกวนแบบ Gaussian
            noise = np.random.normal(0, self.sigma, size=action.shape)
        elif self.noise_type == 'ou':
            # สัญญาณรบกวนแบบ Ornstein-Uhlenbeck
            noise = self._ou_noise(action.shape)
        else:
            # ไม่รู้จักประเภทของสัญญาณรบกวน
            logger.warning(f"ไม่รู้จักประเภทของสัญญาณรบกวน: {self.noise_type}, ใช้ Gaussian แทน")
            noise = np.random.normal(0, self.sigma, size=action.shape)
        
        # จำกัดสัญญาณรบกวน (ถ้ากำหนด)
        if self.noise_clip is not None:
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        
        # เพิ่มสัญญาณรบกวนเข้าไปในการกระทำ
        noisy_action = action + noise
        
        return noisy_action
    
    def _ou_noise(self, shape: Tuple) -> np.ndarray:
        """
        สร้างสัญญาณรบกวนแบบ Ornstein-Uhlenbeck
        
        Parameters:
        shape (Tuple): รูปร่างของสัญญาณรบกวน
        
        Returns:
        np.ndarray: สัญญาณรบกวนแบบ Ornstein-Uhlenbeck
        """
        # สร้างสถานะเริ่มต้นถ้ายังไม่มี
        if self.ou_state is None:
            self.ou_state = np.zeros(shape)
        
        # อัพเดตสถานะ
        dx = self.ou_theta * (self.ou_mu - self.ou_state) + self.sigma * np.random.normal(0, 1, size=shape)
        self.ou_state += dx
        
        return self.ou_state
    
    def update(self, step: Optional[int] = None):
        """
        อัพเดตค่า sigma
        
        Parameters:
        step (int, optional): ขั้นตอนปัจจุบัน
        """
        super().update(step)
        
        # ลดค่า sigma
        self.sigma = max(
            self.sigma_end, 
            self.sigma_start * (self.sigma_decay ** self.step_count)
        )
    
    def get_sigma(self) -> float:
        """
        ดึงค่า sigma ปัจจุบัน
        
        Returns:
        float: ค่า sigma ปัจจุบัน
        """
        return self.sigma
    
    def reset(self):
        """
        รีเซ็ตกลยุทธ์ NoiseBasedExploration
        """
        super().reset()
        self.sigma = self.sigma_start
        if self.noise_type == 'ou':
            self.ou_state = None

class ParameterSpaceNoise(ExplorationStrategy):
    """
    กลยุทธ์การสำรวจโดยใช้สัญญาณรบกวนในพารามิเตอร์ของโมเดล (Parameter Space Noise)
    
    แทนที่จะเพิ่มสัญญาณรบกวนในการกระทำ (action) กลยุทธ์นี้เพิ่มสัญญาณรบกวนในพารามิเตอร์ของโมเดล
    ทำให้นโยบาย (policy) มีความสอดคล้องกับตัวเอง (consistent) มากขึ้น
    """
    
    def __init__(self, initial_stddev: float = 0.1, target_divergence: float = 0.2, 
                 adapt_factor: float = 1.01, min_stddev: float = 0.001, max_stddev: float = 1.0,
                 model: Optional[torch.nn.Module] = None):
        """
        กำหนดค่าเริ่มต้นสำหรับกลยุทธ์ ParameterSpaceNoise
        
        Parameters:
        initial_stddev (float): ค่าเบี่ยงเบนมาตรฐานเริ่มต้นของสัญญาณรบกวน
        target_divergence (float): ค่าความต่างที่ต้องการระหว่างนโยบายหลักและนโยบายที่มีสัญญาณรบกวน
        adapt_factor (float): ตัวคูณสำหรับการปรับค่าเบี่ยงเบนมาตรฐาน
        min_stddev (float): ค่าเบี่ยงเบนมาตรฐานต่ำสุด
        max_stddev (float): ค่าเบี่ยงเบนมาตรฐานสูงสุด
        model (torch.nn.Module, optional): โมเดลหลัก
        """
        super().__init__()
        self.stddev = initial_stddev
        self.target_divergence = target_divergence
        self.adapt_factor = adapt_factor
        self.min_stddev = min_stddev
        self.max_stddev = max_stddev
        
        self.model = model  # โมเดลหลัก
        self.perturbed_model = None  # โมเดลที่มีสัญญาณรบกวน
        
        logger.info(f"สร้างกลยุทธ์ ParameterSpaceNoise (stddev={initial_stddev}, target={target_divergence})")
    
    def set_model(self, model: torch.nn.Module):
        """
        กำหนดโมเดลหลัก
        
        Parameters:
        model (torch.nn.Module): โมเดลหลัก
        """
        self.model = model
        self.perturbed_model = None  # รีเซ็ตโมเดลที่มีสัญญาณรบกวน
    
    def perturb_model(self):
        """
        สร้างโมเดลที่มีสัญญาณรบกวนในพารามิเตอร์
        """
        if self.model is None:
            logger.error("ไม่ได้กำหนดโมเดลหลัก")
            return
        
        # คัดลอกโมเดลหลัก
        self.perturbed_model = type(self.model)()  # สร้างอินสแตนซ์ใหม่ของโมเดลหลัก
        self.perturbed_model.load_state_dict(self.model.state_dict())  # คัดลอกพารามิเตอร์
        
        # เพิ่มสัญญาณรบกวนในพารามิเตอร์
        with torch.no_grad():
            for param, perturbed_param in zip(self.model.parameters(), self.perturbed_model.parameters()):
                # สร้างสัญญาณรบกวนแบบ Gaussian
                noise = torch.randn_like(param) * self.stddev
                perturbed_param.add_(noise)
    
    def select_action(self, state: Union[np.ndarray, torch.Tensor], evaluation: bool = False, 
                      q_values: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[np.ndarray, int]:
        """
        เลือกการกระทำจากโมเดลที่มีสัญญาณรบกวน
        
        Parameters:
        state (np.ndarray or torch.Tensor): สถานะปัจจุบัน
        evaluation (bool): โหมดประเมิน (ใช้โมเดลหลัก)
        q_values (np.ndarray or torch.Tensor, optional): ไม่ได้ใช้ (สำหรับ interface เท่านั้น)
        
        Returns:
        np.ndarray or int: การกระทำที่เลือก
        """
        if self.model is None:
            logger.error("ไม่ได้กำหนดโมเดลหลัก")
            return 0 if isinstance(q_values, (np.ndarray, torch.Tensor)) else np.zeros(1)
        
        # ในโหมดประเมิน ใช้โมเดลหลัก
        if evaluation:
            # แปลงเป็น tensor ถ้าเป็น numpy array
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(next(self.model.parameters()).device)
            
            with torch.no_grad():
                return self.model(state).cpu().numpy()
        
        # สร้างโมเดลที่มีสัญญาณรบกวนถ้ายังไม่มี
        if self.perturbed_model is None:
            self.perturb_model()
        
        # แปลงเป็น tensor ถ้าเป็น numpy array
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.model.parameters()).device)
        
        # ใช้โมเดลที่มีสัญญาณรบกวน
        with torch.no_grad():
            return self.perturbed_model(state).cpu().numpy()
    
    def adapt(self, states: Union[np.ndarray, torch.Tensor]):
        """
        ปรับค่าเบี่ยงเบนมาตรฐานของสัญญาณรบกวนตามความต่างระหว่างนโยบาย
        
        Parameters:
        states (np.ndarray or torch.Tensor): ชุดของสถานะสำหรับการวัดความต่างระหว่างนโยบาย
        """
        if self.model is None or self.perturbed_model is None:
            logger.error("ไม่ได้กำหนดโมเดลหลักหรือยังไม่ได้สร้างโมเดลที่มีสัญญาณรบกวน")
            return
        
        # แปลงเป็น tensor ถ้าเป็น numpy array
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states).to(next(self.model.parameters()).device)
        
        # หาการกระทำจากทั้งสองโมเดล
        with torch.no_grad():
            actions = self.model(states)
            perturbed_actions = self.perturbed_model(states)
        
        # วัดความต่างระหว่างการกระทำ (ใช้ L2 distance)
        diff = torch.sqrt(torch.mean(torch.pow(actions - perturbed_actions, 2)))
        diff = diff.item()
        
        # ปรับค่าเบี่ยงเบนมาตรฐาน
        if diff < self.target_divergence:
            # เพิ่มสัญญาณรบกวนถ้าความต่างน้อยเกินไป
            self.stddev *= self.adapt_factor
        else:
            # ลดสัญญาณรบกวนถ้าความต่างมากเกินไป
            self.stddev /= self.adapt_factor
        
        # จำกัดค่าเบี่ยงเบนมาตรฐาน
        self.stddev = np.clip(self.stddev, self.min_stddev, self.max_stddev)
        
        # สร้างโมเดลที่มีสัญญาณรบกวนใหม่
        self.perturb_model()
    
    def update(self, step: Optional[int] = None):
        """
        อัพเดตกลยุทธ์ ParameterSpaceNoise
        
        Parameters:
        step (int, optional): ขั้นตอนปัจจุบัน
        """
        super().update(step)
        
        # ไม่มีการอัพเดตเพิ่มเติม (การปรับค่าเบี่ยงเบนมาตรฐานทำใน adapt())
    
    def get_stddev(self) -> float:
        """
        ดึงค่าเบี่ยงเบนมาตรฐานปัจจุบัน
        
        Returns:
        float: ค่าเบี่ยงเบนมาตรฐานปัจจุบัน
        """
        return self.stddev
    
    def reset(self):
        """
        รีเซ็ตกลยุทธ์ ParameterSpaceNoise
        """
        super().reset()
        self.perturbed_model = None  # รีเซ็ตโมเดลที่มีสัญญาณรบกวน

def get_exploration_strategy(strategy_type: str, **kwargs) -> ExplorationStrategy:
    """
    สร้างกลยุทธ์การสำรวจตามประเภทที่ระบุ
    
    Parameters:
    strategy_type (str): ประเภทของกลยุทธ์การสำรวจ ('epsilon_greedy', 'boltzmann', 'ucb', 'noise', 'parameter_noise')
    **kwargs: พารามิเตอร์เพิ่มเติมสำหรับกลยุทธ์การสำรวจ
    
    Returns:
    ExplorationStrategy: กลยุทธ์การสำรวจที่สร้างแล้ว
    """
    strategy_type = strategy_type.lower()
    
    if strategy_type == 'epsilon_greedy':
        return EpsilonGreedyExploration(**kwargs)
    elif strategy_type == 'boltzmann':
        return BoltzmannExploration(**kwargs)
    elif strategy_type == 'ucb':
        return UCBExploration(**kwargs)
    elif strategy_type == 'noise':
        return NoiseBasedExploration(**kwargs)
    elif strategy_type == 'parameter_noise':
        return ParameterSpaceNoise(**kwargs)
    else:
        logger.error(f"ไม่รู้จักประเภทของกลยุทธ์การสำรวจ: {strategy_type}, ใช้ EpsilonGreedyExploration แทน")
        return EpsilonGreedyExploration(**kwargs)