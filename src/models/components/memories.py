import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import deque, namedtuple
import torch

# กำหนด namedtuple สำหรับบันทึกประสบการณ์
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Buffer สำหรับเก็บประสบการณ์และสุ่มตัวอย่าง batch สำหรับ Reinforcement Learning
    
    ใช้ในอัลกอริทึมแบบ off-policy เช่น DQN, DDPG, SAC
    """
    
    def __init__(self, capacity: int):
        """
        กำหนดค่าเริ่มต้นสำหรับ ReplayBuffer
        
        Parameters:
        capacity (int): ความจุสูงสุดของ buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        action: Union[int, float, np.ndarray, torch.Tensor], 
        reward: float, 
        next_state: Union[np.ndarray, torch.Tensor], 
        done: bool
    ):
        """
        เพิ่มประสบการณ์ใหม่ลงใน buffer
        
        Parameters:
        state: สถานะปัจจุบัน
        action: การกระทำ
        reward: รางวัล
        next_state: สถานะถัดไป
        done: สถานะจบ (True หรือ False)
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # สร้าง Experience และเพิ่มลงใน buffer
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        สุ่มตัวอย่าง batch จาก buffer
        
        Parameters:
        batch_size (int): ขนาดของ batch
        
        Returns:
        List[Experience]: ตัวอย่าง batch
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def sample_tensors(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, ...]:
        """
        สุ่มตัวอย่าง batch จาก buffer และแปลงเป็น tensors
        
        Parameters:
        batch_size (int): ขนาดของ batch
        device (torch.device): อุปกรณ์ที่จะใช้เก็บ tensors
        
        Returns:
        Tuple[torch.Tensor, ...]: (states, actions, rewards, next_states, dones)
        """
        experiences = self.sample(batch_size)
        
        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float32, device=device)
        
        # แปลง action เป็น tensor ตามประเภท
        if isinstance(experiences[0].action, (int, float)):
            actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=device)
        else:
            actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.float32, device=device)
        
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float32, device=device)
        dones = torch.tensor([float(e.done) for e in experiences], dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """
        ดึงขนาดปัจจุบันของ buffer
        
        Returns:
        int: ขนาดปัจจุบันของ buffer
        """
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Experience:
        """
        ดึงประสบการณ์ที่ตำแหน่ง idx
        
        Parameters:
        idx (int): ตำแหน่งที่ต้องการดึง
        
        Returns:
        Experience: ประสบการณ์ที่ตำแหน่ง idx
        """
        return self.buffer[idx]
    
    def clear(self):
        """
        ล้าง buffer
        """
        self.buffer.clear()

class PrioritizedReplayBuffer:
    """
    Buffer สำหรับเก็บประสบการณ์และสุ่มตัวอย่าง batch ตามลำดับความสำคัญ
    
    ใช้ในอัลกอริทึมแบบ off-policy เช่น Prioritized Experience Replay (PER) ร่วมกับ DQN, DDPG, SAC
    """
    
    def __init__(
        self, 
        capacity: int, 
        alpha: float = 0.6, 
        beta: float = 0.4, 
        beta_increment: float = 1e-3, 
        epsilon: float = 1e-5
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับ PrioritizedReplayBuffer
        
        Parameters:
        capacity (int): ความจุสูงสุดของ buffer
        alpha (float): พารามิเตอร์ alpha ที่ควบคุมระดับของ prioritization (0 = ไม่มี prioritization, 1 = prioritization เต็มที่)
        beta (float): พารามิเตอร์ beta ที่ใช้ในการแก้ไข importance sampling bias (0 = ไม่แก้ไข, 1 = แก้ไขเต็มที่)
        beta_increment (float): ค่าที่ใช้เพิ่ม beta ในแต่ละครั้งที่เรียก sample
        epsilon (float): ค่า epsilon เล็กๆ ที่เพิ่มเข้าไปในทุก priority เพื่อไม่ให้มีค่าเป็น 0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # ใช้ numpy array เพื่อประสิทธิภาพ
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        action: Union[int, float, np.ndarray, torch.Tensor], 
        reward: float, 
        next_state: Union[np.ndarray, torch.Tensor], 
        done: bool
    ):
        """
        เพิ่มประสบการณ์ใหม่ลงใน buffer
        
        Parameters:
        state: สถานะปัจจุบัน
        action: การกระทำ
        reward: รางวัล
        next_state: สถานะถัดไป
        done: สถานะจบ (True หรือ False)
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # สร้าง Experience
        experience = Experience(state, action, reward, next_state, done)
        
        # กำหนด priority สูงสุดเริ่มต้น
        priority = self.priorities.max() if self.size > 0 else 1.0
        
        # เพิ่มลงใน buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # เพิ่ม priority
        self.priorities[self.position] = priority
        
        # อัพเดตตำแหน่งและขนาด
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        สุ่มตัวอย่าง batch จาก buffer ตามลำดับความสำคัญ
        
        Parameters:
        batch_size (int): ขนาดของ batch
        
        Returns:
        Tuple[List[Experience], np.ndarray, np.ndarray]: (experiences, indices, weights)
        """
        # ตรวจสอบว่ามีข้อมูลพอสำหรับการสุ่มหรือไม่
        if self.size < batch_size:
            batch_size = self.size
        
        # คำนวณความน่าจะเป็นในการสุ่ม
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # สุ่มดัชนี
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # ดึงประสบการณ์
        experiences = [self.buffer[idx] for idx in indices]
        
        # คำนวณ importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize
        
        return experiences, indices, weights
    
    def sample_tensors(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, ...]:
        """
        สุ่มตัวอย่าง batch จาก buffer ตามลำดับความสำคัญและแปลงเป็น tensors
        
        Parameters:
        batch_size (int): ขนาดของ batch
        device (torch.device): อุปกรณ์ที่จะใช้เก็บ tensors
        
        Returns:
        Tuple[torch.Tensor, ...]: (states, actions, rewards, next_states, dones, indices, weights)
        """
        experiences, indices, weights = self.sample(batch_size)
        
        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float32, device=device)
        
        # แปลง action เป็น tensor ตามประเภท
        if isinstance(experiences[0].action, (int, float)):
            actions = torch.tensor([e.action for e in experiences], dtype=torch.long, device=device)
        else:
            actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.float32, device=device)
        
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float32, device=device)
        dones = torch.tensor([float(e.done) for e in experiences], dtype=torch.float32, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        อัพเดต priorities ของประสบการณ์
        
        Parameters:
        indices (np.ndarray): ดัชนีของประสบการณ์ที่ต้องการอัพเดต
        priorities (np.ndarray): priorities ใหม่
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self) -> int:
        """
        ดึงขนาดปัจจุบันของ buffer
        
        Returns:
        int: ขนาดปัจจุบันของ buffer
        """
        return self.size

class EpisodeBuffer:
    """
    Buffer สำหรับเก็บประสบการณ์ทั้ง episode
    
    ใช้ในอัลกอริทึมแบบ on-policy เช่น REINFORCE, A2C, PPO
    """
    
    def __init__(self):
        """
        กำหนดค่าเริ่มต้นสำหรับ EpisodeBuffer
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def push(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        action: Union[int, float, np.ndarray, torch.Tensor], 
        reward: float, 
        next_state: Union[np.ndarray, torch.Tensor], 
        done: bool, 
        log_prob: Optional[float] = None, 
        value: Optional[float] = None
    ):
        """
        เพิ่มประสบการณ์ใหม่ลงใน buffer
        
        Parameters:
        state: สถานะปัจจุบัน
        action: การกระทำ
        reward: รางวัล
        next_state: สถานะถัดไป
        done: สถานะจบ (True หรือ False)
        log_prob (float, optional): log probability ของการกระทำ (สำหรับ policy gradient)
        value (float, optional): ค่า value ของสถานะ (สำหรับ actor-critic)
        """
        # แปลงเป็น numpy array ถ้าเป็น tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.cpu().numpy().item()
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy().item()
        
        # เพิ่มลงใน buffer
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if log_prob is not None:
            self.log_probs.append(log_prob)
        
        if value is not None:
            self.values.append(value)
    
    def get_episode(self) -> Dict[str, List]:
        """
        ดึงข้อมูลทั้ง episode
        
        Returns:
        Dict[str, List]: ข้อมูลทั้ง episode
        """
        episode = {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'dones': self.dones
        }
        
        if self.log_probs:
            episode['log_probs'] = self.log_probs
        
        if self.values:
            episode['values'] = self.values
        
        return episode
    
    def get_episode_tensors(self, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """
        ดึงข้อมูลทั้ง episode ในรูปแบบ tensors
        
        Parameters:
        device (torch.device): อุปกรณ์ที่จะใช้เก็บ tensors
        
        Returns:
        Dict[str, torch.Tensor]: ข้อมูลทั้ง episode ในรูปแบบ tensors
        """
        episode = {}
        
        # แปลงเป็น tensors
        episode['states'] = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        
        # แปลง action เป็น tensor ตามประเภท
        if isinstance(self.actions[0], (int, float)):
            episode['actions'] = torch.tensor(self.actions, dtype=torch.long, device=device)
        else:
            episode['actions'] = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        
        episode['rewards'] = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        episode['next_states'] = torch.tensor(np.array(self.next_states), dtype=torch.float32, device=device)
        episode['dones'] = torch.tensor(self.dones, dtype=torch.float32, device=device)
        
        if self.log_probs:
            episode['log_probs'] = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        
        if self.values:
            episode['values'] = torch.tensor(self.values, dtype=torch.float32, device=device)
        
        return episode
    
    def calculate_returns(self, gamma: float = 0.99) -> List[float]:
        """
        คำนวณผลตอบแทนสะสม (discounted returns)
        
        Parameters:
        gamma (float): discount factor
        
        Returns:
        List[float]: ผลตอบแทนสะสม
        """
        returns = []
        discounted_return = 0
        
        # คำนวณย้อนกลับจากท้ายสุด
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_return = 0
            
            discounted_return = reward + gamma * discounted_return
            returns.insert(0, discounted_return)
        
        return returns
    
    def calculate_advantages(self, gamma: float = 0.99, lambda_: float = 0.95) -> List[float]:
        """
        คำนวณ advantages ด้วย Generalized Advantage Estimation (GAE)
        
        Parameters:
        gamma (float): discount factor
        lambda_ (float): GAE parameter
        
        Returns:
        List[float]: advantages
        """
        if not self.values:
            raise ValueError("ต้องมีค่า values ในการคำนวณ advantages")
        
        advantages = []
        last_advantage = 0
        last_value = 0
        
        # คำนวณย้อนกลับจากท้ายสุด
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                last_advantage = 0
                last_value = 0
            
            delta = self.rewards[i] + gamma * last_value * (1 - self.dones[i]) - self.values[i]
            last_advantage = delta + gamma * lambda_ * last_advantage * (1 - self.dones[i])
            last_value = self.values[i]
            
            advantages.insert(0, last_advantage)
        
        return advantages
    
    def __len__(self) -> int:
        """
        ดึงขนาดของ buffer
        
        Returns:
        int: ขนาดของ buffer
        """
        return len(self.states)
    
    def clear(self):
        """
        ล้าง buffer
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

class ExperienceReplay:
    """
    ระบบจัดการประสบการณ์สำหรับ Reinforcement Learning
    
    รวมความสามารถของ ReplayBuffer, PrioritizedReplayBuffer, และ EpisodeBuffer
    """
    
    def __init__(
        self, 
        capacity: int, 
        prioritized: bool = False, 
        alpha: float = 0.6, 
        beta: float = 0.4, 
        beta_increment: float = 1e-3, 
        epsilon: float = 1e-5
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับ ExperienceReplay
        
        Parameters:
        capacity (int): ความจุสูงสุดของ buffer
        prioritized (bool): ใช้ Prioritized Experience Replay หรือไม่
        alpha (float): พารามิเตอร์ alpha สำหรับ PER
        beta (float): พารามิเตอร์ beta สำหรับ PER
        beta_increment (float): ค่าที่ใช้เพิ่ม beta ในแต่ละครั้งที่เรียก sample สำหรับ PER
        epsilon (float): ค่า epsilon เล็กๆ ที่เพิ่มเข้าไปในทุก priority สำหรับ PER
        """
        self.prioritized = prioritized
        
        # สร้าง buffer ตามประเภท
        if prioritized:
            self.buffer = PrioritizedReplayBuffer(
                capacity=capacity,
                alpha=alpha,
                beta=beta,
                beta_increment=beta_increment,
                epsilon=epsilon
            )
        else:
            self.buffer = ReplayBuffer(capacity=capacity)
        
        # สร้าง episode buffer สำหรับแต่ละ episode
        self.episode_buffer = EpisodeBuffer()
    
    def push(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        action: Union[int, float, np.ndarray, torch.Tensor], 
        reward: float, 
        next_state: Union[np.ndarray, torch.Tensor], 
        done: bool, 
        log_prob: Optional[float] = None, 
        value: Optional[float] = None
    ):
        """
        เพิ่มประสบการณ์ใหม่ลงใน buffer
        
        Parameters:
        state: สถานะปัจจุบัน
        action: การกระทำ
        reward: รางวัล
        next_state: สถานะถัดไป
        done: สถานะจบ (True หรือ False)
        log_prob (float, optional): log probability ของการกระทำ (สำหรับ policy gradient)
        value (float, optional): ค่า value ของสถานะ (สำหรับ actor-critic)
        """
        # เพิ่มลงใน replay buffer
        self.buffer.push(state, action, reward, next_state, done)
        
        # เพิ่มลงใน episode buffer
        self.episode_buffer.push(state, action, reward, next_state, done, log_prob, value)
    
    def sample(self, batch_size: int) -> Union[List[Experience], Tuple[List[Experience], np.ndarray, np.ndarray]]:
        """
        สุ่มตัวอย่าง batch จาก buffer
        
        Parameters:
        batch_size (int): ขนาดของ batch
        
        Returns:
        Union[List[Experience], Tuple[List[Experience], np.ndarray, np.ndarray]]: ตัวอย่าง batch
        """
        return self.buffer.sample(batch_size)
    
    def sample_tensors(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, ...]:
        """
        สุ่มตัวอย่าง batch จาก buffer และแปลงเป็น tensors
        
        Parameters:
        batch_size (int): ขนาดของ batch
        device (torch.device): อุปกรณ์ที่จะใช้เก็บ tensors
        
        Returns:
        Tuple[torch.Tensor, ...]: ตัวอย่าง batch ในรูปแบบ tensors
        """
        return self.buffer.sample_tensors(batch_size, device)
    
    def get_episode(self) -> Dict[str, List]:
        """
        ดึงข้อมูลทั้ง episode
        
        Returns:
        Dict[str, List]: ข้อมูลทั้ง episode
        """
        return self.episode_buffer.get_episode()
    
    def get_episode_tensors(self, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """
        ดึงข้อมูลทั้ง episode ในรูปแบบ tensors
        
        Parameters:
        device (torch.device): อุปกรณ์ที่จะใช้เก็บ tensors
        
        Returns:
        Dict[str, torch.Tensor]: ข้อมูลทั้ง episode ในรูปแบบ tensors
        """
        return self.episode_buffer.get_episode_tensors(device)
    
    def calculate_returns(self, gamma: float = 0.99) -> List[float]:
        """
        คำนวณผลตอบแทนสะสม (discounted returns)
        
        Parameters:
        gamma (float): discount factor
        
        Returns:
        List[float]: ผลตอบแทนสะสม
        """
        return self.episode_buffer.calculate_returns(gamma)
    
    def calculate_advantages(self, gamma: float = 0.99, lambda_: float = 0.95) -> List[float]:
        """
        คำนวณ advantages ด้วย Generalized Advantage Estimation (GAE)
        
        Parameters:
        gamma (float): discount factor
        lambda_ (float): GAE parameter
        
        Returns:
        List[float]: advantages
        """
        return self.episode_buffer.calculate_advantages(gamma, lambda_)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        อัพเดต priorities ของประสบการณ์ (สำหรับ PER)
        
        Parameters:
        indices (np.ndarray): ดัชนีของประสบการณ์ที่ต้องการอัพเดต
        priorities (np.ndarray): priorities ใหม่
        """
        if self.prioritized:
            self.buffer.update_priorities(indices, priorities)
    
    def end_episode(self):
        """
        จบ episode ปัจจุบันและเริ่ม episode ใหม่
        """
        self.episode_buffer.clear()
    
    def __len__(self) -> int:
        """
        ดึงขนาดของ buffer
        
        Returns:
        int: ขนาดของ buffer
        """
        return len(self.buffer)
    
    def clear(self):
        """
        ล้าง buffer ทั้งหมด
        """
        self.buffer.clear()
        self.episode_buffer.clear()