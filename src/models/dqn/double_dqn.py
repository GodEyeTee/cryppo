import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import json

from src.models.dqn.dqn import DQN, DQNNetwork, ReplayBuffer
from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class DoubleDQN(DQN):
    """
    อัลกอริทึม Double DQN
    
    Double DQN แก้ปัญหาการประมาณค่า Q ที่สูงเกินไปของ DQN ธรรมดา
    โดยใช้ policy network เพื่อเลือกการกระทำและ target network เพื่อประเมินค่า Q
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        batch_size: int = 64,
        buffer_size: int = 10000,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        activation: str = 'relu',
        weight_decay: float = 0.0,
        clip_grad_norm: Optional[float] = None,
        device: Optional[str] = None,
        config = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับ Double DQN
        
        Parameters:
        state_dim (int): ขนาดของสถานะ
        action_dim (int): จำนวนการกระทำที่เป็นไปได้
        hidden_dims (List[int]): ขนาดของชั้นซ่อนในเครือข่าย
        learning_rate (float): อัตราการเรียนรู้
        gamma (float): ค่าส่วนลด (discount factor)
        epsilon_start (float): ค่า epsilon เริ่มต้น (สำหรับ epsilon-greedy)
        epsilon_end (float): ค่า epsilon สุดท้าย
        epsilon_decay (float): อัตราการลดลงของ epsilon
        target_update_freq (int): ความถี่ในการอัพเดต target network
        batch_size (int): ขนาดของ batch
        buffer_size (int): ขนาดของ replay buffer
        use_batch_norm (bool): ใช้ Batch Normalization หรือไม่
        dropout_rate (float): อัตราการ dropout
        activation (str): ฟังก์ชันกระตุ้น ('relu', 'tanh', 'leaky_relu')
        weight_decay (float): ค่า L2 regularization
        clip_grad_norm (float, optional): ค่า gradient clipping
        device (str, optional): อุปกรณ์ที่ใช้ ('cpu' หรือ 'cuda')
        config (Config, optional): อ็อบเจ็กต์การตั้งค่า
        """
        # เรียกคอนสตรักเตอร์ของคลาสแม่
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            batch_size=batch_size,
            buffer_size=buffer_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            device=device,
            config=config
        )
        
        logger.info(f"สร้าง Double DQN")
    
    def update(self) -> float:
        """
        อัพเดต policy network จาก replay buffer โดยใช้ Double DQN
        
        Returns:
        float: ค่าความสูญเสีย (loss)
        """
        # ตรวจสอบว่ามีข้อมูลพอสำหรับการเทรนหรือไม่
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # สุ่มตัวอย่าง batch จาก replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # แปลงเป็น tensor
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # คำนวณ Q-values ปัจจุบัน
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # คำนวณ Q-values เป้าหมายด้วย Double DQN
        with torch.no_grad():
            # ใช้ policy network เพื่อเลือกการกระทำที่ดีที่สุดในสถานะถัดไป
            next_action_batch = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            
            # ใช้ target network เพื่อประเมินค่า Q ของการกระทำที่เลือก
            next_q_values = self.target_net(next_state_batch).gather(1, next_action_batch)
            
            # คำนวณค่า Q เป้าหมาย
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # คำนวณความสูญเสีย
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # อัพเดตเครือข่าย
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (ถ้ากำหนด)
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()
        
        # อัพเดตเครือข่ายเป้าหมายเป็นระยะ
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # อัพเดต epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.train_count += 1
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """
        บันทึกโมเดลไปยังไฟล์
        
        Parameters:
        path (str): พาธที่จะบันทึก
        """
        # สร้างโฟลเดอร์หากไม่มีอยู่
        os.makedirs(