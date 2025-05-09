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

from src.models.dqn.dqn import DQN, ReplayBuffer
from src.utils.config import get_config

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class DuelingDQNNetwork(nn.Module):
    """
    เครือข่ายประสาทเทียมสำหรับ Dueling DQN
    
    Dueling DQN แยกการประมาณค่า Q ออกเป็นฟังก์ชัน Value และฟังก์ชัน Advantage
    ทำให้การประมาณค่า Q มีประสิทธิภาพมากขึ้น
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับเครือข่าย Dueling DQN
        
        Parameters:
        input_dim (int): ขนาดของอินพุต
        output_dim (int): ขนาดของเอาต์พุต (จำนวนการกระทำ)
        hidden_dims (List[int]): ขนาดของชั้นซ่อน
        activation (str): ฟังก์ชันกระตุ้น ('relu', 'tanh', 'leaky_relu')
        use_batch_norm (bool): ใช้ Batch Normalization หรือไม่
        dropout_rate (float): อัตราการ dropout
        """
        super(DuelingDQNNetwork, self).__init__()
        
        # กำหนดฟังก์ชันกระตุ้น
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.ReLU()
        
        # สร้างโครงสร้างของเครือข่ายร่วม (feature extractor)
        feature_layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims[:-1]):  # ใช้ทุกชั้นยกเว้นชั้นสุดท้าย
            feature_layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                feature_layers.append(nn.BatchNorm1d(dim))
            
            feature_layers.append(self.activation)
            
            if dropout_rate > 0:
                feature_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # ขนาดของ feature extractor output
        feature_output_dim = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
        
        # สร้างเครือข่ายสำหรับ Value function (V)
        value_layers = []
        value_layers.append(nn.Linear(feature_output_dim, hidden_dims[-1]))
        
        if use_batch_norm:
            value_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        
        value_layers.append(self.activation)
        
        if dropout_rate > 0:
            value_layers.append(nn.Dropout(dropout_rate))
        
        value_layers.append(nn.Linear(hidden_dims[-1], 1))  # Value function มีเอาต์พุตเดียว
        
        self.value_stream = nn.Sequential(*value_layers)
        
        # สร้างเครือข่ายสำหรับ Advantage function (A)
        advantage_layers = []
        advantage_layers.append(nn.Linear(feature_output_dim, hidden_dims[-1]))
        
        if use_batch_norm:
            advantage_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        
        advantage_layers.append(self.activation)
        
        if dropout_rate > 0:
            advantage_layers.append(nn.Dropout(dropout_rate))
        
        advantage_layers.append(nn.Linear(hidden_dims[-1], output_dim))  # Advantage function มีเอาต์พุตตามจำนวนการกระทำ
        
        self.advantage_stream = nn.Sequential(*advantage_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        การส่งผ่านไปข้างหน้า (forward pass)
        
        Parameters:
        x (torch.Tensor): อินพุต
        
        Returns:
        torch.Tensor: เอาต์พุต (Q-values)
        """
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        # การลบค่าเฉลี่ยของ Advantage ช่วยให้การประมาณค่า Q มีเสถียรภาพมากขึ้น
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class DuelingDQN(DQN):
    """
    อัลกอริทึม Dueling DQN
    
    Dueling DQN แยกการประมาณค่า Q ออกเป็นฟังก์ชัน Value และฟังก์ชัน Advantage
    ทำให้การประมาณค่า Q มีประสิทธิภาพมากขึ้น โดยเฉพาะในกรณีที่การกระทำหลายๆ อย่างมีผลลัพธ์ไม่แตกต่างกันมากนัก
    """
    
    def __init__(
        self,
        input_size: int = None,
        state_dim: int = None,
        action_dim: int = None,
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
        กำหนดค่าเริ่มต้นสำหรับ Dueling DQN
        
        Parameters:
        input_size (int, optional): ขนาดของ input (ใช้แทน state_dim ถ้าใช้กับ BaseModel)
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
        # โหลดการตั้งค่า
        self.config = config if config is not None else get_config()
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        model_config = self.config.extract_subconfig("model")
        cuda_config = self.config.extract_subconfig("cuda")
        
        # ให้เข้ากันกับ BaseModel interface
        if input_size is not None and state_dim is None:
            state_dim = input_size
        
        # กำหนดค่าพารามิเตอร์
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or model_config.get("hidden_layers", [128, 64])
        self.learning_rate = learning_rate or model_config.get("learning_rate", 0.001)
        self.gamma = gamma or model_config.get("discount_factor", 0.99)
        self.epsilon_start = epsilon_start or model_config.get("exploration_initial", 1.0)
        self.epsilon_end = epsilon_end or model_config.get("exploration_final", 0.01)
        self.epsilon_decay = epsilon_decay or model_config.get("exploration_decay", 0.995)
        self.target_update_freq = target_update_freq or model_config.get("target_update_frequency", 10)
        self.batch_size = batch_size or model_config.get("batch_size", 64)
        self.buffer_size = buffer_size or model_config.get("replay_buffer_size", 10000)
        self.use_batch_norm = use_batch_norm if use_batch_norm is not None else model_config.get("use_batch_norm", False)
        self.dropout_rate = dropout_rate or model_config.get("dropout_rate", 0.0)
        self.activation = activation or model_config.get("activation_function", "relu")
        self.weight_decay = weight_decay or model_config.get("weight_decay", 0.0)
        self.clip_grad_norm = clip_grad_norm or model_config.get("clip_grad_norm", None)
        
        # กำหนดอุปกรณ์
        if device is None:
            self.device = torch.device("cuda" if cuda_config.get("use_cuda", True) and torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # สร้างเครือข่าย
        self.policy_net = DuelingDQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self.target_net = DuelingDQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # คัดลอกพารามิเตอร์จาก policy_net ไปยัง target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # ตั้งค่า target_net เป็นโหมดประเมิน
        
        # กำหนด optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # สร้าง replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # ตัวแปรเพิ่มเติม
        self.epsilon = self.epsilon_start
        self.step_count = 0
        self.update_count = 0
        self.train_count = 0
        
        logger.info(f"สร้าง Dueling DQN (state_dim={state_dim}, action_dim={action_dim}, device={self.device})")
    
    def update(self) -> float:
        """
        อัพเดต policy network จาก replay buffer โดยใช้ Double DQN ร่วมกับ Dueling network
        
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
        
        # คำนวณ Q-values เป้าหมายด้วย Double DQN + Dueling
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # บันทึก state_dict ของโมเดล
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'train_count': self.train_count
        }, path)
        
        # บันทึกการตั้งค่า
        config_path = os.path.splitext(path)[0] + '_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'target_update_freq': self.target_update_freq,
                'batch_size': self.batch_size,
                'buffer_size': self.buffer_size,
                'use_batch_norm': self.use_batch_norm,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation,
                'weight_decay': self.weight_decay,
                'clip_grad_norm': self.clip_grad_norm,
                'model_type': 'dueling_dqn'
            }, f, indent=2)
        
        logger.info(f"บันทึกโมเดลที่: {path}")
    
    def load(self, path: str) -> None:
        """
        โหลดโมเดลจากไฟล์
        
        Parameters:
        path (str): พาธที่จะโหลด
        """
        # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
        if not os.path.exists(path):
            raise FileNotFoundError(f"ไม่พบไฟล์: {path}")
        
        # โหลด state_dict ของโมเดล
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
        self.train_count = checkpoint['train_count']
        
        logger.info(f"โหลดโมเดลจาก: {path}")
    
    # เพิ่มเมธอดสำหรับ BaseModel interface
    def _create_model(self) -> torch.nn.Module:
        """
        สร้างโมเดล (สำหรับ BaseModel interface)
        
        Returns:
        torch.nn.Module: โมเดลที่สร้างแล้ว
        """
        return self.policy_net
    
    def train(self, train_loader, val_loader=None, epochs=None, log_dir=None) -> Dict[str, Any]:
        """
        เทรนโมเดล (สำหรับ BaseModel interface)
        
        Parameters:
        train_loader: DataLoader สำหรับข้อมูลเทรน
        val_loader: DataLoader สำหรับข้อมูล validation
        epochs (int, optional): จำนวนรอบการเทรน
        log_dir (str, optional): ไดเรกทอรีสำหรับบันทึก log
        
        Returns:
        Dict[str, Any]: ประวัติการเทรน
        """
        # ตั้งค่า TensorBoard logger ถ้าจำเป็น
        tensorboard_logger = self._setup_tensorboard(log_dir) if log_dir else None
        
        # กำหนดจำนวนรอบการเทรน
        if epochs is None:
            epochs = 100  # ค่าเริ่มต้น
        
        # ประวัติการเทรน
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            # เทรนโมเดล
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (states,) in enumerate(train_loader):
                # แปลงข้อมูลเป็น numpy array
                states_np = states.cpu().numpy()
                
                # สร้างข้อมูลการเทรนเสมือน
                for state in states_np:
                    # เลือกการกระทำตามนโยบาย epsilon-greedy
                    action = self.select_action(state)
                    
                    # สร้างสถานะถัดไป, รางวัล, และสถานะจบเสมือน
                    next_state = state  # สมมติว่าไม่มีการเปลี่ยนแปลง
                    reward = 0.0  # สมมติว่าไม่มีรางวัล
                    done = False  # สมมติว่าไม่จบ
                    
                    # เก็บประสบการณ์ใน replay buffer
                    self.store_experience(state, action, reward, next_state, done)
                
                # อัพเดตโมเดล
                loss = self.update()
                
                epoch_loss += loss
                num_batches += 1
            
            # คำนวณค่าเฉลี่ยของ loss
            avg_loss = epoch_loss / max(num_batches, 1)
            history['train_loss'].append(avg_loss)
            
            # บันทึกลง TensorBoard
            if tensorboard_logger:
                tensorboard_logger.log_scalar('train_loss', avg_loss, epoch)
            
            # ประเมินกับชุดข้อมูล validation
            if val_loader:
                val_loss = self.evaluate(val_loader)['loss']
                history['val_loss'].append(val_loss)
                
                # บันทึกลง TensorBoard
                if tensorboard_logger:
                    tensorboard_logger.log_scalar('val_loss', val_loss, epoch)
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        
        # ปิด TensorBoard logger
        if tensorboard_logger:
            tensorboard_logger.close()
        
        return history
    
    def predict(self, inputs) -> int:
        """
        ทำนายการกระทำ (สำหรับ BaseModel interface)
        
        Parameters:
        inputs: ข้อมูลนำเข้า
        
        Returns:
        int: การกระทำที่ทำนาย
        """
        # เตรียมข้อมูลนำเข้า
        inputs = self._prepare_input(inputs)
        
        # ทำนายการกระทำ
        with torch.no_grad():
            q_values = self.policy_net(inputs)
            action = q_values.argmax().item()
        
        return action
    
    def evaluate(self, data_loader, metrics_list=None) -> Dict[str, float]:
        """
        ประเมินโมเดล (สำหรับ BaseModel interface)
        
        Parameters:
        data_loader: DataLoader สำหรับข้อมูลทดสอบ
        metrics_list (List[str], optional): รายการ metrics ที่ต้องการประเมิน
        
        Returns:
        Dict[str, float]: ผลการประเมิน
        """
        # ตั้งค่าโมเดลเป็นโหมดประเมิน
        self.policy_net.eval()
        
        # เก็บข้อมูลสำหรับคำนวณ metrics
        total_loss = 0.0
        num_samples = 0
        all_actions = []
        
        with torch.no_grad():
            for batch_idx, (states,) in enumerate(data_loader):
                # แปลงข้อมูลเป็น numpy array
                states_np = states.cpu().numpy()
                
                # ทำนายการกระทำ
                batch_actions = []
                for state in states_np:
                    action = self.select_action(state, evaluation=True)
                    batch_actions.append(action)
                
                # สะสมการกระทำ
                all_actions.extend(batch_actions)
                
                # สมมติว่ามีการคำนวณ loss
                # ในกรณีจริงต้องมีข้อมูลการกระทำที่ถูกต้องและรางวัล
                loss = 0.0
                
                total_loss += loss
                num_samples += len(states)
        
        # คำนวณค่าเฉลี่ยของ loss
        avg_loss = total_loss / max(num_samples, 1)
        
        # ตั้งค่าโมเดลกลับเป็นโหมดเทรน
        self.policy_net.train()
        
        # คำนวณ metrics
        metrics = {
            'loss': avg_loss,
            'action_distribution': {action: all_actions.count(action) / len(all_actions) for action in set(all_actions)} if all_actions else {}
        }
        
        return metrics
    
    def _setup_tensorboard(self, log_dir: str):
        """
        ตั้งค่า TensorBoard logger
        
        Parameters:
        log_dir (str): ไดเรกทอรีสำหรับบันทึก log
        
        Returns:
        TensorboardLogger or None: TensorBoard logger หรือ None หากไม่สามารถตั้งค่าได้
        """
        try:
            from src.utils.loggers import TensorboardLogger
            return TensorboardLogger(log_dir)
        except ImportError:
            logger.warning("ไม่พบโมดูล TensorboardLogger")
            return None