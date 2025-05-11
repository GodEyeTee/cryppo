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
import traceback

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
        กำหนดค่าเริ่มต้นสำหรับ Double DQN
        
        Parameters:
        input_size (int, optional): ขนาดของ input (ใช้แทน state_dim ถ้าใช้กับ BaseModel)
        state_dim (int, optional): ขนาดของสถานะ
        action_dim (int, optional): จำนวนการกระทำที่เป็นไปได้
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
        # ให้เข้ากันกับ BaseModel interface
        if input_size is not None and state_dim is None:
            state_dim = input_size
        
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
        
        try:
            # สุ่มตัวอย่าง batch จาก replay buffer
            batch = self.replay_buffer.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            
            # ตรวจสอบและแสดงข้อมูลสำหรับการดีบัก
            print("State batch type:", type(state_batch))
            if len(state_batch) > 0:
                print("First state type:", type(state_batch[0]))
                print("First state shape:", np.array(state_batch[0]).shape)
            
            print("Action batch type:", type(action_batch))
            if len(action_batch) > 0:
                print("First action:", action_batch[0])
                print("Action batch values:", action_batch[:5])  # แสดง 5 ค่าแรก
            
            # ทำความสะอาดข้อมูล action ให้อยู่ในช่วงที่ถูกต้อง
            cleaned_action_batch = []
            for action in action_batch:
                # ตรวจสอบว่า action เป็นตัวเลขและอยู่ในช่วงที่ถูกต้อง
                try:
                    action_val = int(action)
                    if action_val < 0 or action_val >= self.action_dim:
                        # ถ้า action ไม่ถูกต้อง ให้ใช้ค่าที่ถูกต้องแทน (บีบให้อยู่ในช่วงที่ถูกต้อง)
                        action_val = action_val % self.action_dim
                    cleaned_action_batch.append(action_val)
                except (ValueError, TypeError):
                    # ถ้าไม่สามารถแปลงเป็นตัวเลขได้ ให้ใช้ค่าเริ่มต้น
                    cleaned_action_batch.append(0)
            
            # แปลงข้อมูลเป็น numpy array อย่างชัดเจน
            state_batch_np = np.array(state_batch, dtype=np.float32)
            action_batch_np = np.array(cleaned_action_batch, dtype=np.int64)
            reward_batch_np = np.array(reward_batch, dtype=np.float32)
            next_state_batch_np = np.array(next_state_batch, dtype=np.float32)
            done_batch_np = np.array(done_batch, dtype=np.float32)
            
            # แสดงรูปร่างของ arrays เพื่อตรวจสอบ
            print("State batch shape:", state_batch_np.shape)
            print("Action batch shape:", action_batch_np.shape)
            
            # ตรวจสอบขนาดของข้อมูล state และเพิ่มคอลัมน์เสริมถ้าจำเป็น
            # ถ้าคอลัมน์ที่โมเดลคาดหวังมากกว่าคอลัมน์ที่มีอยู่จริง
            # ใช้ self.state_dim แทน self.input_size
            if state_batch_np.shape[-1] < self.state_dim:
                # คำนวณจำนวนคอลัมน์ที่ต้องเพิ่ม
                extra_columns = self.state_dim - state_batch_np.shape[-1]
                # สร้างคอลัมน์เสริมที่มีค่าเป็น 0
                batch_size, seq_len, _ = state_batch_np.shape
                padding = np.zeros((batch_size, seq_len, extra_columns), dtype=np.float32)
                # เพิ่มคอลัมน์เสริมเข้าไปในข้อมูล
                state_batch_np = np.concatenate([state_batch_np, padding], axis=2)
                print(f"Added {extra_columns} padding columns, new shape:", state_batch_np.shape)
                
                # ทำแบบเดียวกันสำหรับ next_state_batch_np
                next_state_batch_np = np.concatenate([next_state_batch_np, padding], axis=2)
            
            # แปลงเป็น PyTorch tensors
            state_batch = torch.FloatTensor(state_batch_np).to(self.device)
            action_batch = torch.LongTensor(action_batch_np).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch_np).unsqueeze(1).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch_np).to(self.device)
            done_batch = torch.FloatTensor(done_batch_np).unsqueeze(1).to(self.device)
            
            # แสดงรูปร่างของ tensors เพื่อตรวจสอบ
            print("State tensor shape:", state_batch.shape)
            print("Action tensor shape:", action_batch.shape)
            
            # แน่ใจว่า action_batch มีมิติที่ถูกต้อง
            if action_batch.dim() == 1:
                action_batch = action_batch.unsqueeze(1)
                print("Reshaped action tensor:", action_batch.shape)
            
            # คำนวณ Q-values
            try:
                q_values = self.policy_net(state_batch)
                print("Q-values shape:", q_values.shape)
                
                # แปลงรูปร่างของ q_values ถ้าจำเป็น
                if len(q_values.shape) == 3:
                    # เลือกเฉพาะสถานะสุดท้ายของแต่ละลำดับ
                    q_values = q_values[:, -1, :]
                    print("Reshaped Q-values shape:", q_values.shape)
                    
                # ตรวจสอบช่วงของ action_batch
                max_action = int(torch.max(action_batch).item())
                print(f"Maximum action value: {max_action}, Q-values dimension: {q_values.size(1)}")
                
                # ตรวจสอบและป้องกันความผิดพลาดของ action_batch
                if max_action >= q_values.size(1):
                    print(f"Warning: Action value {max_action} exceeds Q-values dimension {q_values.size(1)}")
                    # แก้ไขค่า action ที่ไม่ถูกต้อง
                    action_batch = torch.clamp(action_batch, 0, q_values.size(1) - 1)
                    print(f"Action tensor clamped, new max: {torch.max(action_batch).item()}")
                
                # คำนวณ Q-values ปัจจุบัน
                current_q_values = q_values.gather(1, action_batch)
                
                # คำนวณ Q-values เป้าหมายด้วย Double DQN
                with torch.no_grad():
                    # ใช้ policy network เพื่อเลือกการกระทำที่ดีที่สุดในสถานะถัดไป
                    next_q_values = self.policy_net(next_state_batch)
                    
                    # ถ้า next_q_values มี 3 มิติ ให้ใช้เฉพาะสถานะสุดท้ายของแต่ละลำดับ
                    if len(next_q_values.shape) == 3:
                        next_q_values = next_q_values[:, -1, :]
                    
                    next_action_batch = next_q_values.argmax(dim=1, keepdim=True)
                    
                    # ใช้ target network เพื่อประเมินค่า Q ของการกระทำที่เลือก
                    target_q_values = self.target_net(next_state_batch)
                    
                    # ถ้า target_q_values มี 3 มิติ ให้ใช้เฉพาะสถานะสุดท้ายของแต่ละลำดับ
                    if len(target_q_values.shape) == 3:
                        target_q_values = target_q_values[:, -1, :]
                    
                    target_q_values = target_q_values.gather(1, next_action_batch)
                    
                    # คำนวณค่า Q เป้าหมาย
                    target_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values
                
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
            
            except Exception as e:
                print(f"Error during Q-values calculation or loss computation: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
        
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการอัพเดต: {e}")
            import traceback
            traceback.print_exc()  # พิมพ์ stack trace เพื่อดีบัก
            return 0.0
    
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
                'model_type': 'double_dqn'
            }, f, indent=2)
        
        logger.info(f"บันทึกโมเดลที่: {path}")
    
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
        # ตั้งค่า TensorBoard logger อย่างปลอดภัย
        try:
            import os
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                tensorboard_logger = self._setup_tensorboard(log_dir)
            else:
                tensorboard_logger = None
        except Exception as e:
            print(f"ไม่สามารถตั้งค่า TensorboardLogger ได้: {e}")
            tensorboard_logger = None
        
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
                    next_state = state.copy()  # ทำสำเนาเพื่อหลีกเลี่ยงการอ้างอิงเดียวกัน
                    reward = 0.0  # สมมติว่าไม่มีรางวัล
                    done = False  # สมมติว่าไม่จบ
                    
                    # เก็บประสบการณ์ใน replay buffer
                    self.store_experience(state, action, reward, next_state, done)
                
                # อัพเดตโมเดล
                loss = self.update()
                
                epoch_loss += loss
                num_batches += 1
                
                # แสดงความคืบหน้า
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss:.6f}")
            
            # คำนวณค่าเฉลี่ยของ loss
            avg_loss = epoch_loss / max(num_batches, 1)
            history['train_loss'].append(avg_loss)
            
            # บันทึกลง TensorBoard
            if tensorboard_logger:
                try:
                    tensorboard_logger.log_scalar('train_loss', avg_loss, epoch)
                except Exception as e:
                    print(f"ไม่สามารถบันทึกลง TensorBoard ได้: {e}")
            
            # ประเมินกับชุดข้อมูล validation
            if val_loader:
                try:
                    val_loss = self.evaluate(val_loader)['loss']
                    history['val_loss'].append(val_loss)
                    
                    # บันทึกลง TensorBoard
                    if tensorboard_logger:
                        tensorboard_logger.log_scalar('val_loss', val_loss, epoch)
                    
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                except Exception as e:
                    logger.error(f"เกิดข้อผิดพลาดในการประเมินชุดข้อมูล validation: {e}")
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        
        # ปิด TensorBoard logger อย่างปลอดภัย
        if tensorboard_logger:
            try:
                tensorboard_logger.close()
            except:
                pass
        
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
        try:
            # สร้างไดเรกทอรีด้วย os.makedirs ก่อน
            import os
            os.makedirs(log_dir, exist_ok=True)
            
            # ลองเรียกใช้ TensorboardLogger
            from src.utils.loggers import TensorboardLogger
            return TensorboardLogger(log_dir)
        except Exception as e:
            logger.warning(f"ไม่สามารถตั้งค่า TensorboardLogger ได้: {e}")
            return None
    
    def store_experience(self, state, action, reward, next_state, done) -> None:
        """
        เก็บประสบการณ์ใหม่ลงใน replay buffer
        
        Parameters:
        state: สถานะปัจจุบัน
        action: การกระทำ
        reward: รางวัล
        next_state: สถานะถัดไป
        done: สถานะจบ
        """
        try:
            # ตรวจสอบว่า action เป็น integer และอยู่ในช่วงที่ถูกต้อง
            try:
                action = int(action)
                if action < 0 or action >= self.action_dim:
                    # ถ้า action ไม่ถูกต้อง ให้ใช้ค่าที่ถูกต้องแทน
                    action = action % self.action_dim
            except (ValueError, TypeError):
                # ถ้าไม่สามารถแปลงเป็น integer ได้ ให้ใช้ค่าเริ่มต้น
                action = 0
            
            # แปลงค่าให้เป็นประเภทที่ถูกต้อง
            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)
            reward = float(reward)
            done = bool(done)
            
            # เก็บประสบการณ์
            self.replay_buffer.push(state, action, reward, next_state, done)
            self.step_count += 1
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการเก็บประสบการณ์: {e}")