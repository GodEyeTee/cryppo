import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any

class MLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        output_activation: Optional[str] = None,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        init_weights: bool = True
    ):
        super(MLPNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        self.output_activation_name = output_activation
        if output_activation == 'relu':
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = None
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            elif layer_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))
            
            self.layers.append(self.activation)
            
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in ['relu', 'leaky_relu', 'elu']:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.activation_name in ['tanh', 'sigmoid']:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.xavier_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x
    
class CNNNetwork(nn.Module):
    """
    เครือข่ายประสาทเทียมแบบ Convolutional Neural Network (CNN)
    
    สำหรับใช้กับข้อมูลอินพุตที่เป็นหลายมิติ เช่น รูปภาพหรือ time series ในการเรียนรู้แบบเสริมกำลัง
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],  # (channels, height, width) หรือ (channels, sequence_length)
        output_dim: int,
        conv_layers: List[Dict[str, Any]] = None,
        fc_layers: List[int] = [128, 64],
        activation: str = 'relu',
        output_activation: Optional[str] = None,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        flatten_output: bool = True,
        pool_type: str = 'max'
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับเครือข่าย CNN
        
        Parameters:
        input_shape (Tuple[int, ...]): รูปร่างของอินพุต (channels, height, width) หรือ (channels, sequence_length)
        output_dim (int): ขนาดของเอาต์พุต
        conv_layers (List[Dict[str, Any]], optional): รายละเอียดของชั้น convolutional, เช่น
            [{'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}, ...]
        fc_layers (List[int]): ขนาดของชั้นซ่อนแบบ fully-connected
        activation (str): ฟังก์ชันกระตุ้น ('relu', 'tanh', 'leaky_relu', 'elu')
        output_activation (str, optional): ฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต
        use_batch_norm (bool): ใช้ Batch Normalization หรือไม่
        dropout_rate (float): อัตราการ dropout
        flatten_output (bool): แปลงเอาต์พุตเป็นเวกเตอร์ 1 มิติหรือไม่
        pool_type (str): ประเภทของการ pooling ('max', 'avg')
        """
        super(CNNNetwork, self).__init__()
        
        # เก็บพารามิเตอร์
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.flatten_output = flatten_output
        
        # กำหนดค่าเริ่มต้นของ conv_layers ถ้าไม่ได้ระบุ
        if conv_layers is None:
            conv_layers = [
                {'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ]
        
        # กำหนดฟังก์ชันกระตุ้น
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # กำหนดฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต
        if output_activation == 'relu':
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = None
        
        # กำหนดประเภทของการ pooling
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(2)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(2)
        else:
            self.pool = nn.MaxPool2d(2)
        
        # สร้างชั้น convolutional
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]  # จำนวน channels ของอินพุต
        
        current_shape = list(input_shape)
        
        for i, layer_config in enumerate(conv_layers):
            # ดึงค่าพารามิเตอร์จาก layer_config
            filters = layer_config.get('filters', 32)
            kernel_size = layer_config.get('kernel_size', 3)
            stride = layer_config.get('stride', 1)
            padding = layer_config.get('padding', 0)
            
            # เพิ่มชั้น convolutional
            self.conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            
            # เพิ่ม batch normalization ถ้าจำเป็น
            if use_batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(filters))
            
            # เพิ่มฟังก์ชันกระตุ้น
            self.conv_layers.append(self.activation)
            
            # เพิ่ม pooling
            self.conv_layers.append(self.pool)
            
            # เพิ่ม dropout ถ้าจำเป็น
            if dropout_rate > 0:
                self.conv_layers.append(nn.Dropout2d(dropout_rate))
            
            # อัพเดตจำนวน channels สำหรับชั้นถัดไป
            in_channels = filters
            
            # อัพเดตรูปร่างของเอาต์พุตหลังจากผ่านชั้น convolutional
            current_shape[0] = filters
            
            # อัพเดตขนาดของภาพหลังจากผ่านชั้น convolutional
            h, w = current_shape[1], current_shape[2]
            h = ((h + 2 * padding - kernel_size) // stride) + 1
            w = ((w + 2 * padding - kernel_size) // stride) + 1
            
            # อัพเดตขนาดของภาพหลังจากผ่านชั้น pooling
            h = h // 2
            w = w // 2
            
            current_shape[1], current_shape[2] = h, w
        
        # คำนวณขนาดของข้อมูลหลังจากผ่านชั้น convolutional
        conv_output_size = int(np.prod(current_shape))
        
        # สร้างชั้น fully-connected
        self.fc_layers = nn.ModuleList()
        in_features = conv_output_size
        
        for i, out_features in enumerate(fc_layers):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            
            if use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_features))
            
            self.fc_layers.append(self.activation)
            
            if dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            
            in_features = out_features
        
        # ชั้นเอาต์พุต
        self.output_layer = nn.Linear(in_features, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        การส่งผ่านไปข้างหน้า (forward pass)
        
        Parameters:
        x (torch.Tensor): อินพุต
        
        Returns:
        torch.Tensor: เอาต์พุต
        """
        # ตรวจสอบรูปร่างของอินพุต
        if x.dim() == 3:
            x = x.unsqueeze(0)  # เพิ่มมิติ batch
        
        # ส่งผ่านชั้น convolutional
        for layer in self.conv_layers:
            x = layer(x)
        
        # แปลงเป็นเวกเตอร์ 1 มิติ
        if self.flatten_output:
            x = x.view(x.size(0), -1)
        
        # ส่งผ่านชั้น fully-connected
        for layer in self.fc_layers:
            x = layer(x)
        
        # ส่งผ่านชั้นเอาต์พุต
        x = self.output_layer(x)
        
        # ใช้ฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต (ถ้ามี)
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x

class LSTMNetwork(nn.Module):
    """
    เครือข่ายประสาทเทียมแบบ Long Short-Term Memory (LSTM)
    
    สำหรับใช้กับข้อมูลอินพุตที่เป็นลำดับ (sequence) เช่น ข้อมูลราคาย้อนหลัง
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        fc_layers: List[int] = None,
        output_activation: Optional[str] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับเครือข่าย LSTM
        
        Parameters:
        input_dim (int): ขนาดของอินพุตในแต่ละ time step
        hidden_dim (int): ขนาดของ hidden state
        num_layers (int): จำนวนชั้นของ LSTM
        output_dim (int): ขนาดของเอาต์พุต
        bidirectional (bool): ใช้ LSTM แบบสองทิศทางหรือไม่
        dropout (float): อัตราการ dropout
        batch_first (bool): อินพุตมีรูปร่างเป็น (batch_size, seq_len, input_dim) หรือไม่
        fc_layers (List[int], optional): ขนาดของชั้นซ่อนแบบ fully-connected หลังจาก LSTM
        output_activation (str, optional): ฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต
        """
        super(LSTMNetwork, self).__init__()
        
        # เก็บพารามิเตอร์
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # สร้างชั้น LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # คำนวณขนาดของ LSTM output
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # สร้างชั้น fully-connected หลังจาก LSTM
        fc_dims = fc_layers if fc_layers is not None else []
        self.fc_layers = nn.ModuleList()
        
        in_features = lstm_output_dim
        for out_features in fc_dims:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append(nn.ReLU())
            in_features = out_features
        
        # ชั้นเอาต์พุต
        self.output_layer = nn.Linear(in_features, output_dim)
        
        # กำหนดฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต
        if output_activation == 'relu':
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = None
    
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None, c0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        การส่งผ่านไปข้างหน้า (forward pass)
        
        Parameters:
        x (torch.Tensor): อินพุต (batch_size, seq_len, input_dim) หรือ (seq_len, batch_size, input_dim)
        h0 (torch.Tensor, optional): hidden state เริ่มต้น
        c0 (torch.Tensor, optional): cell state เริ่มต้น
        
        Returns:
        torch.Tensor: เอาต์พุต
        """
        # ตรวจสอบรูปร่างของอินพุต
        if self.batch_first and x.dim() == 2:
            x = x.unsqueeze(0)  # เพิ่มมิติ batch
        elif not self.batch_first and x.dim() == 2:
            x = x.unsqueeze(1)  # เพิ่มมิติ batch
        
        # สร้าง hidden state และ cell state เริ่มต้นถ้าไม่ได้ระบุ
        if h0 is None or c0 is None:
            batch_size = x.size(0) if self.batch_first else x.size(1)
            num_directions = 2 if self.bidirectional else 1
            
            h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=x.device)
        
        # ส่งผ่านชั้น LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # ใช้เอาต์พุตตัวสุดท้ายถ้า batch_first
        if self.batch_first:
            out = lstm_out[:, -1, :]
        else:
            out = lstm_out[-1, :, :]
        
        # ส่งผ่านชั้น fully-connected
        for layer in self.fc_layers:
            out = layer(out)
        
        # ส่งผ่านชั้นเอาต์พุต
        out = self.output_layer(out)
        
        # ใช้ฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต (ถ้ามี)
        if self.output_activation is not None:
            out = self.output_activation(out)
        
        return out

class ActorNetwork(nn.Module):
    """
    เครือข่ายประสาทเทียมสำหรับ Actor ใน Actor-Critic algorithm
    
    สำหรับใช้ใน Actor-Critic, A2C, A3C, PPO, DDPG, และ SAC
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous_action: bool = False,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        output_activation: Optional[str] = None,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับเครือข่าย Actor
        
        Parameters:
        state_dim (int): ขนาดของ state
        action_dim (int): ขนาดของ action
        continuous_action (bool): action เป็นแบบต่อเนื่องหรือไม่
        hidden_dims (List[int]): ขนาดของชั้นซ่อน
        activation (str): ฟังก์ชันกระตุ้น ('relu', 'tanh', 'leaky_relu')
        output_activation (str, optional): ฟังก์ชันกระตุ้นสำหรับชั้นเอาต์พุต
        use_batch_norm (bool): ใช้ Batch Normalization หรือไม่
        dropout_rate (float): อัตราการ dropout
        log_std_min (float): ค่าต่ำสุดของ log standard deviation (สำหรับ continuous action)
        log_std_max (float): ค่าสูงสุดของ log standard deviation (สำหรับ continuous action)
        """
        super(ActorNetwork, self).__init__()
        
        # เก็บพารามิเตอร์
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_action = continuous_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # ใช้ MLP เป็นพื้นฐาน
        self.base_network = MLPNetwork(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
        # ชั้นเอาต์พุต
        if continuous_action:
            # สำหรับ continuous action space, เอาต์พุตคือ mean และ log_std ของการกระทำ
            self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            
            # กำหนดฟังก์ชันกระตุ้นสำหรับ mean layer
            if output_activation == 'tanh':
                self.output_activation = nn.Tanh()
            else:
                self.output_activation = None
        else:
            # สำหรับ discrete action space, เอาต์พุตคือ logits ของการกระทำ
            self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        การส่งผ่านไปข้างหน้า (forward pass)
        
        Parameters:
        state (torch.Tensor): state
        
        Returns:
        torch.Tensor หรือ Tuple[torch.Tensor, torch.Tensor]: action probabilities หรือ (mean, log_std)
        """
        # ส่งผ่านเครือข่ายพื้นฐาน
        features = self.base_network(state)
        
        if self.continuous_action:
            # คำนวณ mean และ log_std ของการกระทำ
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            
            # จำกัดค่า log_std เพื่อความเสถียรภาพ
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            # ใช้ฟังก์ชันกระตุ้นสำหรับ mean (ถ้ามี)
            if self.output_activation is not None:
                mean = self.output_activation(mean)
            
            return mean, log_std
        else:
            # คำนวณ logits ของการกระทำ
            logits = self.output_layer(features)
            
            # คำนวณความน่าจะเป็นของการกระทำด้วย softmax
            probs = F.softmax(logits, dim=-1)
            
            return probs
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        สุ่มการกระทำจาก policy
        
        Parameters:
        state (torch.Tensor): state
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (action, log_prob, entropy) หรือ (action, log_prob, mean)
        """
        if self.continuous_action:
            # คำนวณ mean และ log_std ของการกระทำ
            mean, log_std = self.forward(state)
            
            # สร้างการแจกแจงปกติ
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            
            # สุ่มการกระทำและคำนวณ log probability
            x_t = normal.rsample()  # rsample ใช้ reparameterization trick
            action = x_t
            
            # คำนวณ log probability
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            
            # คำนวณ entropy
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
            
            return action, log_prob, entropy
        else:
            # คำนวณความน่าจะเป็นของการกระทำ
            probs = self.forward(state)
            
            # สร้างการแจกแจง Categorical
            dist = torch.distributions.Categorical(probs)
            
            # สุ่มการกระทำและคำนวณ log probability
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
            
            # คำนวณ entropy
            entropy = dist.entropy().unsqueeze(-1)
            
            return action, log_prob, entropy

class CriticNetwork(nn.Module):
    """
    เครือข่ายประสาทเทียมสำหรับ Critic ใน Actor-Critic algorithm
    
    สำหรับใช้ใน Actor-Critic, A2C, A3C, PPO, DDPG, และ SAC
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับเครือข่าย Critic
        
        Parameters:
        state_dim (int): ขนาดของ state
        action_dim (int, optional): ขนาดของ action (สำหรับ Q-function)
        hidden_dims (List[int]): ขนาดของชั้นซ่อน
        activation (str): ฟังก์ชันกระตุ้น ('relu', 'tanh', 'leaky_relu')
        use_batch_norm (bool): ใช้ Batch Normalization หรือไม่
        dropout_rate (float): อัตราการ dropout
        """
        super(CriticNetwork, self).__init__()
        
        # เก็บพารามิเตอร์
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ตรวจสอบว่าเป็น V-function หรือ Q-function
        self.is_q_function = action_dim is not None
        
        # กำหนดขนาดของอินพุต
        input_dim = state_dim
        if self.is_q_function:
            input_dim += action_dim
        
        # ใช้ MLP เป็นพื้นฐาน
        self.network = MLPNetwork(
            input_dim=input_dim,
            output_dim=1,  # V(s) หรือ Q(s, a) เป็นค่าเดียว
            hidden_dims=hidden_dims,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
    
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        การส่งผ่านไปข้างหน้า (forward pass)
        
        Parameters:
        state (torch.Tensor): state
        action (torch.Tensor, optional): action (สำหรับ Q-function)
        
        Returns:
        torch.Tensor: V(s) หรือ Q(s, a)
        """
        if self.is_q_function:
            if action is None:
                raise ValueError("ต้องระบุ action สำหรับ Q-function")
            
            # รวม state และ action
            x = torch.cat([state, action], dim=-1)
        else:
            # ใช้เฉพาะ state สำหรับ V-function
            x = state
        
        # ส่งผ่านเครือข่าย
        value = self.network(x)
        
        return value