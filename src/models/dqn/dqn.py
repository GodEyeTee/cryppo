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
from collections import deque

from src.utils.config import get_config

logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        super(DQNNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.ReLU()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            if x.shape[-1] < self.input_dim:
                padding_size = self.input_dim - x.shape[-1]
                if x.dim() == 2:
                    padding = torch.zeros(x.size(0), padding_size, device=x.device)
                    x = torch.cat([x, padding], dim=1)
                elif x.dim() == 3:
                    padding = torch.zeros(x.size(0), x.size(1), padding_size, device=x.device)
                    x = torch.cat([x, padding], dim=2)
            elif x.shape[-1] > self.input_dim:
                if x.dim() == 2:
                    x = x[:, :self.input_dim]
                elif x.dim() == 3:
                    x = x[:, :, :self.input_dim]
        
        return self.network(x)

class DQN:
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
        self.config = config if config is not None else get_config()
        
        model_config = self.config.extract_subconfig("model")
        cuda_config = self.config.extract_subconfig("cuda")
        
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
        
        env_config = self.config.extract_subconfig("environment")

        self.trade_fee = float(env_config.get("fee_rate", 0.0))
        self.hold_penalty = float(env_config.get("hold_penalty", 0.0001))
        self.flat_penalty = float(env_config.get("flat_penalty", 0.00005))

        if device is None:
            self.device = torch.device("cuda" if cuda_config.get("use_cuda", True) and torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.policy_net = DQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.epsilon = self.epsilon_start
        self.step_count = 0
        self.update_count = 0
        self.train_count = 0
        
        logger.info(f"Created DQN (state_dim={state_dim}, action_dim={action_dim}, device={self.device})")
    
    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        if not evaluation and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            elif state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0)

            q_values = self.policy_net(state_tensor)

            if q_values.dim() == 3:
                q_values = q_values[:, -1, :]

            if q_values.dim() == 2:
                q_values = q_values.squeeze(0)

            action = int(torch.argmax(q_values).item())
            return max(0, min(action, self.action_dim - 1))
    
    def _calculate_trade_reward(self, action: int, price_delta: float) -> float:
        fee_penalty = abs(price_delta) * self.trade_fee

        if action == 1:  # LONG
            reward = price_delta - fee_penalty
        elif action == 2:  # SHORT
            reward = -price_delta - fee_penalty
        elif action == 3:  # EXIT
            reward = -abs(price_delta) * (self.hold_penalty * 2.0)
        else:  # NONE / HOLD
            reward = -abs(price_delta) * self.flat_penalty

        return float(np.tanh(reward))

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.train_count += 1
        
        return loss.item()
    
    def store_experience(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'train_count': self.train_count
        }, path)
        
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
                'model_type': 'dqn'
            }, f, indent=2)
        
        logger.info(f"Saved model to: {path}")
    
    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
        self.train_count = checkpoint['train_count']
        
        logger.info(f"Loaded model from: {path}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()
        return q_values
