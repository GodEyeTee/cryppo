import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
from typing import Any, Dict, List, Optional

from src.models.dqn.dqn import DQN
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class DoubleDQN(DQN):
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
        if input_size is not None and state_dim is None:
            state_dim = input_size
        
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
        
        logger.info(f"Created Double DQN")
    
    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        try:
            batch = self.replay_buffer.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            
            cleaned_action_batch = []
            for action in action_batch:
                try:
                    action_val = int(action)
                    if action_val < 0 or action_val >= self.action_dim:
                        action_val = action_val % self.action_dim
                    cleaned_action_batch.append(action_val)
                except (ValueError, TypeError):
                    cleaned_action_batch.append(0)
            
            state_batch_np = np.array(state_batch, dtype=np.float32)
            action_batch_np = np.array(cleaned_action_batch, dtype=np.int64)
            reward_batch_np = np.array(reward_batch, dtype=np.float32)
            next_state_batch_np = np.array(next_state_batch, dtype=np.float32)
            done_batch_np = np.array(done_batch, dtype=np.float32)
            
            if state_batch_np.shape[-1] < self.state_dim:
                extra_columns = self.state_dim - state_batch_np.shape[-1]
                batch_size, seq_len, _ = state_batch_np.shape
                padding = np.zeros((batch_size, seq_len, extra_columns), dtype=np.float32)
                state_batch_np = np.concatenate([state_batch_np, padding], axis=2)
                next_state_batch_np = np.concatenate([next_state_batch_np, padding], axis=2)
            
            state_batch = torch.FloatTensor(state_batch_np).to(self.device)
            action_batch = torch.LongTensor(action_batch_np).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch_np).unsqueeze(1).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch_np).to(self.device)
            done_batch = torch.FloatTensor(done_batch_np).unsqueeze(1).to(self.device)
            
            if action_batch.dim() == 1:
                action_batch = action_batch.unsqueeze(1)
            
            try:
                q_values = self.policy_net(state_batch)
                
                if len(q_values.shape) == 3:
                    q_values = q_values[:, -1, :]
                    
                max_action = int(torch.max(action_batch).item())
                
                if max_action >= q_values.size(1):
                    action_batch = torch.clamp(action_batch, 0, q_values.size(1) - 1)
                
                current_q_values = q_values.gather(1, action_batch)
                
                with torch.no_grad():
                    next_q_values = self.policy_net(next_state_batch)
                    
                    if len(next_q_values.shape) == 3:
                        next_q_values = next_q_values[:, -1, :]
                    
                    next_action_batch = next_q_values.argmax(dim=1, keepdim=True)
                    
                    target_q_values = self.target_net(next_state_batch)
                    
                    if len(target_q_values.shape) == 3:
                        target_q_values = target_q_values[:, -1, :]
                    
                    target_q_values = target_q_values.gather(1, next_action_batch)
                    
                    target_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values
                
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
            
            except Exception as e:
                print(f"Error during Q-values calculation or loss computation: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        except Exception as e:
            print(f"Error during update: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
                'model_type': 'double_dqn'
            }, f, indent=2)
        
        logger.info(f"Saved model to: {path}")
    
    def _create_model(self) -> torch.nn.Module:
        return self.policy_net
    



    def train(self, train_loader, val_loader=None, epochs=None, log_dir=None) -> Dict[str, Any]:
        try:
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                tensorboard_logger = self._setup_tensorboard(log_dir)
            else:
                tensorboard_logger = None
        except Exception as e:
            print(f"Cannot setup TensorboardLogger: {e}")
            tensorboard_logger = None

        if epochs is None:
            epochs = 100

        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            epoch_loss = 0.0
            loss_updates = 0
            warmup_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    states, next_states, price_deltas, done_flags = batch
                else:
                    states = batch[0]
                    next_states = states.clone()
                    price_deltas = torch.zeros(states.size(0), dtype=torch.float32)
                    done_flags = torch.zeros(states.size(0), dtype=torch.float32)

                states_np = states.cpu().numpy()
                next_states_np = next_states.cpu().numpy()
                deltas_np = price_deltas.cpu().numpy()
                dones_np = done_flags.cpu().numpy()

                last_loss = None

                for idx_sample in range(states_np.shape[0]):
                    state = states_np[idx_sample]
                    next_state = next_states_np[idx_sample]
                    price_delta = float(deltas_np[idx_sample])
                    done = bool(dones_np[idx_sample])

                    action = self.select_action(state)
                    reward = self._calculate_trade_reward(action, price_delta)

                    self.store_experience(state, action, reward, next_state, done)

                    loss_value = self.update()
                    if loss_value is None:
                        continue

                    last_loss = loss_value
                    epoch_loss += loss_value
                    loss_updates += 1

                if last_loss is None:
                    warmup_batches += 1
                if batch_idx % 10 == 0:
                    if last_loss is None:
                        print(
                            f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: warming_up (buffer {len(self.replay_buffer)}/{self.batch_size})"
                        )
                    else:
                        print(
                            f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: {last_loss:.8f}"
                        )

            avg_loss = (epoch_loss / loss_updates) if loss_updates else None
            history['train_loss'].append(avg_loss)

            if tensorboard_logger and avg_loss is not None:
                try:
                    tensorboard_logger.log_scalar('train_loss', avg_loss, epoch)
                except Exception as e:
                    print(f"Cannot log to TensorBoard: {e}")

            loss_display = f"{avg_loss:.6f}" if avg_loss is not None else "n/a (buffering replay)"

            if val_loader:
                try:
                    val_loss = self.evaluate(val_loader)['loss']
                    history['val_loss'].append(val_loss)

                    if tensorboard_logger:
                        tensorboard_logger.log_scalar('val_loss', val_loss, epoch)

                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_display}, Val Loss: {val_loss:.6f}"
                    )
                except Exception as e:
                    logger.error(f"Error evaluating validation set: {e}")
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_display}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_display}")

            if warmup_batches:
                logger.debug(
                    f"Epoch {epoch+1}: skipped {warmup_batches} batches while filling replay buffer (updates={loss_updates})"
                )

        self.is_trained = True

        if tensorboard_logger:
            try:
                tensorboard_logger.close()
            except Exception:
                pass

        return history

    def predict(self, inputs) -> int:
        inputs = self._prepare_input(inputs)
        
        with torch.no_grad():
            q_values = self.policy_net(inputs)
            action = q_values.argmax().item()
        
        return action
    
    def evaluate(self, data_loader, metrics_list=None) -> Dict[str, float]:
        self.policy_net.eval()
        
        total_loss = 0.0
        num_samples = 0
        all_actions = []
        
        with torch.no_grad():
            for batch_idx, (states,) in enumerate(data_loader):
                states_np = states.cpu().numpy()
                
                batch_actions = []
                for state in states_np:
                    action = self.select_action(state, evaluation=True)
                    batch_actions.append(action)
                
                all_actions.extend(batch_actions)
                
                loss = 0.0
                
                total_loss += loss
                num_samples += len(states)
        
        avg_loss = total_loss / max(num_samples, 1)
        
        self.policy_net.train()
        
        metrics = {
            'loss': avg_loss,
            'action_distribution': {action: all_actions.count(action) / len(all_actions) for action in set(all_actions)} if all_actions else {}
        }
        
        return metrics
    
    def store_experience(self, state, action, reward, next_state, done) -> None:
        try:
            try:
                action = int(action)
                if action < 0 or action >= self.action_dim:
                    action = action % self.action_dim
            except (ValueError, TypeError):
                action = 0
            
            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)
            reward = float(reward)
            done = bool(done)
            
            self.replay_buffer.push(state, action, reward, next_state, done)
            self.step_count += 1
        except Exception as e:
            print(f"Error storing experience: {e}")
