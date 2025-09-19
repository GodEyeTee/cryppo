import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from src.environment.base_env import BaseEnv
from src.environment.simulators.trading_simulator import TradingSimulator
from src.environment.renderers.renderer import Renderer
from src.data.managers.data_manager import MarketDataManager
from src.utils.config import get_config

logger = logging.getLogger(__name__)

class TradingEnv(BaseEnv):
    ACTIONS = {'NONE': 0, 'LONG': 1, 'SHORT': 2, 'EXIT': 3}

    def __init__(
        self,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        window_size: int = 60,
        initial_balance: float = 1e4,
        transaction_fee: float = 0.0025,
        reward_fn: Union[str, Callable] = 'sharpe',
        use_position_info: bool = True,
        normalize_obs: bool = True,
        render_mode: Optional[str] = None,
        config=None
    ):
        self.config = config or get_config()
        env_cfg = self.config.extract_subconfig("environment")
        self.window_size = window_size or env_cfg.get("window_size", 60)
        self.initial_balance = initial_balance or env_cfg.get("initial_balance", 1e4)
        self.transaction_fee = transaction_fee or env_cfg.get("fee_rate", 0.0025)
        self.use_position_info = use_position_info
        self.normalize_obs = normalize_obs
        self.failure_penalty = float(env_cfg.get("failure_penalty", -1000.0))
        self.history = []
        self.current_step = 0

        if file_path:
            self.data_manager = MarketDataManager(
                file_path=file_path,
                symbol=symbol or env_cfg.get("symbol"),
                window_size=self.window_size,
                config=self.config
            )
            if not self.data_manager.data_loaded:
                raise ValueError(f"Failed to load data from {file_path}")
        else:
            self.data_manager = None
            logger.warning("No data source provided; load_data() must be called before use.")

        self.simulator = TradingSimulator(self.initial_balance, self.transaction_fee, config=self.config)
        self.renderer = Renderer(render_mode or env_cfg.get("render_mode"), config=self.config)
        
        action_space = spaces.Discrete(len(self.ACTIONS))
        obs_space = spaces.Box(-np.inf, np.inf, shape=(self.window_size, self._feature_dim()), dtype=np.float32)
        super().__init__(obs_space, action_space, render_mode, self.config)

        self.reward_function = self._get_reward_function(reward_fn)
    
    def _feature_dim(self):
        cols = self.data_manager.data.columns.tolist()
        dim = len(cols) - (1 if 'timestamp' in cols else 0)
        return dim + (3 if self.use_position_info else 0)
    
    def load_data(self, file_path: str) -> None:
        if not self.data_manager:
            self.data_manager = MarketDataManager(file_path=file_path, window_size=self.window_size, config=self.config)
        else:
            self.data_manager.load_data(file_path)
        if not self.data_manager.data_loaded:
            raise ValueError(f"Failed to load data from {file_path}")
    
    def _setup_observation_space(self) -> None:
        if self.data_manager.data is None:
            logger.error("Cannot set observation_space because data is not loaded")
            return
        
        num_features = self.data_manager.data.shape[1]
        if 'timestamp' in self.data_manager.data.columns:
            num_features -= 1
        
        if self.use_position_info:
            num_features += 3
        
        self.obs_shape = (self.window_size, num_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_shape,
            dtype=np.float32
        )
        logger.info(f"Set observation_space: {self.observation_space}")
    
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed, options)
        opts = options or {}
        idx = opts.get('start_index', 0)
        if opts.get('start_date') and 'timestamp' in self.data_manager.raw_data:
            dt = pd.to_datetime(opts['start_date'])
            idxs = np.searchsorted(self.data_manager.raw_data['timestamp'], dt)
            idx = idxs[0] if len(idxs) else idx
        self.current_step = max(0, min(idx, len(self.data_manager.data) - self.window_size))
        self.prices = self.data_manager.raw_data['close'].to_numpy()
        self.dates = self.data_manager.raw_data['timestamp'].to_numpy()
        self.simulator.reset(self.initial_balance)
        self.history.clear()
        obs = self._get_observation()
        info.update({'step': self.current_step, 'price': self.prices[self.current_step]})
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action: {action}, using NONE instead")
            action = self.ACTIONS['NONE']

        if self.done:
            next_obs = self._get_observation()
            return next_obs, 0.0, self.done, False, self.info

        self.steps += 1

        next_state, reward, done, info = self._process_action(action)
        self.state = next_state
        self.done = done
        self.info = info
        self.total_rewards += reward

        executed_action = info.get('action', action)
        price = info.get('price', 0.0)
        date = info.get('date')
        equity = info.get('equity', self.simulator.get_equity(price))

        self.history.append({
            'step': info.get('step', max(self.current_step - 1, 0)),
            'price': price,
            'date': date,
            'action': executed_action,
            'reward': reward,
            'balance': self.simulator.balance,
            'equity': equity,
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit,
            'failure_reason': info.get('failure_reason')
        })

        truncated = bool(self.max_steps and self.steps >= self.max_steps)

        if self.render_mode != "none":
            self.render()

        return next_state, reward, done, truncated, info

    def _process_action(self, action: int):
        if not self.action_space.contains(action):
            action = self.ACTIONS['NONE']

        step_index = self.current_step
        price = self.prices[step_index]
        prev_price = self.prices[step_index - 1] if step_index > 0 else price

        operation_ok = True
        if action == self.ACTIONS['LONG']:
            operation_ok = self.simulator.open_long_position(price)
        elif action == self.ACTIONS['SHORT']:
            operation_ok = self.simulator.open_short_position(price)
        elif action == self.ACTIONS['EXIT']:
            operation_ok = self.simulator.close_position(price)

        failure_reason = self.simulator.failure_reason
        if not operation_ok and failure_reason:
            info = self._make_info(price, step_index, action)
            info['failure_reason'] = failure_reason
            obs = self._get_observation()
            penalty = self.failure_penalty
            self.simulator.clear_failure()
            return obs, penalty, True, info

        self.simulator.update(price)
        failure_reason = self.simulator.failure_reason
        if failure_reason and not self.simulator.has_position():
            info = self._make_info(price, step_index, action)
            info['failure_reason'] = failure_reason
            obs = self._get_observation()
            penalty = self.failure_penalty
            self.simulator.clear_failure()
            return obs, penalty, True, info

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if done and self.simulator.has_position():
            self.simulator.close_position(price)

        reward = self.reward_function(self.simulator, price, prev_price)
        obs = self._get_observation()
        info = self._make_info(price, step_index, action)

        return obs, reward, done, info

    def _get_observation(self):
        df = self.data_manager.data
        start = max(0, self.current_step - self.window_size + 1)
        window = df.iloc[start:self.current_step+1].copy()
        # Keep only numeric columns to avoid datetime (e.g., close_time)
        window = window.select_dtypes(include=['number']).copy()
        
        arr = window.reindex(list(range(self.current_step - self.window_size + 1, self.current_step+1))).fillna(0).to_numpy()
        
        if self.use_position_info:
            pos = np.full((self.window_size, 3), [
                1 if self.simulator.position_type=='long' else -1 if self.simulator.position_type=='short' else 0,
                self.simulator.units,
                self.simulator.profit
            ])
            arr = np.hstack([arr, pos])
            
        return arr.astype(np.float32)
    
    def _make_info(self, price, step_index, action):
        date = None
        if hasattr(self, 'dates') and 0 <= step_index < len(self.dates):
            date = self.dates[step_index]

        balance = self.simulator.balance
        equity = self.simulator.get_equity(price)

        return {
            'step': step_index,
            'price': price,
            'date': date,
            'balance': balance,
            'equity': equity,
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit,
            'action': action,
            'failure_reason': None
        }

    def _get_initial_state(self, options: Dict[str, Any] = None) -> np.ndarray:
        return self._get_observation()
    
    def _render_frame(self) -> Optional[Union[np.ndarray, str]]:
        if self.data_manager is None or self.current_step >= len(self.prices):
            return None
        
        render_data = {
            'step': self.current_step,
            'price': self.prices[self.current_step - 1] if self.current_step > 0 else 0,
            'date': self.dates[self.current_step - 1] if self.current_step > 0 else None,
            'balance': self.simulator.balance,
            'equity': self.simulator.get_equity(self.prices[self.current_step - 1] if self.current_step > 0 else 0),
            'position': self.simulator.position_type,
            'units': self.simulator.units,
            'profit': self.simulator.profit,
            'return': self.simulator.total_return,
            'history': self.history
        }
        
        return self.renderer.render(render_data)
    
    def render(self):
        if self.render_mode == 'none': return None
        return self.renderer.render({'history': self.history})
    
    def close(self):
        super().close()
        if self.renderer:
            self.renderer.close()
    
    def _get_reward_function(self, rt: Union[str, Callable]) -> Callable:
        funcs = {
            'profit': lambda sim, c, p: sim.profit,
            'return': self._return_reward,
            'sharpe': self._sharpe_reward,
            'sortino': self._sortino_reward,
            'calmar': self._calmar_reward,
            'custom': self._custom_reward
        }
        return funcs.get(rt, funcs['profit'])
    
    def _return_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        if simulator.position_type == 'none':
            return 0.0
        
        price_change = (current_price - prev_price) / prev_price
        
        if simulator.position_type == 'long':
            return price_change * 100
        elif simulator.position_type == 'short':
            return -price_change * 100
        
        return 0.0
    
    def _sharpe_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        returns = self._return_reward(simulator, current_price, prev_price)
        
        if len(self.history) < 10:
            return returns
        # use only entries that contain reward
        returns_history = [h['reward'] for h in self.history[-10:] if 'reward' in h]
        returns_history.append(returns)
        
        mean_return = np.mean(returns_history)
        std_return = np.std(returns_history)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _sortino_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        returns = self._return_reward(simulator, current_price, prev_price)
        
        if len(self.history) < 10:
            return returns
        
        returns_history = [h['reward'] for h in self.history[-10:] if 'reward' in h]
        returns_history.append(returns)
        
        mean_return = np.mean(returns_history)
        negative_returns = [r for r in returns_history if r < 0]
        
        if not negative_returns:
            return mean_return
        
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))
        
        if downside_deviation == 0:
            return 0.0
        
        return mean_return / downside_deviation
    
    def _calmar_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        returns = self._return_reward(simulator, current_price, prev_price)
        
        if len(self.history) < 10:
            return returns
        
        returns_history = [h['reward'] for h in self.history[-10:] if 'reward' in h]
        returns_history.append(returns)
        
        mean_return = np.mean(returns_history) * 252
        
        equity_curve = [h['balance'] + h['profit'] for h in self.history]
        equity_curve.append(simulator.balance + simulator.profit)
        
        max_drawdown = 0.0
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        if max_drawdown == 0:
            return mean_return
        
        return mean_return / (max_drawdown * 100)
    
    def _custom_reward(self, simulator: TradingSimulator, current_price: float, prev_price: float) -> float:
        profit_reward = simulator.profit
        sharpe_reward = self._sharpe_reward(simulator, current_price, prev_price)
        return 0.7 * profit_reward + 0.3 * sharpe_reward
    
    def save_history(self, file_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            history = []
            for item in self.history:
                item_copy = item.copy()
                if 'date' in item_copy and item_copy['date'] is not None:
                    if isinstance(item_copy['date'], np.datetime64) or isinstance(item_copy['date'], pd.Timestamp):
                        item_copy['date'] = item_copy['date'].strftime('%Y-%m-%d %H:%M:%S')
                history.append(item_copy)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved trading history to: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving trading history: {e}")
            return False
    
    def load_history(self, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            for item in history:
                if 'date' in item and item['date'] is not None:
                    if isinstance(item['date'], str):
                        item['date'] = pd.to_datetime(item['date'])
            
            self.history = history
            logger.info(f"Loaded trading history from: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading trading history: {e}")
            return False
    
    def plot_performance(self, output_file: Optional[str] = None, show: bool = True) -> Optional[plt.Figure]:
        if not self.history:
            logger.warning("No trading history")
            return None
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        steps = [h['step'] for h in self.history]
        dates = [h['date'] for h in self.history]
        prices = [h['price'] for h in self.history]
        balances = [h['balance'] + h['profit'] for h in self.history]
        positions = [h['position'] for h in self.history]
        actions = [h['action'] for h in self.history]
        rewards = [h['reward'] for h in self.history]
        
        ax1.plot(steps, prices, label='Price', color='blue')
        ax1.set_ylabel('Price')
        ax1.set_title('Trading Performance')
        ax1.grid(True)
        
        ax1.plot(steps, balances, label='Equity', color='green', alpha=0.7)
        ax1.legend()
        
        for i, pos in enumerate(positions):
            if pos == 'long':
                ax1.scatter(steps[i], prices[i], marker='^', color='green', s=100)
            elif pos == 'short':
                ax1.scatter(steps[i], prices[i], marker='v', color='red', s=100)
        
        ax2.bar(steps, actions, color='purple', alpha=0.7)
        ax2.set_ylabel('Action')
        ax2.set_yticks(list(self.ACTIONS.values()))
        ax2.set_yticklabels(list(self.ACTIONS.keys()))
        ax2.grid(True)
        
        ax3.bar(steps, rewards, color='orange', alpha=0.7)
        ax3.set_ylabel('Reward')
        ax3.set_xlabel('Step')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved chart to: {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
