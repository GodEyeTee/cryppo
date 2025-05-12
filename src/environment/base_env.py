import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from src.utils.config import get_config
from src.environment.simulators.trading_simulator import TradingSimulator
from src.environment.renderers.renderer import Renderer
from src.data.managers.data_manager import MarketDataManager

logger = logging.getLogger(__name__)

class BaseEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'console', 'none']}

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        render_mode: Optional[str] = None,
        config=None,
    ):
        self.config = config or get_config()
        env_cfg = self.config.extract_subconfig("environment")

        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = (-float('inf'), float('inf'))

        render_mode = render_mode or env_cfg.get("render_mode", "none")
        if render_mode not in self.metadata['render_modes']:
            logger.warning(f"Invalid render_mode '{render_mode}', defaulting to 'none'.")
            render_mode = 'none'
        self.render_mode = render_mode

        self.state = None
        self.done = False
        self.info = {}
        self.steps = 0
        self.episode = 0
        self.total_rewards = 0.0
        self.max_steps = env_cfg.get("max_episode_steps")

        self.np_random, _ = gym.utils.seeding.np_random(self.config.get("general.random_seed", None))
        logger.info(f"Initialized BaseEnv (render={self.render_mode})")
    
    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = self.config.get("general.random_seed", None)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        self.episode += 1
        self.steps = 0
        self.total_rewards = 0.0
        self.done = False
        self.info = {}
        if seed is not None:
            self.np_random.seed(seed)
        self.state = self._get_initial_state(options or {})
        return self.state, self.info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self.done:
            logger.warning("Call to step() after done=True. Reset first.")
            return self.state, 0.0, True, False, self.info
        self.steps += 1
        truncated = self.max_steps and self.steps >= self.max_steps
        next_state, reward, done, info = self._process_action(action)
        self.state, self.done = next_state, done
        self.total_rewards += reward
        info.update({'steps': self.steps, 'episode': self.episode, 'total_rewards': self.total_rewards})
        return next_state, reward, done, truncated, info
    
    def render(self):
        return None if self.render_mode == 'none' else self._render_frame()
    
    def close(self):
        pass
    
    def _get_initial_state(self, options: Dict[str, Any]) -> Any:
        raise NotImplementedError
    
    def _process_action(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError
    
    def _render_frame(self) -> Any:
        raise NotImplementedError
    
    def get_state_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'done': self.done,
            'info': self.info,
            'steps': self.steps,
            'episode': self.episode,
            'total_rewards': self.total_rewards
        }
    
    def set_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.state = state_dict.get('state', self.state)
        self.done = state_dict.get('done', self.done)
        self.info = state_dict.get('info', self.info)
        self.steps = state_dict.get('steps', self.steps)
        self.episode = state_dict.get('episode', self.episode)
        self.total_rewards = state_dict.get('total_rewards', self.total_rewards)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(obs_space={self.observation_space}, action_space={self.action_space}, render_mode={self.render_mode})"
