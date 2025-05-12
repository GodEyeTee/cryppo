from src.environment.base_env import BaseEnv
from src.environment.trading_env import TradingEnv
from src.environment.simulators.trading_simulator import TradingSimulator
from src.environment.renderers.renderer import Renderer

__all__ = ['BaseEnv', 'TradingEnv', 'TradingSimulator', 'Renderer']

def create_trading_env(config=None, **kwargs):
    return TradingEnv(config=config, **kwargs)

def create_simulator(config=None, **kwargs):
    return TradingSimulator(config=config, **kwargs)

def create_renderer(config=None, **kwargs):
    return Renderer(config=config, **kwargs)
