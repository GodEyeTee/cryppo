"""
โมดูลสภาพแวดล้อมจำลองสำหรับ CRYPPO (Cryptocurrency Position Optimization)

โมดูลนี้ประกอบด้วยคลาสและฟังก์ชันสำหรับการจำลองสภาพแวดล้อมการเทรดที่ใช้ใน Reinforcement Learning
"""

from src.environment.base_env import BaseEnv
from src.environment.trading_env import TradingEnv
from src.environment.simulators.trading_simulator import TradingSimulator
from src.environment.renderers.renderer import Renderer

__all__ = [
    'BaseEnv',
    'TradingEnv',
    'TradingSimulator',
    'Renderer'
]

def create_trading_env(config=None, **kwargs):
    """
    สร้างสภาพแวดล้อมการเทรดใหม่
    
    Parameters:
    config (Config, optional): อ็อบเจ็กต์การตั้งค่า
    **kwargs: พารามิเตอร์เพิ่มเติมสำหรับ TradingEnv
    
    Returns:
    TradingEnv: อินสแตนซ์ของสภาพแวดล้อมการเทรด
    """
    return TradingEnv(config=config, **kwargs)

def create_simulator(config=None, **kwargs):
    """
    สร้างตัวจำลองการเทรดใหม่
    
    Parameters:
    config (Config, optional): อ็อบเจ็กต์การตั้งค่า
    **kwargs: พารามิเตอร์เพิ่มเติมสำหรับ TradingSimulator
    
    Returns:
    TradingSimulator: อินสแตนซ์ของตัวจำลองการเทรด
    """
    return TradingSimulator(config=config, **kwargs)

def create_renderer(config=None, **kwargs):
    """
    สร้างตัวแสดงผลใหม่
    
    Parameters:
    config (Config, optional): อ็อบเจ็กต์การตั้งค่า
    **kwargs: พารามิเตอร์เพิ่มเติมสำหรับ Renderer
    
    Returns:
    Renderer: อินสแตนซ์ของตัวแสดงผล
    """
    return Renderer(config=config, **kwargs)