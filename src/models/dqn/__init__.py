"""
โมดูลสำหรับอัลกอริทึม Deep Q-Network (DQN) และตัวแปรต่างๆ
"""

from src.models.dqn.dqn import DQN
from src.models.dqn.double_dqn import DoubleDQN
from src.models.dqn.dueling_dqn import DuelingDQN

__all__ = [
    'DQN',
    'DoubleDQN', 
    'DuelingDQN'
]

def create_dqn(model_type: str = 'dqn', **kwargs):
    """
    สร้างโมเดล DQN ตามประเภทที่ระบุ
    
    Parameters:
    model_type (str): ประเภทของโมเดล ('dqn', 'double_dqn', 'dueling_dqn')
    **kwargs: พารามิเตอร์เพิ่มเติมสำหรับโมเดล
    
    Returns:
    DQN or DoubleDQN or DuelingDQN: อินสแตนซ์ของโมเดล DQN
    """
    if model_type.lower() == 'dqn':
        return DQN(**kwargs)
    elif model_type.lower() == 'double_dqn':
        return DoubleDQN(**kwargs)
    elif model_type.lower() == 'dueling_dqn':
        return DuelingDQN(**kwargs)
    else:
        raise ValueError(f"ประเภทโมเดล '{model_type}' ไม่ถูกต้อง, ต้องเป็น 'dqn', 'double_dqn', หรือ 'dueling_dqn'")