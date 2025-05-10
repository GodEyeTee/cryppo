import logging
from typing import Dict, Any, Optional, Type

from src.models.base_model import BaseModel
from src.models.dqn.dqn import DQN
from src.models.dqn.double_dqn import DoubleDQN
from src.models.dqn.dueling_dqn import DuelingDQN

logger = logging.getLogger('models.factory')

class ModelFactory:
    """
    Factory สำหรับสร้างโมเดลต่างๆ
    """
    
    # ทะเบียนโมเดล
    _models = {
        'dqn': DQN,
        'double_dqn': DoubleDQN,
        'dueling_dqn': DuelingDQN,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """
        ลงทะเบียนโมเดลใหม่
        
        Parameters:
        name (str): ชื่อของโมเดล
        model_class (Type[BaseModel]): คลาสของโมเดล
        """
        cls._models[name.lower()] = model_class
        logger.info(f"ลงทะเบียนโมเดล {name}")
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Optional[Type[BaseModel]]:
        """
        ดึงคลาสของโมเดลจากชื่อ
        
        Parameters:
        model_type (str): ประเภทของโมเดล
        
        Returns:
        Type[BaseModel]: คลาสของโมเดล หรือ None หากไม่พบ
        """
        model_type = model_type.lower()
        
        if model_type in cls._models:
            return cls._models[model_type]
        else:
            logger.error(f"ไม่พบโมเดลประเภท: {model_type}")
            return None
    
    @classmethod
    def create_model(cls, model_type: str, input_size: int, config: Any) -> Optional[BaseModel]:
        """
        สร้างโมเดลจากชื่อ
        
        Parameters:
        model_type (str): ประเภทของโมเดล
        input_size (int): ขนาดของ input
        config (Any): การตั้งค่าของโมเดล
        
        Returns:
        BaseModel: โมเดลที่สร้างแล้ว หรือ None หากไม่สามารถสร้างได้
        """
        model_class = cls.get_model_class(model_type)
        
        if model_class:
            try:
                # กำหนดค่า action_dim และ input_size ตายตัว (ในกรณีฉุกเฉิน)
                # ค่าเหล่านี้ควรสอดคล้องกับข้อมูลจริง
                action_dim = 4  # แก้ไขตามจำนวนการกระทำจริง
                manual_input_size = 25  # แก้ไขตามจำนวนคอลัมน์จริง
                
                # ตรวจสอบว่าโมเดลเป็น DQN หรือ subclass ของ DQN หรือไม่
                if issubclass(model_class, DQN):
                    model = model_class(input_size=manual_input_size, action_dim=action_dim, config=config)
                else:
                    model = model_class(input_size=manual_input_size, config=config)
                
                logger.info(f"สร้างโมเดล {model_type} สำเร็จด้วย input_size={manual_input_size}, action_dim={action_dim}")
                return model
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการสร้างโมเดล {model_type}: {e}")
                return None
        
        return None
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """
        แสดงรายการโมเดลที่มีอยู่
        
        Returns:
        Dict[str, Type[BaseModel]]: รายการโมเดลที่มีอยู่
        """
        return cls._models