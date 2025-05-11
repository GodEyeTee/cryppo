import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger('utils.config')

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        # ค่าการตั้งค่าเริ่มต้น
        self.config = {}
        
        # โหลดการตั้งค่าเริ่มต้น
        self._load_default_config()
        
        # ถ้ามีการระบุไฟล์การตั้งค่า ให้โหลดเพิ่มเติม
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self):
        """
        โหลดการตั้งค่าเริ่มต้นจากไฟล์ default_config.yaml
        """
        try:
            # ค้นหาไฟล์การตั้งค่าเริ่มต้น
            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'configs',
                'default_config.yaml'
            )
            
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"โหลดการตั้งค่าเริ่มต้นจาก {default_config_path}")
            else:
                logger.warning(f"ไม่พบไฟล์การตั้งค่าเริ่มต้น {default_config_path}")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดการตั้งค่าเริ่มต้น: {e}")
    
    def load_config(self, config_path: str):
        if not os.path.exists(config_path):
            logger.error(f"ไม่พบไฟล์การตั้งค่า: {config_path}")
            return False
        
        try:
            # โหลดไฟล์การตั้งค่าตามนามสกุล
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.error(f"นามสกุลไฟล์ไม่รองรับ: {config_path}")
                return False
            
            # อัพเดตการตั้งค่าด้วยข้อมูลใหม่
            self.update_from_dict(config_data)
            
            logger.info(f"โหลดการตั้งค่าจาก {config_path}")
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดการตั้งค่า: {e}")
            return False
    
    def save_config(self, config_path: str, format: str = 'yaml'):
        try:
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # บันทึกไฟล์ตามรูปแบบที่กำหนด
            if format.lower() == 'yaml':
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            elif format.lower() == 'json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
            else:
                logger.error(f"รูปแบบไม่รองรับ: {format}")
                return False
            
            logger.info(f"บันทึกการตั้งค่าที่ {config_path}")
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกการตั้งค่า: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        
        # แยกคีย์ตามจุด
        key_parts = key.split('.')
        
        # เริ่มต้นที่ config หลัก
        current = self.config
        
        # เข้าถึงคีย์ทีละส่วน
        for part in key_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set_cuda_env():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    def set(self, key: str, value: Any) -> bool:
        if not key:
            return False
        
        # แยกคีย์ตามจุด
        key_parts = key.split('.')
        
        # เริ่มต้นที่ config หลัก
        current = self.config
        
        # เข้าถึงคีย์ทีละส่วนจนถึงส่วนสุดท้าย
        for i, part in enumerate(key_parts[:-1]):
            # ถ้าคีย์ไม่มีอยู่ ให้สร้างเป็น dict ว่าง
            if part not in current:
                current[part] = {}
            
            # ถ้าคีย์ไม่ใช่ dict ให้แทนที่ด้วย dict
            if not isinstance(current[part], dict):
                current[part] = {}
            
            current = current[part]
        
        # ตั้งค่าที่คีย์สุดท้าย
        current[key_parts[-1]] = value
        
        return True
    
    def update_from_dict(self, data: Dict[str, Any]):
        # อัพเดตการตั้งค่าแบบลึก (deep update)
        self._deep_update(self.config, data)
    
    def update_config_from_args(config, args, mapping):
        for arg_name, config_path in mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                value = getattr(args, arg_name)
                
                # ตรวจสอบกรณีพิเศษ
                if arg_name in ['stop_loss', 'take_profit'] and value is not None:
                    value = value / 100.0 
                    
                config.set(config_path, value)
        
        return config
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # ถ้าทั้งคู่เป็น dict ให้อัพเดตแบบลึก
                self._deep_update(target[key], value)
            else:
                # ถ้าไม่ใช่ dict ให้แทนที่ค่า
                target[key] = value
    
    def extract_subconfig(self, section: str) -> Dict[str, Any]:
        return self.config.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config
    
    def reset(self):
        self.config = {}
        self._load_default_config()
    
    def sections(self) -> List[str]:
        return list(self.config.keys())

# Singleton instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    elif config_path:
        # ถ้ามีการระบุไฟล์การตั้งค่า ให้โหลดเพิ่มเติม
        _config_instance.load_config(config_path)
    
    return _config_instance