import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger('utils.config')

class ConfigManager:
    """
    จัดการการตั้งค่าของระบบ
    
    รองรับการโหลดจากไฟล์ YAML, JSON และการอัพเดตค่าต่างๆ ในระหว่างการทำงาน
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        กำหนดค่าเริ่มต้นสำหรับ ConfigManager
        
        Parameters:
        config_path (str, optional): พาธไปยังไฟล์การตั้งค่า
        """
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
        """
        โหลดการตั้งค่าจากไฟล์
        
        Parameters:
        config_path (str): พาธไปยังไฟล์การตั้งค่า (YAML หรือ JSON)
        
        Returns:
        bool: True หากโหลดสำเร็จ, False หากไม่สำเร็จ
        """
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
        """
        บันทึกการตั้งค่าไปยังไฟล์
        
        Parameters:
        config_path (str): พาธไปยังไฟล์การตั้งค่า
        format (str): รูปแบบของไฟล์ ('yaml' หรือ 'json')
        
        Returns:
        bool: True หากบันทึกสำเร็จ, False หากไม่สำเร็จ
        """
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
        """
        ดึงค่าจากการตั้งค่า
        
        Parameters:
        key (str): คีย์ของการตั้งค่า (รองรับการเข้าถึงแบบ 'section.subsection.key')
        default (Any): ค่าเริ่มต้นหากไม่พบคีย์
        
        Returns:
        Any: ค่าของการตั้งค่า หรือค่าเริ่มต้นหากไม่พบคีย์
        """
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
    
    def set(self, key: str, value: Any) -> bool:
        """
        ตั้งค่าในการตั้งค่า
        
        Parameters:
        key (str): คีย์ของการตั้งค่า (รองรับการเข้าถึงแบบ 'section.subsection.key')
        value (Any): ค่าที่ต้องการตั้ง
        
        Returns:
        bool: True หากตั้งค่าสำเร็จ, False หากไม่สำเร็จ
        """
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
        """
        อัพเดตการตั้งค่าจาก dict
        
        Parameters:
        data (Dict[str, Any]): ข้อมูลการตั้งค่าใหม่
        """
        # อัพเดตการตั้งค่าแบบลึก (deep update)
        self._deep_update(self.config, data)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        อัพเดต dict แบบลึก (deep update)
        
        Parameters:
        target (Dict[str, Any]): dict เป้าหมาย
        source (Dict[str, Any]): dict ต้นฉบับ
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # ถ้าทั้งคู่เป็น dict ให้อัพเดตแบบลึก
                self._deep_update(target[key], value)
            else:
                # ถ้าไม่ใช่ dict ให้แทนที่ค่า
                target[key] = value
    
    def extract_subconfig(self, section: str) -> Dict[str, Any]:
        """
        ดึงการตั้งค่าย่อยจากส่วนที่ระบุ
        
        Parameters:
        section (str): ส่วนของการตั้งค่าที่ต้องการดึง
        
        Returns:
        Dict[str, Any]: การตั้งค่าย่อย
        """
        return self.config.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        แปลงการตั้งค่าเป็น dict
        
        Returns:
        Dict[str, Any]: การตั้งค่าในรูปแบบ dict
        """
        return self.config
    
    def reset(self):
        """
        รีเซ็ตการตั้งค่าเป็นค่าเริ่มต้น
        """
        self.config = {}
        self._load_default_config()
    
    def sections(self) -> List[str]:
        """
        ดึงรายการส่วนของการตั้งค่า
        
        Returns:
        List[str]: รายการส่วนของการตั้งค่า
        """
        return list(self.config.keys())

# Singleton instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    ดึง singleton instance ของ ConfigManager
    
    Parameters:
    config_path (str, optional): พาธไปยังไฟล์การตั้งค่า
    
    Returns:
    ConfigManager: instance ของ ConfigManager
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    elif config_path:
        # ถ้ามีการระบุไฟล์การตั้งค่า ให้โหลดเพิ่มเติม
        _config_instance.load_config(config_path)
    
    return _config_instance