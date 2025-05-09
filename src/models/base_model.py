import os
import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple

from src.utils.loggers import TensorboardLogger

logger = logging.getLogger('models.base')

class BaseModel(ABC):
    """
    คลาสพื้นฐานสำหรับโมเดลทั้งหมด
    
    คลาสนี้กำหนด interface ที่โมเดลทั้งหมดต้องปฏิบัติตาม
    """
    
    def __init__(self, input_size: int, config: Any):
        """
        กำหนดค่าเริ่มต้นสำหรับโมเดล
        
        Parameters:
        input_size (int): ขนาดของ input
        config (Any): การตั้งค่าของโมเดล
        """
        self.input_size = input_size
        self.config = config
        
        # ดึงการตั้งค่าที่เกี่ยวข้อง
        model_config = config.extract_subconfig("model")
        cuda_config = config.extract_subconfig("cuda")
        
        # ตั้งค่า device
        self.use_cuda = cuda_config.get("use_cuda", True) and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{cuda_config.get('device', 0)}" if self.use_cuda else "cpu")
        
        # ตั้งค่า seed
        seed = config.get("general.random_seed", 42)
        self._set_seed(seed)
        
        # สร้างโมเดล
        self.model = self._create_model()
        
        # ย้ายโมเดลไปยัง device
        self.model = self.model.to(self.device)
        
        # สถานะการเทรน
        self.is_trained = False
        
        logger.info(f"สร้างโมเดล {self.__class__.__name__} สำเร็จ (device: {self.device})")
    
    def _set_seed(self, seed: int):
        """
        ตั้งค่า seed สำหรับความสามารถในการทำซ้ำ
        
        Parameters:
        seed (int): ค่า seed
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        """
        สร้างโมเดล
        
        Returns:
        torch.nn.Module: โมเดลที่สร้างแล้ว
        """
        pass
    
    @abstractmethod
    def train(self, train_loader, val_loader=None, epochs=None, log_dir=None) -> Dict[str, Any]:
        """
        เทรนโมเดล
        
        Parameters:
        train_loader: DataLoader สำหรับข้อมูลเทรน
        val_loader: DataLoader สำหรับข้อมูล validation
        epochs (int, optional): จำนวนรอบการเทรน
        log_dir (str, optional): ไดเรกทอรีสำหรับบันทึก log
        
        Returns:
        Dict[str, Any]: ประวัติการเทรน
        """
        pass
    
    @abstractmethod
    def predict(self, inputs) -> Any:
        """
        ทำนายผลลัพธ์
        
        Parameters:
        inputs: ข้อมูลนำเข้า
        
        Returns:
        Any: ผลลัพธ์การทำนาย
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader, metrics_list=None) -> Dict[str, float]:
        """
        ประเมินโมเดล
        
        Parameters:
        data_loader: DataLoader สำหรับข้อมูลทดสอบ
        metrics_list (List[str], optional): รายการ metrics ที่ต้องการประเมิน
        
        Returns:
        Dict[str, float]: ผลการประเมิน
        """
        pass
    
    def save(self, path: str) -> bool:
        """
        บันทึกโมเดล
        
        Parameters:
        path (str): พาธไปยังไฟล์ที่ต้องการบันทึก
        
        Returns:
        bool: True หากบันทึกสำเร็จ, False หากไม่สำเร็จ
        """
        try:
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # บันทึกข้อมูลที่จำเป็น
            state = {
                'model_state_dict': self.model.state_dict(),
                'model_class': self.__class__.__name__,
                'input_size': self.input_size,
                'is_trained': self.is_trained,
                'model_config': self.config.extract_subconfig("model")
            }
            
            # บันทึกไฟล์
            torch.save(state, path)
            logger.info(f"บันทึกโมเดลที่: {path}")
            
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกโมเดล: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        โหลดโมเดล
        
        Parameters:
        path (str): พาธไปยังไฟล์โมเดล
        
        Returns:
        bool: True หากโหลดสำเร็จ, False หากไม่สำเร็จ
        """
        try:
            # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
            if not os.path.exists(path):
                logger.error(f"ไม่พบไฟล์โมเดล: {path}")
                return False
            
            # โหลดโมเดล
            state = torch.load(path, map_location=self.device)
            
            # ตรวจสอบความเข้ากันได้
            if 'model_class' in state and state['model_class'] != self.__class__.__name__:
                logger.warning(f"ชนิดของโมเดลไม่ตรงกัน: {state['model_class']} vs {self.__class__.__name__}")
            
            if 'input_size' in state and state['input_size'] != self.input_size:
                logger.warning(f"ขนาดของ input ไม่ตรงกัน: {state['input_size']} vs {self.input_size}")
            
            # โหลด state dict
            self.model.load_state_dict(state['model_state_dict'])
            
            # ตั้งค่าสถานะการเทรน
            if 'is_trained' in state:
                self.is_trained = state['is_trained']
            
            logger.info(f"โหลดโมเดลจาก: {path}")
            
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            return False
    
    def summary(self) -> str:
        """
        แสดงข้อมูลสรุปของโมเดล
        
        Returns:
        str: ข้อมูลสรุปของโมเดล
        """
        # นับจำนวน parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # สร้างข้อความสรุป
        summary = f"โมเดล: {self.__class__.__name__}\n"
        summary += f"ขนาดของ input: {self.input_size}\n"
        summary += f"จำนวน parameters ทั้งหมด: {num_params:,}\n"
        summary += f"จำนวน parameters ที่เทรนได้: {num_trainable_params:,}\n"
        summary += f"Device: {self.device}\n"
        summary += f"สถานะการเทรน: {'เทรนแล้ว' if self.is_trained else 'ยังไม่ได้เทรน'}\n"
        
        # เพิ่มโครงสร้างของโมเดล
        summary += f"โครงสร้างของโมเดล:\n{self.model}\n"
        
        return summary
    
    def to(self, device: Union[str, torch.device]) -> 'BaseModel':
        """
        ย้ายโมเดลไปยัง device ที่ระบุ
        
        Parameters:
        device (str or torch.device): device ที่ต้องการย้ายไป
        
        Returns:
        BaseModel: โมเดลที่ย้ายแล้ว
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = self.model.to(self.device)
        return self
    
    def _prepare_input(self, inputs: Any) -> Any:
        """
        เตรียมข้อมูลนำเข้าให้พร้อมสำหรับโมเดล
        
        Parameters:
        inputs (Any): ข้อมูลนำเข้า
        
        Returns:
        Any: ข้อมูลนำเข้าที่เตรียมแล้ว
        """
        # แปลงเป็น tensor ถ้ายังไม่ใช่
        if not isinstance(inputs, torch.Tensor):
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            else:
                inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # ย้ายไปยัง device ที่ถูกต้อง
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        
        return inputs
    
    def _setup_tensorboard(self, log_dir: str) -> Optional[TensorboardLogger]:
        """
        ตั้งค่า TensorBoard logger
        
        Parameters:
        log_dir (str): ไดเรกทอรีสำหรับบันทึก log
        
        Returns:
        TensorboardLogger or None: TensorBoard logger หรือ None หากไม่สามารถตั้งค่าได้
        """
        if log_dir:
            return TensorboardLogger(log_dir)
        return None