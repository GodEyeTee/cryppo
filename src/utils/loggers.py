import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None, 
                formatter: Optional[logging.Formatter] = None) -> logging.Logger:
    """
    ตั้งค่า logger สำหรับโมดูล
    
    Parameters:
    name (str): ชื่อของ logger
    level (int): ระดับของ log
    log_file (str, optional): ไฟล์สำหรับบันทึก log
    formatter (logging.Formatter, optional): formatter ของ log
    
    Returns:
    logging.Logger: logger ที่ตั้งค่าแล้ว
    """
    # สร้าง logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ตรวจสอบว่ามี handler แล้วหรือไม่
    if not logger.handlers:
        # กำหนด formatter
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # สร้าง console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # สร้าง file handler (ถ้ามีการระบุ log_file)
        if log_file:
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def setup_rotating_logger(name: str, level: int = logging.INFO, log_dir: str = 'logs', 
                         max_bytes: int = 10485760, backup_count: int = 5) -> logging.Logger:
    """
    ตั้งค่า logger แบบหมุนเวียนไฟล์ (rotating file)
    
    Parameters:
    name (str): ชื่อของ logger
    level (int): ระดับของ log
    log_dir (str): ไดเรกทอรีสำหรับบันทึก log
    max_bytes (int): ขนาดสูงสุดของไฟล์ log (bytes)
    backup_count (int): จำนวนไฟล์ log สำรอง
    
    Returns:
    logging.Logger: logger ที่ตั้งค่าแล้ว
    """
    # สร้าง logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ตรวจสอบว่ามี handler แล้วหรือไม่
    if not logger.handlers:
        # กำหนด formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # สร้าง console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # สร้างโฟลเดอร์หากไม่มี
        os.makedirs(log_dir, exist_ok=True)
        
        # สร้าง rotating file handler
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_daily_logger(name: str, level: int = logging.INFO, log_dir: str = 'logs', 
                      backup_count: int = 30) -> logging.Logger:
    """
    ตั้งค่า logger แบบรายวัน (daily)
    
    Parameters:
    name (str): ชื่อของ logger
    level (int): ระดับของ log
    log_dir (str): ไดเรกทอรีสำหรับบันทึก log
    backup_count (int): จำนวนไฟล์ log สำรอง
    
    Returns:
    logging.Logger: logger ที่ตั้งค่าแล้ว
    """
    # สร้าง logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ตรวจสอบว่ามี handler แล้วหรือไม่
    if not logger.handlers:
        # กำหนด formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # สร้าง console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # สร้างโฟลเดอร์หากไม่มี
        os.makedirs(log_dir, exist_ok=True)
        
        # สร้าง daily file handler
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y-%m-%d"
        logger.addHandler(file_handler)
    
    return logger

class TensorboardLogger:
    """
    บันทึก log สำหรับ TensorBoard
    """
    
    def __init__(self, log_dir: str):
        """
        กำหนดค่าเริ่มต้นสำหรับ TensorboardLogger
        
        Parameters:
        log_dir (str): ไดเรกทอรีสำหรับบันทึก log
        """
        self.log_dir = log_dir
        self.writer = None
        
        # ลองนำเข้า tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # สร้างโฟลเดอร์หากไม่มี
            os.makedirs(log_dir, exist_ok=True)
            
            # สร้าง writer
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logging.warning("ไม่พบ tensorboard โปรดติดตั้งด้วย 'pip install tensorboard'")
    
    def log_scalar(self, tag: str, value: Any, step: int):
        """
        บันทึกค่า scalar
        
        Parameters:
        tag (str): ชื่อของค่า
        value (Any): ค่าที่ต้องการบันทึก
        step (int): ขั้นตอนปัจจุบัน
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, Any], step: int):
        """
        บันทึกหลายค่า scalar
        
        Parameters:
        tag (str): ชื่อของกลุ่มค่า
        values (Dict[str, Any]): ค่าที่ต้องการบันทึก
        step (int): ขั้นตอนปัจจุบัน
        """
        if self.writer:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: Any, step: int):
        """
        บันทึก histogram
        
        Parameters:
        tag (str): ชื่อของ histogram
        values (Any): ค่าที่ต้องการบันทึก
        step (int): ขั้นตอนปัจจุบัน
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_graph(self, model, input_tensor):
        """
        บันทึก graph ของโมเดล
        
        Parameters:
        model: โมเดลที่ต้องการบันทึก
        input_tensor: tensor ตัวอย่างสำหรับการทำ forward pass
        """
        if self.writer:
            self.writer.add_graph(model, input_tensor)
    
    def close(self):
        """
        ปิด writer
        """
        if self.writer:
            self.writer.close()