import os
import logging
import logging.handlers
from typing import Optional, Dict, Any

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None, 
                formatter: Optional[logging.Formatter] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def setup_rotating_logger(name: str, level: int = logging.INFO, log_dir: str = 'logs', 
                         max_bytes: int = 10485760, backup_count: int = 5) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        os.makedirs(log_dir, exist_ok=True)
        
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
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        os.makedirs(log_dir, exist_ok=True)
        
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
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logging.warning("Tensorboard not found. Install with 'pip install tensorboard'")
    
    def log_scalar(self, tag: str, value: Any, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, Any], step: int):
        if self.writer:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: Any, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_graph(self, model, input_tensor):
        if self.writer:
            self.writer.add_graph(model, input_tensor)
    
    def close(self):
        if self.writer:
            self.writer.close()