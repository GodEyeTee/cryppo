import os
import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple

from src.utils.loggers import TensorboardLogger

logger = logging.getLogger('models.base')

class BaseModel(ABC):
    def __init__(self, input_size: int, config: Any):
        self.input_size = input_size
        self.config = config
        
        model_config = config.extract_subconfig("model")
        cuda_config = config.extract_subconfig("cuda")
        
        self.use_cuda = cuda_config.get("use_cuda", True) and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{cuda_config.get('device', 0)}" if self.use_cuda else "cpu")
        
        seed = config.get("general.random_seed", 42)
        self._set_seed(seed)
        
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        self.is_trained = False
        
        logger.info(f"Created model {self.__class__.__name__} (device: {self.device})")
    
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        pass
    
    @abstractmethod
    def train(self, train_loader, val_loader=None, epochs=None, log_dir=None) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def predict(self, inputs) -> Any:
        pass
    
    @abstractmethod
    def evaluate(self, data_loader, metrics_list=None) -> Dict[str, float]:
        pass
    
    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            state = {
                'model_state_dict': self.model.state_dict(),
                'model_class': self.__class__.__name__,
                'input_size': self.input_size,
                'is_trained': self.is_trained,
                'model_config': self.config.extract_subconfig("model")
            }
            
            torch.save(state, path)
            logger.info(f"Saved model to: {path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            state = torch.load(path, map_location=self.device)
            
            if 'model_class' in state and state['model_class'] != self.__class__.__name__:
                logger.warning(f"Model class mismatch: {state['model_class']} vs {self.__class__.__name__}")
            
            if 'input_size' in state and state['input_size'] != self.input_size:
                logger.warning(f"Input size mismatch: {state['input_size']} vs {self.input_size}")
            
            self.model.load_state_dict(state['model_state_dict'])
            
            if 'is_trained' in state:
                self.is_trained = state['is_trained']
            
            logger.info(f"Loaded model from: {path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def summary(self) -> str:
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"Model: {self.__class__.__name__}\n"
        summary += f"Input size: {self.input_size}\n"
        summary += f"Total parameters: {num_params:,}\n"
        summary += f"Trainable parameters: {num_trainable_params:,}\n"
        summary += f"Device: {self.device}\n"
        summary += f"Training status: {'Trained' if self.is_trained else 'Not trained'}\n"
        summary += f"Model structure:\n{self.model}\n"
        
        return summary
    
    def to(self, device: Union[str, torch.device]) -> 'BaseModel':
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = self.model.to(self.device)
        return self
    
    def _prepare_input(self, inputs: Any) -> Any:
        if not isinstance(inputs, torch.Tensor):
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            else:
                inputs = torch.tensor(inputs, dtype=torch.float32)
        
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        
        return inputs
    
    def _setup_tensorboard(self, log_dir: str) -> Optional[TensorboardLogger]:
        if log_dir:
            return TensorboardLogger(log_dir)
        return None
