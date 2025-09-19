import logging
from typing import Dict, Any, Optional, Type

from src.models.base_model import BaseModel
from src.models.dqn.dqn import DQN
from src.models.dqn.double_dqn import DoubleDQN
from src.models.dqn.dueling_dqn import DuelingDQN

logger = logging.getLogger('models.factory')

class ModelFactory:
    _models = {
        'dqn': DQN,
        'double_dqn': DoubleDQN,
        'dueling_dqn': DuelingDQN,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        cls._models[name.lower()] = model_class
        logger.info(f"Registered model {name}")
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Optional[Type[BaseModel]]:
        model_type = model_type.lower()
        
        if model_type in cls._models:
            return cls._models[model_type]
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
    
    @classmethod
    def create_model(cls, model_type: str, input_size: int, config: Any) -> Optional[BaseModel]:
        model_class = cls.get_model_class(model_type)
        
        if model_class:
            try:
                logger.info(f"Creating model {model_type} with input_size={input_size}")
                
                action_dim = config.get("environment.action_dim")
                if action_dim is None:
                    try:
                        from src.environment.trading_env import TradingEnv
                        action_dim = len(getattr(TradingEnv, "ACTIONS", {})) or 4
                    except Exception:
                        action_dim = 4  # Fallback when environment cannot be imported
                action_dim = int(action_dim)

                if issubclass(model_class, DQN):
                    model = model_class(
                        input_size=input_size, 
                        action_dim=action_dim, 
                        config=config
                    )
                else:
                    model = model_class(input_size=input_size, config=config)
                
                logger.info(f"Created model {model_type} with input_size={input_size}, action_dim={action_dim}")
                return model
            except Exception as e:
                logger.error(f"Error creating model {model_type}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return None
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Type[BaseModel]]:
        return cls._models
