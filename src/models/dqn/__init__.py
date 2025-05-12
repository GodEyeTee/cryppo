from src.models.dqn.dqn import DQN
from src.models.dqn.double_dqn import DoubleDQN
from src.models.dqn.dueling_dqn import DuelingDQN

__all__ = ['DQN', 'DoubleDQN', 'DuelingDQN']

def create_dqn(model_type: str = 'dqn', **kwargs):
    if model_type.lower() == 'dqn':
        return DQN(**kwargs)
    elif model_type.lower() == 'double_dqn':
        return DoubleDQN(**kwargs)
    elif model_type.lower() == 'dueling_dqn':
        return DuelingDQN(**kwargs)
    else:
        raise ValueError(f"Invalid model type '{model_type}', must be 'dqn', 'double_dqn', or 'dueling_dqn'")
