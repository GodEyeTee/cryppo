"""Configuration management utilities for CRYPPO.

This module exposes a :class:`ConfigManager` that is responsible for loading,
storing and mutating configuration dictionaries.  It also contains a couple of
helper functions that are consumed by the CLI – notably ``set_cuda_env`` which
was previously missing from the public interface and caused imports to fail.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger('utils.config')

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        self._load_default_config()
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self):
        try:
            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'configs',
                'default_config.yaml'
            )
            
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded default config from {default_config_path}")
            else:
                logger.warning(f"Default config file not found: {default_config_path}")
        except Exception as e:
            logger.error(f"Error loading default config: {e}")
    
    def load_config(self, config_path: str):
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
        
        try:
            if config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.error(f"Unsupported file extension: {config_path}")
                return False
            
            self.update_from_dict(config_data)
            logger.info(f"Loaded config from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    def save_config(self, config_path: str, format: str = 'yaml'):
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if format.lower() == 'yaml':
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            elif format.lower() == 'json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Saved config to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        key_parts = key.split('.')
        current = self.config
        for part in key_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> bool:
        if not key:
            return False
        key_parts = key.split('.')
        current = self.config
        for i, part in enumerate(key_parts[:-1]):
            if part not in current:
                current[part] = {}
            if not isinstance(current[part], dict):
                current[part] = {}
    
            current = current[part]
        current[key_parts[-1]] = value
        
        return True
    
    def update_from_dict(self, data: Dict[str, Any]):
        self._deep_update(self.config, data)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
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

_config_instance = None


def set_cuda_env() -> None:
    """Configure CUDA related environment variables for deterministic runs."""

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("Enabled CUDA_LAUNCH_BLOCKING for deterministic CUDA execution")


def update_config_from_args(config: ConfigManager, args: Any, mapping: Dict[str, str]) -> ConfigManager:
    """Apply CLI arguments to the configuration object.

    Parameters
    ----------
    config:
        Instance of :class:`ConfigManager` to be updated.
    args:
        Parsed arguments object, typically from :mod:`argparse`.
    mapping:
        Dictionary mapping argument names to configuration paths within the
        configuration dictionary.
    """

    for arg_name, config_path in mapping.items():
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)

            if value is None:
                continue

            if arg_name in ["stop_loss", "take_profit"]:
                value = value / 100.0

            config.set(config_path, value)

    return config


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    elif config_path:
        _config_instance.load_config(config_path)
    
    return _config_instance