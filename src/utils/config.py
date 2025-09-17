"""Compatibility layer for configuration utilities.

This module re-exports the configuration helpers from
``src.utils.config_manager`` so that existing imports that reference
``src.utils.config`` continue to function.
"""

from .config_manager import ConfigManager, get_config

set_cuda_env = ConfigManager.set_cuda_env
update_config_from_args = ConfigManager.update_config_from_args

__all__ = [
    "ConfigManager",
    "get_config",
    "set_cuda_env",
    "update_config_from_args",
]
