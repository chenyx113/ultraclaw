"""
Configuration utilities for Ultra-Claw.

This module provides configuration loading and management.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel

from ultra_claw.core.models import AgentConfig
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (YAML)
        
    Returns:
        Loaded configuration
    """
    # Find config file
    if config_path is None:
        config_path = os.getenv("ULTRACLAW_CONFIG_PATH")
    
    if config_path is None:
        # Search for config in common locations
        search_paths = [
            Path("ultra-claw.yaml"),
            Path("config/ultra-claw.yaml"),
            Path.home() / ".ultra-claw" / "config.yaml",
            Path("/etc/ultra-claw/config.yaml"),
        ]
        
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path is None or not os.path.exists(config_path):
        logger.warning("No config file found, using defaults")
        return AgentConfig()
    
    # Load YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Expand environment variables
    config_dict = _expand_env_vars(config_dict)
    
    logger.info(f"Loaded config from {config_path}")
    return AgentConfig(**config_dict)


def save_config(config: AgentConfig, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Path to save to
    """
    config_dict = config.model_dump()
    
    # Ensure directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info(f"Saved config to {config_path}")


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config."""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            default = None
            if ":-" in var_name:
                var_name, default = var_name.split(":-", 1)
            return os.getenv(var_name, default)
        return obj
    return obj
