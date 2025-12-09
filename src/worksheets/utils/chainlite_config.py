"""Centralized chainlite configuration management.

This module provides utilities for loading and managing the global chainlite
configuration from llm_config.yaml. It ensures the configuration is loaded
once and available throughout the application.
"""

import os
from pathlib import Path
from typing import Optional

from chainlite import initialize_llm_config
from loguru import logger

# Global flag to track if config has been loaded
_CONFIG_LOADED = False
_CONFIG_PATH: Optional[str] = None


def get_project_root() -> Path:
    """Get the project root directory by looking for pyproject.toml."""
    current_path = Path(__file__).resolve()
    
    # Walk up the directory tree to find pyproject.toml
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: assume we're in src/worksheets/utils and go up 3 levels
    return current_path.parent.parent.parent


def find_llm_config_path() -> str:
    """Find the llm_config.yaml file in the project."""
    project_root = get_project_root()
    config_path = project_root / "llm_config.yaml"
    
    if not config_path.exists():
        # Try alternative locations
        alternative_paths = [
            project_root / "config" / "llm_config.yaml",
            project_root / "src" / "llm_config.yaml",
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                config_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find llm_config.yaml in project root: {project_root}. "
                f"Please ensure the file exists at {project_root / 'llm_config.yaml'}"
            )
    
    return str(config_path)


def load_chainlite_config(config_path: Optional[str] = None, force_reload: bool = False) -> None:
    """Load the chainlite configuration from llm_config.yaml.
    
    This function ensures the configuration is loaded only once unless force_reload is True.
    
    Args:
        config_path: Optional path to the config file. If None, will auto-discover.
        force_reload: If True, reload the config even if already loaded.
    
    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    global _CONFIG_LOADED, _CONFIG_PATH
    
    if _CONFIG_LOADED and not force_reload:
        logger.debug(f"Chainlite config already loaded from: {_CONFIG_PATH}")
        return
    
    if config_path is None:
        config_path = find_llm_config_path()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        initialize_llm_config(config_path)
        _CONFIG_LOADED = True
        _CONFIG_PATH = config_path
        logger.info(f"Successfully loaded chainlite config from: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load chainlite config from {config_path}: {e}")
        raise


def ensure_chainlite_config_loaded() -> None:
    """Ensure chainlite config is loaded. Call this before using chainlite functions."""
    if not _CONFIG_LOADED:
        load_chainlite_config()


def get_config_path() -> Optional[str]:
    """Get the path of the currently loaded config file."""
    return _CONFIG_PATH


def is_config_loaded() -> bool:
    """Check if the chainlite config has been loaded."""
    return _CONFIG_LOADED


# Auto-load config when module is imported (optional - can be disabled)
def auto_load_config() -> None:
    """Automatically load config when module is imported."""
    try:
        load_chainlite_config()
    except Exception as e:
        logger.warning(f"Failed to auto-load chainlite config: {e}")
        logger.info("Config will be loaded on first use")


# Uncomment the line below if you want auto-loading on import
# auto_load_config() 