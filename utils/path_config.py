"""
path_config.py - Centralized path configuration management for bioRAG

This module provides a single source of truth for all paths used in the bioRAG project.
Paths are loaded from configs/paths.yaml and can be accessed programmatically.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PathConfig:
    """
    Singleton class to manage project paths from a YAML configuration file.
    
    Usage:
        from utils.path_config import get_path
        
        # Get a path
        cache_dir = get_path('cache.upload_file')
        
        # Get absolute path
        abs_path = get_path('data.dataset_outputs', absolute=True)
    """
    
    _instance = None
    _config = None
    _project_root = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load the paths configuration from YAML file."""
        # Find project root (directory containing this file's parent's parent)
        current_file = Path(__file__).resolve()
        # Assuming this file is in utils/ or configs/
        self._project_root = current_file.parent.parent
        
        # Look for paths.yaml in configs directory
        config_path = self._project_root / 'configs' / 'paths.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Path configuration file not found: {config_path}\n"
                "Please create configs/paths.yaml"
            )
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def _get_nested_value(self, keys: str) -> Any:
        """
        Get a nested value from config using dot notation.
        
        Args:
            keys: Dot-separated path like 'cache.upload_file'
            
        Returns:
            The value at the specified path
        """
        parts = keys.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                raise KeyError(f"Path key '{keys}' not found in configuration")
        
        return value
    
    def get(self, key: str, absolute: bool = False, create: bool = False) -> str:
        """
        Get a path from the configuration.
        
        Args:
            key: Dot-separated path key (e.g., 'cache.upload_file')
            absolute: If True, return absolute path relative to project root
            create: If True, create the directory if it doesn't exist
            
        Returns:
            The path as a string
        """
        path_str = self._get_nested_value(key)
        
        if absolute:
            # If path is already absolute, return as-is
            if os.path.isabs(path_str):
                path = Path(path_str)
            else:
                # Make it absolute relative to project root
                path = self._project_root / path_str
        else:
            path = Path(path_str)
        
        if create:
            path.mkdir(parents=True, exist_ok=True)
        
        return str(path)
    
    def get_project_root(self) -> str:
        """Get the project root directory."""
        return str(self._project_root)
    
    def get_all_paths(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all paths in a section or the entire configuration.
        
        Args:
            section: Optional section name (e.g., 'cache', 'data')
            
        Returns:
            Dictionary of paths
        """
        if section:
            return self._config.get(section, {})
        return self._config


# Global instance
_path_config = PathConfig()


def get_path(key: str, absolute: bool = False, create: bool = False) -> str:
    """
    Convenience function to get a path from configuration.
    
    Args:
        key: Dot-separated path key (e.g., 'cache.upload_file')
        absolute: If True, return absolute path relative to project root
        create: If True, create the directory if it doesn't exist
        
    Returns:
        The path as a string
        
    Example:
        >>> from utils.path_config import get_path
        >>> cache_dir = get_path('cache.upload_file', absolute=True, create=True)
    """
    return _path_config.get(key, absolute=absolute, create=create)


def get_project_root() -> str:
    """Get the project root directory."""
    return _path_config.get_project_root()


def get_all_paths(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all paths in a section or the entire configuration.
    
    Args:
        section: Optional section name (e.g., 'cache', 'data')
        
    Returns:
        Dictionary of paths
    """
    return _path_config.get_all_paths(section)


# Convenience functions for commonly used paths
def get_cache_dir(subdir: str = '', absolute: bool = True, create: bool = True) -> str:
    """Get cache directory path."""
    base = get_path('base.cache', absolute=absolute, create=create)
    if subdir:
        path = os.path.join(base, subdir)
        if create:
            os.makedirs(path, exist_ok=True)
        return path
    return base


def get_data_dir(subdir: str = '', absolute: bool = True, create: bool = True) -> str:
    """Get data directory path."""
    base = get_path('data.dataset_outputs', absolute=absolute, create=create)
    if subdir:
        path = os.path.join(base, subdir)
        if create:
            os.makedirs(path, exist_ok=True)
        return path
    return base


def get_log_dir(subdir: str = '', absolute: bool = True, create: bool = True) -> str:
    """Get log directory path."""
    base = get_path('logs.base', absolute=absolute, create=create)
    if subdir:
        path = os.path.join(base, subdir)
        if create:
            os.makedirs(path, exist_ok=True)
        return path
    return base


if __name__ == '__main__':
    # Test the configuration
    print("Project root:", get_project_root())
    print("\nCache paths:")
    for key, value in get_all_paths('cache').items():
        print(f"  {key}: {value}")
    
    print("\nTest get_path:")
    print("  upload_file (relative):", get_path('cache.upload_file'))
    print("  upload_file (absolute):", get_path('cache.upload_file', absolute=True))
