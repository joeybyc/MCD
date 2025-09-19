"""
Author: B. Chen

Base configuration classes for model management.
"""

import os
from pathlib import Path
from dataclasses import dataclass
import torch


def get_default_cache_dir() -> str:
    """Get default cache directory following common ML library conventions."""
    # Check environment variable first (like HuggingFace does)
    env_cache = os.getenv('ASOCT_MCD_CACHE')
    if env_cache:
        return env_cache
    
    # Follow HuggingFace/PyTorch convention
    if os.name == 'nt':  # Windows
        cache_root = Path.home() / '.cache'
    else:  # Linux/Mac
        cache_root = Path(os.getenv('XDG_CACHE_HOME', Path.home() / '.cache'))
    
    return str(cache_root / 'asoct_mcd' / 'models')


@dataclass
class BaseModelConfig:
    """Base model configuration."""
    
    name: str
    device: str = 'auto'
    cache_dir: str = None  # Will use default if None
    
    def __post_init__(self):
        """Set default cache directory if not provided."""
        if self.cache_dir is None:
            self.cache_dir = get_default_cache_dir()
    
    def resolve_device(self) -> str:
        """Resolve device string to actual device."""
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device