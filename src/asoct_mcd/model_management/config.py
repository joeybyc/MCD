"""
Author: B. Chen

Base configuration classes for model management.
"""

from dataclasses import dataclass
import torch


@dataclass
class BaseModelConfig:
    """Base model configuration."""
    
    name: str
    device: str = 'auto'
    cache_dir: str = './models'
    
    def resolve_device(self) -> str:
        """Resolve device string to actual device."""
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device