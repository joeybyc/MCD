"""
Author: B. Chen

Model management module for AS-OCT image processing.
Provides unified model creation and management interface.
"""

from .interfaces import BaseModel, BaseModelLoader
from .config import BaseModelConfig
from .factory import ModelFactory, register_model
from .loaders import HTTPModelLoader

__all__ = [
    # Core interfaces
    'BaseModel',
    'BaseModelLoader',
    
    # Configuration
    'BaseModelConfig',
    
    # Factory and registration
    'ModelFactory',
    'register_model',
    
    # Loaders
    'HTTPModelLoader'
]