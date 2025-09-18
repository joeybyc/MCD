"""
Author: B. Chen

Model management module for AS-OCT image processing.
Provides centralized model loading, caching, and lifecycle management.
"""

from .interfaces import BaseModelWrapper, SegmentationModelWrapper, ClassificationModelWrapper
from .config import ModelConfig, DEFAULT_MODELS
from .registry import ModelRegistry
from .wrappers import SAMModelAdapter, PyTorchClassificationAdapter

__all__ = [
    # Core interfaces
    'BaseModelWrapper',
    'SegmentationModelWrapper', 
    'ClassificationModelWrapper',
    
    # Configuration
    'ModelConfig',
    'DEFAULT_MODELS',
    
    # Registry for centralized management
    'ModelRegistry',
    
    # Adapter classes for integration
    'SAMModelAdapter',
    'PyTorchClassificationAdapter'
]