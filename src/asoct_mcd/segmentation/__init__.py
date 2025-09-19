"""
Author: B. Chen

Segmentation module for AS-OCT image processing.
Provides unified segmentation interface and dependency injection for models.
"""

from .interfaces import BaseSegmentor, ZeroShotModelWrapper, TrainedModelWrapper, PromptGeneratorWrapper
from .factories import SegmentorFactory

__all__ = [
    # Abstract interface and protocols
    'BaseSegmentor',
    'ZeroShotModelWrapper',
    'TrainedModelWrapper', 
    'PromptGeneratorWrapper',
    
    # Factory class for creating implementations
    'SegmentorFactory'
]