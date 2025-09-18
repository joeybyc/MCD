"""
Author: B. Chen

Classification module for AS-OCT image processing.
Provides unified classification interface for distinguishing cells from noise.
"""

from .interfaces import BaseClassifier, ClassificationModelWrapper
from .factories import ClassifierFactory

__all__ = [
    # Abstract interface and protocol
    'BaseClassifier',
    'ClassificationModelWrapper',
    
    # Factory class for creating implementations
    'ClassifierFactory'
]