"""
Author: B. Chen

Threshold methods module for AS-OCT image processing.
Provides various thresholding algorithms through factory patterns.
"""

from .interfaces import ThresholdMethod
from .factories import ThresholdFactory

__all__ = [
    # Abstract interface for user extension
    'ThresholdMethod',
    
    # Factory class for creating implementations
    'ThresholdFactory'
]