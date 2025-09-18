"""
Author: B. Chen

Prompt generation module for AS-OCT image analysis.
Provides prompt generation capabilities for zero-shot segmentation.
"""

from .interfaces import BasePromptGenerator, PointPromptGenerator
from .factories import PromptGeneratorFactory

__all__ = [
    # Abstract interfaces for user extension
    'BasePromptGenerator',
    'PointPromptGenerator',
    
    # Factory class for creating implementations
    'PromptGeneratorFactory'
]