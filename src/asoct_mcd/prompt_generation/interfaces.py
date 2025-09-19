"""
Author: B. Chen

Abstract base classes for prompt generation operations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Any


class BasePromptGenerator(ABC):
    """Abstract base class for prompt generation operations."""
    
    @abstractmethod
    def generate(self, image: np.ndarray) -> Any:
        """
        Generate prompts from input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Generated prompts (format depends on implementation)
        """
        pass


class PointPromptGenerator(BasePromptGenerator):
    """Abstract base class for point-based prompt generators."""
    
    @abstractmethod
    def generate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point prompts from input image.
        
        Args:
            image: Input grayscale image as numpy array
            
        Returns:
            Tuple of (points, labels) where:
            - points: Array of shape (N, 2) containing (x, y) coordinates
            - labels: Array of shape (N,) containing point labels (1 for positive)
        """
        pass