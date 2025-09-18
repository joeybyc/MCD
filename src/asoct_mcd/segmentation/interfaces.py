"""
Author: B. Chen

Abstract base classes for segmentation operations.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSegmentor(ABC):
    """Abstract base class for image segmentation operations."""
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image to produce binary mask.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary mask in 0/255 format
        """
        pass