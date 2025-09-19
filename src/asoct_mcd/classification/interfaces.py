"""
Author: B. Chen

Abstract base classes and protocols for classification operations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Protocol, List


class BaseClassifier(ABC):
    """Abstract base class for image classification operations."""
    
    @abstractmethod
    def classify(self, image: np.ndarray) -> int:
        """
        Classify a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 or 1)
        """
        pass
    
    @abstractmethod
    def classify_batch(self, images: List[np.ndarray]) -> List[int]:
        """
        Classify a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of classification results (0 or 1)
        """
        pass


class ClassificationModelWrapper(Protocol):
    """Protocol for classification model wrapper."""
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict classification for single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 or 1)
        """
        pass
    
    def predict_batch(self, images: List[np.ndarray]) -> List[int]:
        """
        Predict classification for batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of classification results (0 or 1)
        """
        pass