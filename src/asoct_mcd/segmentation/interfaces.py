"""
Author: B. Chen

Abstract base classes and protocols for segmentation operations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Protocol, Any


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


class ZeroShotModelWrapper(Protocol):
    """Protocol for zero-shot model wrapper."""
    
    def get_mask(self, image: np.ndarray, prompts: Any) -> np.ndarray:
        """
        Get segmentation mask using prompts.
        
        Args:
            image: Input image as numpy array
            prompts: Prompts for segmentation (format depends on implementation)
            
        Returns:
            Binary mask in 0/255 format
        """
        pass


class TrainedModelWrapper(Protocol):
    """Protocol for trained model wrapper."""
    
    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Get segmentation mask from direct inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary mask in 0/255 format
        """
        pass


class PromptGeneratorWrapper(Protocol):
    """Protocol for prompt generator wrapper."""
    
    def generate(self, image: np.ndarray) -> Any:
        """
        Generate prompts for segmentation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Prompts (format depends on implementation)
        """
        pass