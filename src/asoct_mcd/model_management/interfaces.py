"""
Author: B. Chen

Abstract base classes and protocols for model management.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional
import torch


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, name: str, device: str = 'auto'):
        """
        Initialize base model wrapper.
        
        Args:
            name: Model identifier
            device: Device to load model on ('cpu', 'cuda', or 'auto')
        """
        self.name = name
        self.device = self._resolve_device(device)
        self.is_loaded = False
        self.model: Optional[Any] = None
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        pass
    
    def health_check(self) -> bool:
        """Check if model is loaded and healthy."""
        return self.is_loaded and self.model is not None
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata information."""
        pass


class SegmentationModelWrapper(BaseModelWrapper):
    """Abstract wrapper for segmentation models."""
    
    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform segmentation on input image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional segmentation parameters
            
        Returns:
            Binary mask in 0/255 format
        """
        pass


class ClassificationModelWrapper(BaseModelWrapper):
    """Abstract wrapper for classification models."""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predict classification for single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_batch(self, images: list) -> list:
        """
        Predict classification for batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of classification results
        """
        pass