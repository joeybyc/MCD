"""
Author: B. Chen

Wrapper adapters for integrating with segmentation and classification modules.
"""

import numpy as np
from typing import Any, List
from .registry import ModelRegistry
from ..segmentation.interfaces import ZeroShotModelWrapper as SegmentationZeroShotModelWrapper
from ..classification.interfaces import ClassificationModelWrapper as ClassificationModelProtocol


class SAMModelAdapter:
    """Adapter to make SAM wrapper compatible with segmentation module."""
    
    def __init__(self, model_name: str = 'sam_vit_b'):
        """
        Initialize SAM adapter.
        
        Args:
            model_name: Name of SAM model in registry
        """
        self.model_name = model_name
        self.registry = ModelRegistry()
    
    def get_mask(self, image: np.ndarray, prompts: Any) -> np.ndarray:
        """
        Get segmentation mask using SAM model.
        
        Args:
            image: Input image as numpy array
            prompts: Prompts for segmentation (points, labels tuple)
            
        Returns:
            Binary mask in 0/255 format
        """
        sam_model = self.registry.get_model(self.model_name)
        return sam_model.segment(image, prompts)


class PyTorchClassificationAdapter:
    """Adapter to make PyTorch wrapper compatible with classification module."""
    
    def __init__(self, model_name: str = 'spatial_attention_network'):
        """
        Initialize PyTorch classification adapter.
        
        Args:
            model_name: Name of classification model in registry
        """
        self.model_name = model_name
        self.registry = ModelRegistry()
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict classification for single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 or 1)
        """
        model = self.registry.get_model(self.model_name)
        return model.predict(image)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[int]:
        """
        Predict classification for batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of classification results (0 or 1)
        """
        model = self.registry.get_model(self.model_name)
        return model.predict_batch(images)