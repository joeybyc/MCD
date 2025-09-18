"""
Author: B. Chen

Trained segmentation model implementation (e.g., UNet).
"""

import numpy as np
from typing import Protocol
from ..interfaces import BaseSegmentor


class TrainedModelWrapper(Protocol):
    """Protocol for trained model wrapper."""
    
    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Get segmentation mask from direct inference."""
        pass

class TrainedSegmentor(BaseSegmentor):
    """Segmentation using trained models like UNet."""
    
    def __init__(self, model_wrapper: TrainedModelWrapper):
        """
        Initialize trained segmentor with model wrapper.
        
        Args:
            model_wrapper: Trained model wrapper
        """
        self.model_wrapper = model_wrapper
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image using trained model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary mask in 0/255 format
        """
        # Direct inference, no prompts needed
        mask = self.model_wrapper.get_mask(image)
        
        return mask