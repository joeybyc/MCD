"""
Author: B. Chen

Cell classification implementation for distinguishing cells from noise.
"""

import numpy as np
from typing import List
from ..interfaces import BaseClassifier, ClassificationModelWrapper


class CellClassifier(BaseClassifier):
    """Cell classifier for distinguishing cells from noise."""
    
    def __init__(self, model_wrapper: ClassificationModelWrapper):
        """
        Initialize cell classifier with model wrapper.
        
        Args:
            model_wrapper: Classification model wrapper
        """
        self.model_wrapper = model_wrapper
    
    def classify(self, image: np.ndarray) -> int:
        """
        Classify a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 for noise, 1 for cell)
        """
        return self.model_wrapper.predict(image)
    
    def classify_batch(self, images: List[np.ndarray]) -> List[int]:
        """
        Classify a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of classification results (0 for noise, 1 for cell)
        """
        return self.model_wrapper.predict_batch(images)