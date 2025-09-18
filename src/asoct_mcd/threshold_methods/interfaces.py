"""
Author: B. Chen

Abstract base classes for threshold operations.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2


class ThresholdMethod(ABC):
    """Abstract base class for image thresholding operations."""
    
    def _validate_grayscale(self, image: np.ndarray) -> None:
        """Validate that image is grayscale (single channel)."""
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale (single channel)")
    
    @abstractmethod
    def _calculate_threshold(self, image: np.ndarray) -> float:
        """
        Calculate the threshold value for the given image.
        
        Args:
            image: Validated grayscale image
            
        Returns:
            Threshold value as float
        """
        pass
    
    def apply_threshold(self, image: np.ndarray, lambda_factor: float = 1.0, return_thresh: bool = False):
        """
        Apply thresholding to the input image.
        
        Args:
            image: Input grayscale image
            lambda_factor: Adjustment factor for threshold value
            return_thresh: Whether to return threshold value along with mask
            
        Returns:
            Binary mask if return_thresh=False, else tuple of (mask, threshold_value)
        """
        self._validate_grayscale(image)
        threshold_value = self._calculate_threshold(image) * lambda_factor
        _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        
        if return_thresh:
            return mask, threshold_value
        return mask