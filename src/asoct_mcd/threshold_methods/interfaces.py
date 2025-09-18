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
    
    def apply_threshold(self, image: np.ndarray, lambda_factor: float = 1, 
                    return_thresh: bool = False, output_format: str = 'uint8'):
        """
        Args:
            output_format: 'uint8' for 0/255 format, 'bool' for True/False format
        """
        self._validate_grayscale(image)
        threshold_value = self._calculate_threshold(image) * lambda_factor
        _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        
        if output_format == 'bool':
            mask = mask.astype(bool)
        
        if return_thresh:
            return mask, threshold_value
        return mask