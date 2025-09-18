"""
Author: B. Chen

Image format conversion implementations.
"""

import cv2
import numpy as np
from .interfaces import ImageConverter


class GrayscaleConverter(ImageConverter):
    """Converts color images to grayscale format."""
    
    def convert(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if it's in color format.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image