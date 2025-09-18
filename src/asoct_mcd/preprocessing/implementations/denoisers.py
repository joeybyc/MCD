"""
Author: B. Chen

Image denoising algorithm implementations.
"""

import cv2
import numpy as np
from ..interfaces import GrayscaleDenoiser


class NLMDenoiser(GrayscaleDenoiser):
    """Non-local means denoising algorithm implementation."""
    
    def __init__(self, h: int = 10, template_window: int = 7, search_window: int = 21):
        """
        Initialize NLM denoiser with parameters.
        
        Args:
            h: Filter strength. Higher h removes more noise but removes details too
            template_window: Size of template patch used to compute weights
            search_window: Size of search area used to compute weights
        """
        self.h = h
        self.template_window = template_window
        self.search_window = search_window
    
    def _denoise_impl(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising to validated grayscale image."""
        return cv2.fastNlMeansDenoising(
            image, None, self.h, self.template_window, self.search_window
        )


class MedianDenoiser(GrayscaleDenoiser):
    """Median filter denoising algorithm implementation."""
    
    def __init__(self, kernel_size: int = 1):
        """
        Initialize median denoiser.
        
        Args:
            kernel_size: Size of the median filter kernel (must be odd)
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        self.kernel_size = kernel_size
    
    def _denoise_impl(self, image: np.ndarray) -> np.ndarray:
        """Apply median filtering to validated grayscale image."""
        return cv2.medianBlur(image, self.kernel_size)