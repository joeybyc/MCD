"""
Author: B. Chen

Abstract base classes for image preprocessing operations.
"""

from abc import ABC, abstractmethod
import numpy as np


class ImageConverter(ABC):
    """Abstract base class for image format conversion operations."""
    
    @abstractmethod
    def convert(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image format.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Converted image as numpy array
        """
        pass


class ImageDenoiser(ABC):
    """Abstract base class for image denoising operations."""
    
    @abstractmethod
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Denoised image as numpy array
        """
        pass


class GrayscaleDenoiser(ImageDenoiser, ABC):
    """Abstract base class for denoisers that work on grayscale images only."""
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising with grayscale validation.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
            
        Raises:
            ValueError: If input image is not grayscale
        """
        self._validate_grayscale(image)
        return self._denoise_impl(image)
    
    def _validate_grayscale(self, image: np.ndarray) -> None:
        """Validate that image is grayscale (single channel)."""
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale (single channel)")
    
    @abstractmethod
    def _denoise_impl(self, image: np.ndarray) -> np.ndarray:
        """
        Implement the actual denoising algorithm.
        
        Args:
            image: Validated grayscale image
            
        Returns:
            Denoised image
        """
        pass