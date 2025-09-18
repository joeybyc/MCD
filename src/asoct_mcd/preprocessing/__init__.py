"""
Author: B. Chen

Preprocessing module for AS-OCT image processing.
Provides image conversion and denoising capabilities through factory patterns.
"""

from .interfaces import ImageConverter, ImageDenoiser, GrayscaleDenoiser
from .factories import ConverterFactory, DenoiserFactory
from .pipeline import PreprocessingPipeline

__all__ = [
    # Abstract interfaces for user extension
    'ImageConverter',
    'ImageDenoiser', 
    'GrayscaleDenoiser',
    
    # Factory classes for creating implementations
    'ConverterFactory',
    'DenoiserFactory',
    
    # Pipeline for chaining operations
    'PreprocessingPipeline'
]