"""
Author: B. Chen

Preprocessing module for AS-OCT image processing.
Provides image conversion and denoising capabilities through factory patterns.
"""

from .interfaces import ImageConverter, ImageDenoiser, GrayscaleDenoiser
from .factories import ConverterFactory, DenoiserFactory
from .pipeline import PreprocessingPipeline

__all__ = [
    'ImageConverter',
    'ImageDenoiser', 
    'GrayscaleDenoiser',
    'ConverterFactory',
    'DenoiserFactory',
    'PreprocessingPipeline'
]