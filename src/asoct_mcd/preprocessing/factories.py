"""
Author: B. Chen

Factory classes for creating preprocessing components.
"""

from typing import Dict, Type
from .interfaces import ImageConverter, ImageDenoiser
from .converters import GrayscaleConverter
from .denoisers import NLMDenoiser, MedianDenoiser


class ConverterFactory:
    """Factory for creating image converter instances."""
    
    _converters: Dict[str, Type[ImageConverter]] = {
        'grayscale': GrayscaleConverter,
    }
    
    @classmethod
    def create_converter(cls, converter_type: str, **kwargs) -> ImageConverter:
        """
        Create a converter instance.
        
        Args:
            converter_type: Type of converter ('grayscale')
            **kwargs: Parameters for the converter constructor
            
        Returns:
            Converter instance
            
        Raises:
            ValueError: If converter type is not supported
        """
        if converter_type not in cls._converters:
            available = ', '.join(cls._converters.keys())
            raise ValueError(f"Unknown converter: {converter_type}. Available: {available}")
        
        return cls._converters[converter_type](**kwargs)
    
    @classmethod
    def register_converter(cls, name: str, converter_class: Type[ImageConverter]) -> None:
        """
        Register a new converter type.
        
        Args:
            name: Name for the converter type
            converter_class: Converter class that implements ImageConverter interface
        """
        cls._converters[name] = converter_class
    
    @classmethod
    def get_available_converters(cls) -> list:
        """
        Get list of available converter types.
        
        Returns:
            List of available converter type names
        """
        return list(cls._converters.keys())


class DenoiserFactory:
    """Factory for creating image denoiser instances."""
    
    _denoisers: Dict[str, Type[ImageDenoiser]] = {
        'nlm': NLMDenoiser,
        'median': MedianDenoiser,
    }
    
    @classmethod
    def create_denoiser(cls, denoiser_type: str, **kwargs) -> ImageDenoiser:
        """
        Create a denoiser instance.
        
        Args:
            denoiser_type: Type of denoiser ('nlm' or 'median')
            **kwargs: Parameters for the denoiser constructor
            
        Returns:
            Denoiser instance
            
        Raises:
            ValueError: If denoiser type is not supported
        """
        if denoiser_type not in cls._denoisers:
            available = ', '.join(cls._denoisers.keys())
            raise ValueError(f"Unknown denoiser: {denoiser_type}. Available: {available}")
        
        return cls._denoisers[denoiser_type](**kwargs)
    
    @classmethod
    def register_denoiser(cls, name: str, denoiser_class: Type[ImageDenoiser]) -> None:
        """
        Register a new denoiser type.
        
        Args:
            name: Name for the denoiser type
            denoiser_class: Denoiser class that implements ImageDenoiser interface
        """
        cls._denoisers[name] = denoiser_class
    
    @classmethod
    def get_available_denoisers(cls) -> list:
        """
        Get list of available denoiser types.
        
        Returns:
            List of available denoiser type names
        """
        return list(cls._denoisers.keys())