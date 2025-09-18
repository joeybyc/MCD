"""
Author: B. Chen

Factory classes for creating threshold method components.
"""

from typing import Dict, Type
from .interfaces import ThresholdMethod
from .implementations.thresholding import OtsuThreshold, IsodataThreshold, MeanThreshold


class ThresholdFactory:
    """Factory for creating threshold method instances."""
    
    _methods: Dict[str, Type[ThresholdMethod]] = {
        'otsu': OtsuThreshold,
        'isodata': IsodataThreshold,
        'mean': MeanThreshold,
    }
    
    @classmethod
    def create_threshold(cls, method_type: str, **kwargs) -> ThresholdMethod:
        """
        Create a threshold method instance.
        
        Args:
            method_type: Type of threshold method ('otsu', 'isodata', or 'mean')
            **kwargs: Parameters for the threshold method constructor
            
        Returns:
            Threshold method instance
            
        Raises:
            ValueError: If method type is not supported
        """
        if method_type not in cls._methods:
            available = ', '.join(cls._methods.keys())
            raise ValueError(f"Unknown threshold method: {method_type}. Available: {available}")
        
        return cls._methods[method_type](**kwargs)
    
    @classmethod
    def register_method(cls, name: str, method_class: Type[ThresholdMethod]) -> None:
        """
        Register a new threshold method type.
        
        Args:
            name: Name for the threshold method type
            method_class: Method class that implements ThresholdMethod interface
        """
        cls._methods[name] = method_class
    
    @classmethod
    def get_available_methods(cls) -> list:
        """
        Get list of available threshold method types.
        
        Returns:
            List of available method type names
        """
        return list(cls._methods.keys())