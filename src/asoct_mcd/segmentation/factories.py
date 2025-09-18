"""
Author: B. Chen

Factory classes for creating segmentation components.
"""

from typing import Dict, Type, Any
from .interfaces import BaseSegmentor
from .implementations.zero_shot import ZeroShotSegmentor
from .implementations.trained import TrainedSegmentor


class SegmentorFactory:
    """Factory for creating segmentor instances."""
    
    _segmentors: Dict[str, Type[BaseSegmentor]] = {
        'zero_shot': ZeroShotSegmentor,
        'trained': TrainedSegmentor,
    }
    
    @classmethod
    def create_segmentor(cls, segmentor_type: str, **kwargs) -> BaseSegmentor:
        """
        Create a segmentor instance with injected dependencies.
        
        Args:
            segmentor_type: Type of segmentor ('zero_shot' or 'trained')
            **kwargs: Dependencies required by the segmentor
                     For zero_shot: model_wrapper, prompt_generator
                     For trained: model_wrapper
            
        Returns:
            Segmentor instance
            
        Raises:
            ValueError: If segmentor type is not supported or dependencies are missing
        """
        if segmentor_type not in cls._segmentors:
            available = ', '.join(cls._segmentors.keys())
            raise ValueError(f"Unknown segmentor: {segmentor_type}. Available: {available}")
        
        segmentor_class = cls._segmentors[segmentor_type]
        
        try:
            return segmentor_class(**kwargs)
        except TypeError as e:
            raise ValueError(f"Invalid dependencies for {segmentor_type}: {e}")
    
    @classmethod
    def register_segmentor(cls, name: str, segmentor_class: Type[BaseSegmentor]) -> None:
        """Register a new segmentor type."""
        cls._segmentors[name] = segmentor_class
    
    @classmethod
    def get_available_segmentors(cls) -> list:
        """Get list of available segmentor types."""
        return list(cls._segmentors.keys())