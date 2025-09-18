"""
Author: B. Chen

Preprocessing pipeline for chaining multiple operations.
"""

from typing import List, Union
import numpy as np
from .interfaces import ImageConverter, ImageDenoiser


class PreprocessingPipeline:
    """Pipeline for executing multiple preprocessing operations in sequence."""
    
    def __init__(self):
        """Initialize empty preprocessing pipeline."""
        self.steps: List[Union[ImageConverter, ImageDenoiser]] = []
    
    def add_converter(self, converter: ImageConverter) -> 'PreprocessingPipeline':
        """
        Add an image converter to the pipeline.
        
        Args:
            converter: Image converter instance
            
        Returns:
            Self for method chaining
        """
        self.steps.append(converter)
        return self
    
    def add_denoiser(self, denoiser: ImageDenoiser) -> 'PreprocessingPipeline':
        """
        Add an image denoiser to the pipeline.
        
        Args:
            denoiser: Image denoiser instance
            
        Returns:
            Self for method chaining
        """
        self.steps.append(denoiser)
        return self
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Execute all preprocessing steps on the input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image after applying all pipeline steps
        """
        for step in self.steps:
            if isinstance(step, ImageConverter):
                image = step.convert(image)
            elif isinstance(step, ImageDenoiser):
                image = step.denoise(image)
        
        return image
    
    def clear(self) -> None:
        """Clear all steps from the pipeline."""
        self.steps.clear()
    
    def __len__(self) -> int:
        """Return number of steps in the pipeline."""
        return len(self.steps)