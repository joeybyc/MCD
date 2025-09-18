"""
Author: B. Chen

Zero-shot segmentation implementation using vision foundation models.
"""

import numpy as np
from typing import Protocol, Tuple, Any
from ..interfaces import BaseSegmentor


class ZeroShotModelWrapper(Protocol):
    """Protocol for zero-shot model wrapper."""
    
    def get_mask(self, image: np.ndarray, prompts: Any) -> np.ndarray:
        """Get segmentation mask using point prompts."""
        pass


class PromptGenerator(Protocol):
    """Protocol for prompt generator."""
    
    def generate(self, image: np.ndarray) -> Any:
        """Generate prompts for segmentation."""
        pass


class ZeroShotSegmentor(BaseSegmentor):
    """Zero-shot segmentation using model wrapper and prompt generator."""
    
    def __init__(self, model_wrapper: ZeroShotModelWrapper, 
                 prompt_generator: PromptGenerator):
        """
        Initialize zero-shot segmentor with dependencies.
        
        Args:
            model_wrapper: Zero-shot model wrapper
            prompt_generator: Prompt generator
        """
        self.model_wrapper = model_wrapper
        self.prompt_generator = prompt_generator
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image using zero-shot approach with prompts.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary mask in 0/255 format
        """
        # Generate prompts
        prompts = self.prompt_generator.generate(image)
        
        # Get mask from model wrapper
        mask = self.model_wrapper.get_mask(
            image=image,
            prompts=prompts
        )
        
        return mask