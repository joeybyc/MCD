"""
Author: B. Chen

Model wrapper implementations for pipeline integration.
"""

import numpy as np
from typing import List, Any


class ZeroShotSegmentationWrapper:
    """Zero-shot segmentation model wrapper."""
    
    def __init__(self, model_wrapper):
        """Initialize with model wrapper."""
        self.model_wrapper = model_wrapper
    
    def get_mask(self, image: np.ndarray, prompts: Any) -> np.ndarray:
        """Get segmentation mask using prompts."""
        return self.model_wrapper.segment(image, prompts)


class TrainedSegmentationWrapper:
    """Trained segmentation model wrapper."""
    
    def __init__(self, model_wrapper):
        """Initialize with model wrapper."""
        self.model_wrapper = model_wrapper
    
    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Get segmentation mask from direct inference."""
        return self.model_wrapper.segment(image)


class ClassificationWrapper:
    """Classification model wrapper."""
    
    def __init__(self, model_wrapper):
        """Initialize with model wrapper."""
        self.model_wrapper = model_wrapper
    
    def predict(self, image: np.ndarray) -> int:
        """Predict classification for single image."""
        return self.model_wrapper.classify(image)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[int]:
        """Predict classification for batch of images."""
        return self.model_wrapper.classify_batch(images)


class PromptGenerationWrapper:
    """Prompt generation wrapper."""
    
    def __init__(self, prompt_generator):
        """Initialize with prompt generator."""
        self.prompt_generator = prompt_generator
    
    def generate(self, image: np.ndarray) -> Any:
        """Generate prompts for segmentation."""
        return self.prompt_generator.generate(image)