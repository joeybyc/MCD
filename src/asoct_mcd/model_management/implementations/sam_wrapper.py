"""
Author: B. Chen

SAM model wrapper implementation.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Union
import torch
from segment_anything import sam_model_registry, SamPredictor

from ..interfaces import SegmentationModelWrapper
from ..config import ModelConfig
from ..loaders import ModelLoader


class SAMWrapper(SegmentationModelWrapper):
    """Wrapper for SAM (Segment Anything Model)."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize SAM wrapper.
        
        Args:
            config: SAM model configuration
        """
        super().__init__(config.name, config.device)
        self.config = config
        self.model_type = config.model_params.get('model_type', 'vit_b')
        self.loader = ModelLoader()
        self.predictor = None
    
    def load(self) -> None:
        """Load SAM model into memory."""
        if self.is_loaded:
            return
        
        # Ensure model file is available
        model_path = self.loader.prepare_model(self.config)
        
        try:
            # Load SAM model
            sam = sam_model_registry[self.model_type](checkpoint=model_path)
            sam.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(sam)
            self.model = sam
            self.is_loaded = True
            
            print(f"Successfully loaded SAM model: {self.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model {self.name}: {e}")
    
    def unload(self) -> None:
        """Release SAM model from memory."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def segment(self, image: np.ndarray, prompts: Union[Tuple[np.ndarray, np.ndarray], Any],
                **kwargs) -> np.ndarray:
        """
        Perform segmentation using SAM with prompts.
        
        Args:
            image: Input image as numpy array
            prompts: Tuple of (points, labels) or prompt object
            **kwargs: Additional parameters (unused)
            
        Returns:
            Binary mask in 0/255 format
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError(f"Model {self.name} is not loaded")
        
        # Ensure image is in RGB format for SAM
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be 2D or 3D array")
        
        # Extract points and labels from prompts
        if isinstance(prompts, tuple) and len(prompts) == 2:
            point_coords, point_labels = prompts
        else:
            raise ValueError("Prompts must be tuple of (points, labels)")
        
        # Set image for predictor
        self.predictor.set_image(image_rgb)
        
        # Perform prediction
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,  # Single mask output like original code
        )
        
        # Convert to binary mask (0/255 format)
        mask = np.squeeze(masks)
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        return binary_mask
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get SAM model metadata information."""
        return {
            'name': self.name,
            'type': 'sam',
            'model_type': self.model_type,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'checkpoint_path': self.config.local_path
        }