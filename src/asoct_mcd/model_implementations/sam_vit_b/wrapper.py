"""
Author: B. Chen

SAM ViT-B model wrapper implementation.
"""

import cv2
import numpy as np
from typing import Tuple, Union
import torch
from segment_anything import sam_model_registry, SamPredictor

from ...model_management import BaseModel, HTTPModelLoader, register_model
from .config import SAMViTBConfig


@register_model("sam_vit_b")
class SAMViTBWrapper(BaseModel):
    """SAM ViT-B model wrapper."""
    
    def __init__(self, config: SAMViTBConfig = None):
        """Initialize SAM wrapper."""
        self.config = config or SAMViTBConfig()
        self.loader = HTTPModelLoader(self.config.cache_dir)
        self._model = None
        self._predictor = None
        self.name = self.config.name
    
    def load(self) -> None:
        """Load SAM model into memory."""
        if self.is_loaded:
            return
        
        model_path = self.loader.download_if_needed(self.config)
        
        try:
            sam = sam_model_registry[self.config.model_type](checkpoint=model_path)
            sam.to(device=self.config.resolve_device())
            
            self._model = sam
            self._predictor = SamPredictor(sam)
            
            print(f"Loaded SAM model: {self.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {e}")
    
    def unload(self) -> None:
        """Release SAM model from memory."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        
        if self._model is not None:
            del self._model
            self._model = None
        
        if self.config.resolve_device() == 'cuda':
            torch.cuda.empty_cache()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._predictor is not None
    
    def segment(self, image: np.ndarray, prompts: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Segment image using SAM with point prompts.
        
        Args:
            image: Input image as numpy array
            prompts: Tuple of (points, labels)
            
        Returns:
            Binary mask in 0/255 format
        """
        if not self.is_loaded:
            self.load()
        
        # Convert to RGB for SAM
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Image must be 2D or 3D array")
        
        point_coords, point_labels = prompts
        
        # Set image and predict
        self._predictor.set_image(image_rgb)
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        
        # Squeeze the single-dimensional entries from the shape of an array
        mask = np.squeeze(masks)

        # Convert mask to binary format with values 0 and 255
        image_mask = (mask > 0).astype(np.uint8) * 255

        image_mask = self._select_mask_with_points(image_mask, point_coords)

        return image_mask
    
    def _select_mask_with_points(self, mask_image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Keeps objects in the mask image that contain specified points.

        Parameters:
        - mask_image (numpy.ndarray): Binary mask image with objects marked as 255 and background as 0.
        - points (numpy.ndarray): Array of points in the format [[w1, h1], [w2, h2], ...].

        Returns:
        - numpy.ndarray: Filtered mask image with only the objects containing the points.
        """

        # Find all the contours in the mask image
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw the selected contours
        selected_mask = np.zeros_like(mask_image)

        # Iterate through each contour
        for contour in contours:
            # Check if any point is inside the current contour
            for point in points:
                # Convert point to tuple format
                point_tuple = (int(point[0]), int(point[1]))

                # Check if the point is inside the contour
                if cv2.pointPolygonTest(contour, point_tuple, False) >= 0:
                    # Draw the contour (fill it) on the selected_mask
                    cv2.drawContours(selected_mask, [contour], -1, (255), thickness=cv2.FILLED)
                    break  # Break the loop if at least one point is inside the contour

        return selected_mask
