"""
Author: B. Chen

I2ACP (Image-to-AC-Prompt) algorithm implementation.
"""

import cv2
import numpy as np
from typing import Tuple
from ..interfaces import PointPromptGenerator
from ...preprocessing import DenoiserFactory, ConverterFactory


class I2ACPGenerator(PointPromptGenerator):
    """I2ACP algorithm for generating AC region prompts."""
    
    def __init__(self, offset_ratio: float = 0.02, area_ratio_threshold: float = 0.65):
        """
        Initialize I2ACP generator.
        
        Args:
            offset_ratio: Ratio of image width for prompt point offsets (default: 0.1)
            area_ratio_threshold: Threshold for merging disconnected AS components (default: 0.65)
        """
        self.offset_ratio = offset_ratio
        self.area_ratio_threshold = area_ratio_threshold
        self.denoiser = DenoiserFactory.create_denoiser('nlm')
        self.converter = ConverterFactory.create_converter('grayscale')
    
    def generate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prompt points for AC segmentation using I2ACP algorithm."""
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = self.converter.convert(image)
        elif len(image.shape) != 2:
            raise ValueError("Input image must be 2D or 3D array")
        
        # Apply denoising
        denoised = self.denoiser.denoise(image)
        
        # Create binary mask using denoised image mean as threshold 
        threshold = np.mean(denoised)
        _, binary_mask = cv2.threshold(denoised, threshold, 255, cv2.THRESH_BINARY)
        
        # Find anterior segment components
        as_mask = self._find_anterior_segment(binary_mask)
        
        # Calculate centroid and generate prompt points using original image dimensions
        centroid_x, centroid_y = self._get_centroid(as_mask)
        points, labels = self._generate_prompt_points(image, centroid_x, centroid_y)
        
        return points, labels
    
    def _find_anterior_segment(self, binary_mask: np.ndarray) -> np.ndarray:
        """Find and merge anterior segment components."""
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels < 2:
            return binary_mask
        
        # Find largest and second largest components in O(n)
        largest_label = largest_area = 0
        second_largest_label = second_largest_area = 0
        
        for label in range(1, num_labels):
            area = np.sum(labels == label)
            
            if area > largest_area:
                # Update second largest before largest
                second_largest_area = largest_area
                second_largest_label = largest_label
                # Update largest
                largest_area = area
                largest_label = label
            elif area > second_largest_area:
                second_largest_area = area
                second_largest_label = label
        
        # Create mask with largest component
        result_mask = np.zeros_like(binary_mask)
        result_mask[labels == largest_label] = 255
        
        # Merge second largest if it meets ratio threshold
        if second_largest_area > 0:
            area_ratio = second_largest_area / largest_area
            if area_ratio > self.area_ratio_threshold:
                result_mask[labels == second_largest_label] = 255
        
        return result_mask
    
    def _get_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Calculate centroid of mask, fallback to image center if no segment found."""
        rows, cols = np.where(mask == 255)
        
        if len(rows) == 0:
            # Fallback: use image center as centroid
            height, width = mask.shape
            return width // 2, height // 2
        
        centroid_x = int(np.mean(cols))
        centroid_y = int(np.mean(rows))
        return centroid_x, centroid_y
    
    def _generate_prompt_points(self, image: np.ndarray, centroid_x: int, 
                              centroid_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prompt points around centroid."""
        height, width = image.shape
        offset = int(width * self.offset_ratio)
        
        # Generate two points with horizontal offset
        point1 = [centroid_x + offset, centroid_y]
        point2 = [centroid_x - offset, centroid_y]
        
        points = np.array([point1, point2])
        labels = np.array([1, 1])  # Both positive prompts
        
        return points, labels