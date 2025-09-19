"""
Author: B. Chen

Core MCD pipeline implementation.
"""

import cv2
import numpy as np
from typing import List, Tuple

from ..image_processing import intersect_masks, filter_objects_by_size, get_centroids, crop_regions
from .config import MCDPipelineConfig
from .result import CellDetectionResult


class MCDPipeline:
    """Main MCD pipeline for cell detection in AS-OCT images."""
    
    def __init__(self, segmentor, threshold_method, classifier, 
                 preprocessing_pipeline=None, config=None):
        """Initialize pipeline with components and configuration."""
        self.segmentor = segmentor
        self.threshold_method = threshold_method
        self.classifier = classifier
        self.preprocessing = preprocessing_pipeline
        self.config = config or MCDPipelineConfig()
    
    def detect_cells(self, image_path: str) -> CellDetectionResult:
        """
        Detect cells in AS-OCT image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Cell detection result with locations and masks
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # Preprocessing
        if self.preprocessing:
            image = self.preprocessing.process(image)
        
        # Field-of-Focus: segment AC region
        chamber_mask = self.segmentor.segment(image)
        
        # Fine-grained Object Detection: generate candidates
        candidate_mask = self._generate_candidates(image, chamber_mask)
        candidate_locations = get_centroids(candidate_mask)
        
        # Classification: filter cells from noise
        cell_locations = self._classify_candidates(image_path, candidate_locations)
        
        # Create result
        result = CellDetectionResult(
            cell_locations=cell_locations,
            chamber_mask=chamber_mask,
            candidate_locations=candidate_locations
        )
        
        result.add_info('image_path', image_path)
        result.add_info('threshold_method', self.config.threshold.method)
        result.add_info('lambda_factor', self.config.threshold.lambda_factor)
        
        return result
    
    def _generate_candidates(self, image: np.ndarray, chamber_mask: np.ndarray) -> np.ndarray:
        """Generate candidate regions using MiRP algorithm."""
        # Apply thresholding
        threshold_mask = self.threshold_method.apply_threshold(
            image, 
            lambda_factor=self.config.threshold.lambda_factor
        )
        
        # Intersect with chamber mask
        merged_mask = intersect_masks(chamber_mask, threshold_mask)
        
        # Filter by size
        return filter_objects_by_size(
            merged_mask,
            min_size=self.config.threshold.min_size,
            max_size=self.config.threshold.max_size
        )
    
    def _classify_candidates(self, image_path: str, 
                           candidates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Classify candidate regions to identify cells."""
        if not candidates:
            return []
        
        # Crop regions around candidates
        cropped_regions = crop_regions(
            image_path, 
            candidates, 
            crop_size=self.config.classification.crop_size
        )
        
        # Batch classification
        predictions = self.classifier.classify_batch(cropped_regions)
        
        # Filter positive predictions
        return [loc for loc, pred in zip(candidates, predictions) if pred == 1]