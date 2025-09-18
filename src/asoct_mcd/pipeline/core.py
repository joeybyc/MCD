"""
Author: B. Chen

Core cell detection pipeline implementation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union

from .config import PipelineConfig
from .results import CellDetectionResult, BatchDetectionResult
from ..model_management import ModelRegistry, SAMModelAdapter, PyTorchClassificationAdapter
from ..prompt_generation import PromptGeneratorFactory
from ..segmentation import SegmentorFactory
from ..classification import ClassifierFactory  
from ..threshold_methods import ThresholdFactory
from ..image_processing import get_centroids, intersect_masks, filter_objects_by_size, crop_regions


class CellDetectionPipeline:
    """MCD cell detection pipeline."""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize pipeline with configuration."""
        self.config = config or PipelineConfig()
        self._setup_components()
    
    def _setup_components(self):
        """Setup pipeline components."""
        # Model adapters
        self.sam_adapter = SAMModelAdapter(self.config.sam_model)
        self.classifier_adapter = PyTorchClassificationAdapter(self.config.classifier_model)
        
        # Prompt generator
        self.prompt_generator = PromptGeneratorFactory.create_generator(
            'i2acp',
            offset_ratio=self.config.offset_ratio,
            area_ratio_threshold=self.config.area_ratio_threshold
        )
        
        # Segmentor
        class PromptWrapper:
            def __init__(self, generator):
                self.generator = generator
            def generate(self, image):
                return self.generator.generate(image)
        
        self.segmentor = SegmentorFactory.create_segmentor(
            'zero_shot',
            model_wrapper=self.sam_adapter,
            prompt_generator=PromptWrapper(self.prompt_generator)
        )
        
        # Classifier
        self.classifier = ClassifierFactory.create_classifier(
            'cell',
            model_wrapper=self.classifier_adapter
        )
        
        # Threshold method
        self.threshold_method = ThresholdFactory.create_threshold(self.config.threshold_method)
    
    def detect_cells(self, image_path: str, save_intermediates: bool = False) -> CellDetectionResult:
        """Detect cells in a single image."""
        image_path = str(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Step 1: Chamber segmentation
        chamber_mask = self.segmentor.segment(image)
        
        # Step 2: Threshold-based candidate detection
        threshold_mask = self.threshold_method.apply_threshold(
            image, 
            lambda_factor=self.config.threshold_lambda
        )
        
        # Step 3: Merge masks and filter by size
        merged_mask = intersect_masks(chamber_mask, threshold_mask)
        candidate_mask = filter_objects_by_size(
            merged_mask,
            min_size=self.config.lower_bound,
            max_size=self.config.upper_bound
        )
        
        # Step 4: Cell classification
        cell_locations = get_centroids(candidate_mask)
        if cell_locations:
            cropped_regions = crop_regions(image_path, cell_locations, self.config.box_size)
            predictions = self.classifier.classify_batch(cropped_regions)
            
            # Filter cells based on classification
            cell_locations = [loc for loc, pred in zip(cell_locations, predictions) if pred == 1]
        
        # Step 5: Create final cell mask
        cell_mask = np.zeros_like(image)
        if cell_locations:
            for x, y in cell_locations:
                cv2.circle(cell_mask, (int(x), int(y)), self.config.box_size//2, 255, -1)
        
        # Create result
        result = CellDetectionResult(
            image_path=image_path,
            chamber_mask=chamber_mask,
            cell_mask=cell_mask,
            cell_locations=cell_locations
        )
        
        if save_intermediates:
            points, _ = self.prompt_generator.generate(image)
            result.threshold_mask = threshold_mask
            result.candidate_mask = candidate_mask
            result.prompt_points = points
        
        return result
    
    def detect_batch(self, image_paths: List[str], save_intermediates: bool = False) -> BatchDetectionResult:
        """Detect cells in multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.detect_cells(image_path, save_intermediates)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
        
        return BatchDetectionResult(results=results)