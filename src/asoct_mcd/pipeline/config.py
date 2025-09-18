"""
Author: B. Chen

Configuration for cell detection pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for MCD cell detection pipeline."""
    
    # Model selection
    sam_model: str = 'sam_vit_b'
    classifier_model: str = 'spatial_attention_network'
    
    # Threshold parameters
    threshold_method: str = 'otsu'  # 'otsu', 'isodata', 'mean'
    threshold_lambda: float = 0.83
    
    # Size filtering
    lower_bound: int = 1
    upper_bound: int = 25
    
    # Bounding box visualization
    box_size: int = 10
    
    # I2ACP parameters
    offset_ratio: float = 0.02
    area_ratio_threshold: float = 0.65
    
    # Device configuration
    device: str = 'auto'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.threshold_lambda <= 0:
            raise ValueError("threshold_lambda must be positive")
        if self.lower_bound < 1:
            raise ValueError("lower_bound must be >= 1")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be > lower_bound")
        if not 0 < self.offset_ratio < 1:
            raise ValueError("offset_ratio must be between 0 and 1")
        if not 0 < self.area_ratio_threshold < 1:
            raise ValueError("area_ratio_threshold must be between 0 and 1")