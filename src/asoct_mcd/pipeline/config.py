"""
Author: B. Chen

Configuration classes for MCD pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SegmentationConfig:
    """Segmentation configuration."""
    method: str = "zero_shot"
    model_name: str = "sam_vit_b"
    prompt_generator: str = "i2acp"
    offset_ratio: float = 0.02
    area_ratio_threshold: float = 0.65


@dataclass
class ThresholdConfig:
    """Threshold detection configuration."""
    method: str = "otsu"
    lambda_factor: float = 0.83
    min_size: int = 1
    max_size: int = 25


@dataclass
class ClassificationConfig:
    """Classification configuration."""
    model_name: str = "spatial_attention_network"
    crop_size: int = 20
    batch_size: int = 128


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    enable: bool = True
    denoising_method: str = "nlm"
    convert_to_grayscale: bool = True
    nlm_h: int = 10
    nlm_template_window: int = 7
    nlm_search_window: int = 21


@dataclass
class MCDPipelineConfig:
    """Complete MCD pipeline configuration."""
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    device: str = "auto"
    model_cache_dir: str = "./models"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MCDPipelineConfig':
        """Create configuration from dictionary."""
        return cls(
            segmentation=SegmentationConfig(**config_dict.get('segmentation', {})),
            threshold=ThresholdConfig(**config_dict.get('threshold', {})),
            classification=ClassificationConfig(**config_dict.get('classification', {})),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            device=config_dict.get('device', 'auto'),
            model_cache_dir=config_dict.get('model_cache_dir', './models')
        )