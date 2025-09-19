"""
Author: B. Chen

Spatial attention network configuration.
"""

from dataclasses import dataclass
from ...model_management import BaseModelConfig


@dataclass
class SpatialAttentionConfig(BaseModelConfig):
    """Spatial attention network configuration."""
    name: str = "spatial_attention"
    checkpoint_url: str = 'https://github.com/joeybyc/MCD/raw/main/models/spatial_attention_network.pth'
    local_filename: str = 'spatial_attention_network.pth'
    size_img: int = 20
    batch_size: int = 128