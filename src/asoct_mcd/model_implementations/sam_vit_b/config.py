"""
Author: B. Chen

SAM ViT-B model configuration.
"""

from dataclasses import dataclass
from ...model_management import BaseModelConfig


@dataclass
class SAMViTBConfig(BaseModelConfig):
    """SAM ViT-B model configuration."""
    name: str = "sam_vit_b"
    checkpoint_url: str = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    local_filename: str = 'sam_vit_b_01ec64.pth'
    model_type: str = 'vit_b'