"""
Author: B. Chen

Configuration classes for model management.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    
    name: str
    model_type: str  # 'sam', 'pytorch_classification'
    checkpoint_url: str
    local_path: str
    device: str = 'auto'
    load_on_startup: bool = False
    cache_enabled: bool = True
    model_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if not self.model_type:
            raise ValueError("Model type cannot be empty")
        if not self.checkpoint_url:
            raise ValueError("Checkpoint URL cannot be empty")
        if not self.local_path:
            raise ValueError("Local path cannot be empty")


# Predefined model configurations
DEFAULT_MODELS = {
    'sam_vit_b': ModelConfig(
        name='sam_vit_b',
        model_type='sam',
        checkpoint_url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        local_path='models/sam_vit_b_01ec64.pth',
        device='auto',
        load_on_startup=False,
        model_params={'model_type': 'vit_b'}
    ),
    
    'spatial_attention_network': ModelConfig(
        name='spatial_attention_network',
        model_type='pytorch_classification',
        checkpoint_url='https://github.com/joeybyc/MCD/raw/main/models/spatial_attention_network.pth',
        local_path='models/spatial_attention_network.pth',
        device='auto',
        load_on_startup=False,
        model_params={'size_img': 20, 'batch_size': 128}
    )
}