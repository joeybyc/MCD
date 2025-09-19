"""
Author: B. Chen

Implementation modules for segmentation operations.
"""

from .zero_shot import ZeroShotSegmentor
from .trained import TrainedSegmentor

__all__ = [
    'ZeroShotSegmentor',
    'TrainedSegmentor'
]