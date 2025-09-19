"""
Author: B. Chen

MCD Pipeline module for complete cell detection workflow.
Provides high-level interface for AS-OCT cell detection.
"""

# Auto-import model implementations to trigger registration
import asoct_mcd.model_implementations

from .builder import MCDPipelineBuilder
from .pipeline import MCDPipeline
from .config import MCDPipelineConfig
from .result import CellDetectionResult

__all__ = [
    'MCDPipelineBuilder',
    'MCDPipeline',
    'MCDPipelineConfig',
    'CellDetectionResult'
]