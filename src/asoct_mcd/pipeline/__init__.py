"""
Author: B. Chen

Cell detection pipeline module for AS-OCT image processing.
Provides high-level pipeline for complete cell detection workflow.
"""

from .config import PipelineConfig
from .results import CellDetectionResult, BatchDetectionResult
from .core import CellDetectionPipeline
from .builder import PipelineBuilder, create_default_pipeline, create_custom_pipeline

__all__ = [
    # Configuration
    'PipelineConfig',
    
    # Results
    'CellDetectionResult',
    'BatchDetectionResult',
    
    # Core pipeline
    'CellDetectionPipeline',
    
    # Builder pattern
    'PipelineBuilder',
    'create_default_pipeline',
    'create_custom_pipeline'
]