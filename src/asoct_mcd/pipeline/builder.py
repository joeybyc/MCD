"""
Author: B. Chen

Builder pattern for pipeline configuration.
"""

from .config import PipelineConfig
from .core import CellDetectionPipeline


class PipelineBuilder:
    """Builder for cell detection pipeline with fluent interface."""
    
    def __init__(self):
        """Initialize builder with default config."""
        self.config = PipelineConfig()
    
    def with_threshold_method(self, method: str) -> 'PipelineBuilder':
        """Set threshold method."""
        self.config.threshold_method = method
        return self
    
    def with_threshold_lambda(self, lambda_val: float) -> 'PipelineBuilder':
        """Set threshold lambda factor."""
        self.config.threshold_lambda = lambda_val
        return self
    
    def with_size_bounds(self, lower: int, upper: int) -> 'PipelineBuilder':
        """Set object size filtering bounds."""
        self.config.lower_bound = lower
        self.config.upper_bound = upper
        return self
    
    def with_models(self, sam_model: str = None, classifier_model: str = None) -> 'PipelineBuilder':
        """Set model names."""
        if sam_model:
            self.config.sam_model = sam_model
        if classifier_model:
            self.config.classifier_model = classifier_model
        return self
    
    def with_device(self, device: str) -> 'PipelineBuilder':
        """Set computation device."""
        self.config.device = device
        return self
    
    def build(self) -> CellDetectionPipeline:
        """Build the configured pipeline."""
        return CellDetectionPipeline(self.config)


# Convenience function
def create_default_pipeline() -> CellDetectionPipeline:
    """Create pipeline with default configuration."""
    return PipelineBuilder().build()


def create_custom_pipeline(**kwargs) -> CellDetectionPipeline:
    """Create pipeline with custom parameters."""
    config = PipelineConfig(**kwargs)
    return CellDetectionPipeline(config)