"""
Author: B. Chen

Builder class for MCD pipeline construction.
"""

import yaml
from typing import Dict, Any

from ..model_management import ModelFactory
from ..segmentation import SegmentorFactory
from ..threshold_methods import ThresholdFactory
from ..classification import ClassifierFactory
from ..prompt_generation import PromptGeneratorFactory
from ..preprocessing import PreprocessingPipeline, DenoiserFactory, ConverterFactory

from .config import MCDPipelineConfig
from .wrappers import (
    ZeroShotSegmentationWrapper, TrainedSegmentationWrapper,
    ClassificationWrapper, PromptGenerationWrapper
)
from .pipeline import MCDPipeline


class MCDPipelineBuilder:
    """Builder for MCD pipeline with configuration support."""
    
    def __init__(self, config: MCDPipelineConfig = None):
        """Initialize builder with optional configuration."""
        self.config = config or MCDPipelineConfig()
        self._segmentor = None
        self._threshold_method = None
        self._classifier = None
        self._preprocessing = None
    
    def from_config(self, config: MCDPipelineConfig) -> 'MCDPipelineBuilder':
        """Set configuration and reset builder."""
        self.config = config
        return self.reset()
    
    def from_dict(self, config_dict: Dict[str, Any]) -> 'MCDPipelineBuilder':
        """Create from dictionary configuration."""
        self.config = MCDPipelineConfig.from_dict(config_dict)
        return self.reset()
    
    def from_yaml(self, yaml_path: str) -> 'MCDPipelineBuilder':
        """Create from YAML configuration file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return self.from_dict(config_dict)
    
    def reset(self) -> 'MCDPipelineBuilder':
        """Reset all components."""
        self._segmentor = None
        self._threshold_method = None
        self._classifier = None
        self._preprocessing = None
        return self
    
    def build(self) -> MCDPipeline:
        """Build configured pipeline."""
        # Auto-build components from config
        self._build_segmentor()
        self._build_threshold_method()
        self._build_classifier()
        if self.config.preprocessing.enable:
            self._build_preprocessing()
        
        # Validate components
        if not all([self._segmentor, self._threshold_method, self._classifier]):
            raise ValueError("Missing required pipeline components")
        
        return MCDPipeline(
            segmentor=self._segmentor,
            threshold_method=self._threshold_method,
            classifier=self._classifier,
            preprocessing_pipeline=self._preprocessing,
            config=self.config
        )
    
    def _build_segmentor(self):
        """Build segmentor from configuration."""
        seg_config = self.config.segmentation
        
        if seg_config.method == "zero_shot":
            # Create model wrapper
            model_wrapper = ModelFactory.create_model(seg_config.model_name)
            segmentation_wrapper = ZeroShotSegmentationWrapper(model_wrapper)
            
            # Create prompt generator
            prompt_gen = PromptGeneratorFactory.create_generator(
                seg_config.prompt_generator,
                offset_ratio=seg_config.offset_ratio,
                area_ratio_threshold=seg_config.area_ratio_threshold
            )
            prompt_wrapper = PromptGenerationWrapper(prompt_gen)
            
            self._segmentor = SegmentorFactory.create_segmentor(
                "zero_shot",
                model_wrapper=segmentation_wrapper,
                prompt_generator=prompt_wrapper
            )
        
        elif seg_config.method == "trained":
            model_wrapper = ModelFactory.create_model(seg_config.model_name)
            segmentation_wrapper = TrainedSegmentationWrapper(model_wrapper)
            
            self._segmentor = SegmentorFactory.create_segmentor(
                "trained",
                model_wrapper=segmentation_wrapper
            )
    
    def _build_threshold_method(self):
        """Build threshold method from configuration."""
        self._threshold_method = ThresholdFactory.create_threshold(
            self.config.threshold.method
        )
    
    def _build_classifier(self):
        """Build classifier from configuration."""
        model_wrapper = ModelFactory.create_model(self.config.classification.model_name)
        classification_wrapper = ClassificationWrapper(model_wrapper)
        
        self._classifier = ClassifierFactory.create_classifier(
            "cell",
            model_wrapper=classification_wrapper
        )
    
    def _build_preprocessing(self):
        """Build preprocessing pipeline from configuration."""
        prep_config = self.config.preprocessing
        pipeline = PreprocessingPipeline()
        
        if prep_config.convert_to_grayscale:
            converter = ConverterFactory.create_converter('grayscale')
            pipeline.add_converter(converter)
        
        if prep_config.denoising_method != "none":
            if prep_config.denoising_method == "nlm":
                denoiser = DenoiserFactory.create_denoiser(
                    'nlm',
                    h=prep_config.nlm_h,
                    template_window=prep_config.nlm_template_window,
                    search_window=prep_config.nlm_search_window
                )
            else:
                denoiser = DenoiserFactory.create_denoiser(prep_config.denoising_method)
            
            pipeline.add_denoiser(denoiser)
        
        self._preprocessing = pipeline