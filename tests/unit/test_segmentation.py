"""
Author: B. Chen

Unit tests for segmentation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.asoct_mcd.segmentation import (
    BaseSegmentor, 
    ZeroShotModelWrapper, 
    TrainedModelWrapper,
    PromptGeneratorWrapper,
    SegmentorFactory
)
from src.asoct_mcd.segmentation.implementations.zero_shot import ZeroShotSegmentor
from src.asoct_mcd.segmentation.implementations.trained import TrainedSegmentor


class TestProtocols:
    """Test Protocol definitions and type checking."""
    
    def test_zero_shot_model_wrapper_protocol(self):
        """Test ZeroShotModelWrapper protocol implementation."""
        mock_wrapper = Mock(spec=ZeroShotModelWrapper)
        
        # Mock the protocol method
        expected_mask = np.zeros((100, 100), dtype=np.uint8) * 255
        mock_wrapper.get_mask.return_value = expected_mask
        
        # Test protocol compliance
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        prompts = {"points": np.array([[50, 50]]), "labels": np.array([1])}
        
        result = mock_wrapper.get_mask(image, prompts)
        
        mock_wrapper.get_mask.assert_called_once_with(image, prompts)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected_mask)
    
    def test_trained_model_wrapper_protocol(self):
        """Test TrainedModelWrapper protocol implementation."""
        mock_wrapper = Mock(spec=TrainedModelWrapper)
        
        expected_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_wrapper.get_mask.return_value = expected_mask
        
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = mock_wrapper.get_mask(image)
        
        mock_wrapper.get_mask.assert_called_once_with(image)
        np.testing.assert_array_equal(result, expected_mask)
    
    def test_prompt_generator_wrapper_protocol(self):
        """Test PromptGeneratorWrapper protocol implementation."""
        mock_generator = Mock(spec=PromptGeneratorWrapper)
        
        expected_prompts = {"points": np.array([[25, 25], [75, 75]]), "labels": np.array([1, 1])}
        mock_generator.generate.return_value = expected_prompts
        
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = mock_generator.generate(image)
        
        mock_generator.generate.assert_called_once_with(image)
        assert result == expected_prompts


class TestZeroShotSegmentor:
    """Test ZeroShotSegmentor implementation."""
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create mock zero-shot model wrapper."""
        wrapper = Mock(spec=ZeroShotModelWrapper)
        return wrapper
    
    @pytest.fixture
    def mock_prompt_generator(self):
        """Create mock prompt generator."""
        generator = Mock(spec=PromptGeneratorWrapper)
        return generator
    
    @pytest.fixture
    def zero_shot_segmentor(self, mock_model_wrapper, mock_prompt_generator):
        """Create ZeroShotSegmentor instance with mocked dependencies."""
        return ZeroShotSegmentor(mock_model_wrapper, mock_prompt_generator)
    
    def test_init(self, mock_model_wrapper, mock_prompt_generator):
        """Test ZeroShotSegmentor initialization."""
        segmentor = ZeroShotSegmentor(mock_model_wrapper, mock_prompt_generator)
        
        assert segmentor.model_wrapper is mock_model_wrapper
        assert segmentor.prompt_generator is mock_prompt_generator
        assert isinstance(segmentor, BaseSegmentor)
    
    def test_segment_success(self, zero_shot_segmentor, mock_model_wrapper, mock_prompt_generator):
        """Test successful segmentation workflow."""
        # Setup test data
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_prompts = {"points": np.array([[50, 50]]), "labels": np.array([1])}
        expected_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Setup mocks
        mock_prompt_generator.generate.return_value = test_prompts
        mock_model_wrapper.get_mask.return_value = expected_mask
        
        # Execute
        result = zero_shot_segmentor.segment(test_image)
        
        # Verify
        mock_prompt_generator.generate.assert_called_once_with(test_image)
        mock_model_wrapper.get_mask.assert_called_once_with(
            image=test_image,
            prompts=test_prompts
        )
        np.testing.assert_array_equal(result, expected_mask)
        assert result.dtype == np.uint8
    
    def test_segment_with_different_prompt_formats(self, zero_shot_segmentor, 
                                                  mock_model_wrapper, mock_prompt_generator):
        """Test segmentation with various prompt formats."""
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        expected_mask = np.zeros((50, 50), dtype=np.uint8)
        
        # Test different prompt formats
        prompt_formats = [
            np.array([[25, 25], [30, 30]]),  # Simple array
            {"boxes": np.array([10, 10, 40, 40])},  # Dict format
            ("text_prompt", "segment the object"),  # Tuple format
        ]
        
        for prompts in prompt_formats:
            mock_prompt_generator.generate.return_value = prompts
            mock_model_wrapper.get_mask.return_value = expected_mask
            
            result = zero_shot_segmentor.segment(test_image)
            
            mock_model_wrapper.get_mask.assert_called_with(
                image=test_image,
                prompts=prompts
            )
            np.testing.assert_array_equal(result, expected_mask)


class TestTrainedSegmentor:
    """Test TrainedSegmentor implementation."""
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create mock trained model wrapper."""
        wrapper = Mock(spec=TrainedModelWrapper)
        return wrapper
    
    @pytest.fixture
    def trained_segmentor(self, mock_model_wrapper):
        """Create TrainedSegmentor instance with mocked dependencies."""
        return TrainedSegmentor(mock_model_wrapper)
    
    def test_init(self, mock_model_wrapper):
        """Test TrainedSegmentor initialization."""
        segmentor = TrainedSegmentor(mock_model_wrapper)
        
        assert segmentor.model_wrapper is mock_model_wrapper
        assert isinstance(segmentor, BaseSegmentor)
    
    def test_segment_success(self, trained_segmentor, mock_model_wrapper):
        """Test successful segmentation with trained model."""
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        expected_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        mock_model_wrapper.get_mask.return_value = expected_mask
        
        result = trained_segmentor.segment(test_image)
        
        mock_model_wrapper.get_mask.assert_called_once_with(test_image)
        np.testing.assert_array_equal(result, expected_mask)
        assert result.dtype == np.uint8


class TestSegmentorFactory:
    """Test SegmentorFactory functionality."""
    
    def test_get_available_segmentors(self):
        """Test getting list of available segmentors."""
        available = SegmentorFactory.get_available_segmentors()
        
        assert isinstance(available, list)
        assert 'zero_shot' in available
        assert 'trained' in available
    
    def test_create_zero_shot_segmentor_success(self):
        """Test successful zero-shot segmentor creation."""
        mock_model = Mock(spec=ZeroShotModelWrapper)
        mock_prompt = Mock(spec=PromptGeneratorWrapper)
        
        segmentor = SegmentorFactory.create_segmentor(
            'zero_shot',
            model_wrapper=mock_model,
            prompt_generator=mock_prompt
        )
        
        assert isinstance(segmentor, ZeroShotSegmentor)
        assert segmentor.model_wrapper is mock_model
        assert segmentor.prompt_generator is mock_prompt
    
    def test_create_trained_segmentor_success(self):
        """Test successful trained segmentor creation."""
        mock_model = Mock(spec=TrainedModelWrapper)
        
        segmentor = SegmentorFactory.create_segmentor(
            'trained',
            model_wrapper=mock_model
        )
        
        assert isinstance(segmentor, TrainedSegmentor)
        assert segmentor.model_wrapper is mock_model
    
    def test_create_segmentor_unknown_type(self):
        """Test error handling for unknown segmentor type."""
        with pytest.raises(ValueError, match="Unknown segmentor: invalid"):
            SegmentorFactory.create_segmentor('invalid')
    
    def test_create_zero_shot_missing_dependencies(self):
        """Test error handling when zero-shot dependencies are missing."""
        mock_model = Mock(spec=ZeroShotModelWrapper)
        
        # Missing prompt_generator
        with pytest.raises(ValueError, match="Invalid dependencies"):
            SegmentorFactory.create_segmentor(
                'zero_shot',
                model_wrapper=mock_model
            )
    
    def test_create_trained_missing_dependencies(self):
        """Test error handling when trained dependencies are missing."""
        # Missing model_wrapper
        with pytest.raises(ValueError, match="Invalid dependencies"):
            SegmentorFactory.create_segmentor('trained')
    
    def test_register_new_segmentor(self):
        """Test registering a custom segmentor type."""
        class CustomSegmentor(BaseSegmentor):
            def __init__(self, custom_param):
                self.custom_param = custom_param
            
            def segment(self, image):
                return np.zeros_like(image)
        
        # Register custom segmentor
        SegmentorFactory.register_segmentor('custom', CustomSegmentor)
        
        # Check it's available
        available = SegmentorFactory.get_available_segmentors()
        assert 'custom' in available
        
        # Create instance
        segmentor = SegmentorFactory.create_segmentor(
            'custom',
            custom_param="test"
        )
        assert isinstance(segmentor, CustomSegmentor)
        assert segmentor.custom_param == "test"


class TestIntegration:
    """Integration tests for complete segmentation workflow."""
    
    def test_end_to_end_zero_shot_workflow(self):
        """Test complete zero-shot segmentation workflow."""
        # Create real implementations with mocked internals
        model_wrapper = Mock(spec=ZeroShotModelWrapper)
        prompt_generator = Mock(spec=PromptGeneratorWrapper)
        
        # Setup test data
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_prompts = {"points": np.array([[50, 50]]), "labels": np.array([1])}
        expected_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Setup mocks
        prompt_generator.generate.return_value = test_prompts
        model_wrapper.get_mask.return_value = expected_mask
        
        # Create segmentor via factory
        segmentor = SegmentorFactory.create_segmentor(
            'zero_shot',
            model_wrapper=model_wrapper,
            prompt_generator=prompt_generator
        )
        
        # Execute workflow
        result = segmentor.segment(test_image)
        
        # Verify complete workflow
        prompt_generator.generate.assert_called_once_with(test_image)
        model_wrapper.get_mask.assert_called_once_with(
            image=test_image,
            prompts=test_prompts
        )
        np.testing.assert_array_equal(result, expected_mask)
    
    def test_end_to_end_trained_workflow(self):
        """Test complete trained model segmentation workflow."""
        model_wrapper = Mock(spec=TrainedModelWrapper)
        
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        expected_mask = np.zeros((100, 100), dtype=np.uint8)
        
        model_wrapper.get_mask.return_value = expected_mask
        
        segmentor = SegmentorFactory.create_segmentor(
            'trained',
            model_wrapper=model_wrapper
        )
        
        result = segmentor.segment(test_image)
        
        model_wrapper.get_mask.assert_called_once_with(test_image)
        np.testing.assert_array_equal(result, expected_mask)