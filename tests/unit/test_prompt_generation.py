"""
Author: B. Chen

Unit tests for prompt generation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.asoct_mcd.prompt_generation.interfaces import BasePromptGenerator, PointPromptGenerator
from src.asoct_mcd.prompt_generation.implementations.i2acp import I2ACPGenerator
from src.asoct_mcd.prompt_generation.factories import PromptGeneratorFactory


class TestBasePromptGenerator:
    """Test cases for BasePromptGenerator interface."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BasePromptGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePromptGenerator()


class TestPointPromptGenerator:
    """Test cases for PointPromptGenerator interface."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that PointPromptGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PointPromptGenerator()


class TestI2ACPGenerator:
    """Test cases for I2ACPGenerator implementation."""
    
    @pytest.fixture
    def generator(self):
        """Create I2ACPGenerator instance for testing."""
        return I2ACPGenerator(offset_ratio=0.1, area_ratio_threshold=0.65)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample grayscale image for testing."""
        # Create a 100x100 image with a bright central region (simulating anterior segment)
        image = np.zeros((100, 100), dtype=np.uint8)
        image[30:70, 30:70] = 200  # Bright central region
        return image
    
    def test_initialization(self):
        """Test I2ACPGenerator initialization with custom parameters."""
        generator = I2ACPGenerator(offset_ratio=0.15, area_ratio_threshold=0.7)
        assert generator.offset_ratio == 0.15
        assert generator.area_ratio_threshold == 0.7
    
    def test_initialization_defaults(self):
        """Test I2ACPGenerator initialization with default parameters."""
        generator = I2ACPGenerator()
        assert generator.offset_ratio == 0.1
        assert generator.area_ratio_threshold == 0.65
    
    def test_generate_with_valid_image(self, generator, sample_image):
        """Test prompt generation with valid grayscale image."""
        points, labels = generator.generate(sample_image)
        
        # Check output format
        assert isinstance(points, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert points.shape == (2, 2)  # 2 points, each with (x, y)
        assert labels.shape == (2,)
        assert np.all(labels == 1)  # All positive prompts
    
    def test_generate_with_invalid_dimensions(self, generator):
        """Test that images with invalid dimensions raise ValueError."""
        invalid_image = np.zeros((100,), dtype=np.uint8)  # 1D array
        with pytest.raises(ValueError, match="Input image must be 2D or 3D array"):
            generator.generate(invalid_image)

    def test_generate_with_color_image(self, generator):
        """Test that color images are automatically converted to grayscale."""
        color_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        points, labels = generator.generate(color_image)
        
        assert isinstance(points, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert points.shape[1] == 2
        assert len(labels) == len(points)
    
    def test_generate_prompt_points_logic(self, generator):
        """Test prompt point generation logic."""
        image = np.ones((100, 200), dtype=np.uint8) * 128  # 100x200 image
        centroid_x, centroid_y = 100, 50
        
        points, labels = generator._generate_prompt_points(image, centroid_x, centroid_y)
        
        expected_offset = int(200 * 0.1)  # 20 pixels
        expected_points = np.array([[120, 50], [80, 50]])  # centroid ± offset
        
        np.testing.assert_array_equal(points, expected_points)
        np.testing.assert_array_equal(labels, np.array([1, 1]))
    
    def test_get_centroid_valid_mask(self, generator):
        """Test centroid calculation with valid mask."""
        # Create mask with white region from (20,20) to (40,40)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:41, 20:41] = 255
        
        centroid_x, centroid_y = generator._get_centroid(mask)
        
        # Expected centroid is at center of the region
        assert centroid_x == 30
        assert centroid_y == 30
    
    def test_get_centroid_empty_mask(self, generator):
        """Test that empty mask falls back to image center."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        
        centroid_x, centroid_y = generator._get_centroid(empty_mask)
        
        # Should return image center as fallback
        assert centroid_x == 50  # 100 // 2
        assert centroid_y == 50  # 100 // 2
    
    def test_find_anterior_segment_single_component(self, generator):
        """Test anterior segment finding with single large component."""
        # Create binary mask with one large component
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[30:70, 30:70] = 255
        
        result = generator._find_anterior_segment(binary_mask)
        
        np.testing.assert_array_equal(result, binary_mask)
    
    def test_find_anterior_segment_merge_components(self, generator):
        """Test anterior segment finding with two components that should merge."""
        # Create binary mask with two components of similar size
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[20:40, 20:40] = 255  # Component 1: 400 pixels
        binary_mask[60:75, 60:75] = 255  # Component 2: 225 pixels (ratio = 0.56 < 0.65)
        
        # Set area ratio threshold to 0.5 to ensure merging
        generator.area_ratio_threshold = 0.5
        result = generator._find_anterior_segment(binary_mask)
        
        # Both components should be present
        assert np.sum(result[20:40, 20:40] == 255) == 400
        assert np.sum(result[60:75, 60:75] == 255) == 225
    
    def test_find_anterior_segment_no_merge(self, generator):
        """Test anterior segment finding with components that shouldn't merge."""
        # Create binary mask with one large and one very small component
        binary_mask = np.zeros((100, 100), dtype=np.uint8)
        binary_mask[20:60, 20:60] = 255  # Large component
        binary_mask[70:75, 70:75] = 255  # Small component (ratio << 0.65)
        
        result = generator._find_anterior_segment(binary_mask)
        
        # Only large component should remain
        assert np.sum(result[20:60, 20:60] == 255) == 1600
        assert np.sum(result[70:75, 70:75] == 255) == 0


class TestPromptGeneratorFactory:
    """Test cases for PromptGeneratorFactory."""
    
    def test_create_i2acp_generator(self):
        """Test creating I2ACP generator through factory."""
        generator = PromptGeneratorFactory.create_generator('i2acp')
        
        assert isinstance(generator, I2ACPGenerator)
        assert isinstance(generator, PointPromptGenerator)
    
    def test_create_generator_with_kwargs(self):
        """Test creating generator with custom parameters."""
        generator = PromptGeneratorFactory.create_generator(
            'i2acp', 
            offset_ratio=0.15, 
            area_ratio_threshold=0.7
        )
        
        assert generator.offset_ratio == 0.15
        assert generator.area_ratio_threshold == 0.7
    
    def test_create_unknown_generator(self):
        """Test that unknown generator type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown generator: unknown"):
            PromptGeneratorFactory.create_generator('unknown')
    
    def test_register_generator(self):
        """Test registering new generator type."""
        # Create mock generator class
        MockGenerator = Mock(spec=BasePromptGenerator)
        
        PromptGeneratorFactory.register_generator('mock', MockGenerator)
        
        assert 'mock' in PromptGeneratorFactory.get_available_generators()
        
        # Clean up
        del PromptGeneratorFactory._generators['mock']
    
    def test_get_available_generators(self):
        """Test getting list of available generators."""
        available = PromptGeneratorFactory.get_available_generators()
        
        assert isinstance(available, list)
        assert 'i2acp' in available
    
    def test_factory_error_message_includes_available_types(self):
        """Test that error message shows available generator types."""
        with pytest.raises(ValueError) as exc_info:
            PromptGeneratorFactory.create_generator('nonexistent')
        
        error_message = str(exc_info.value)
        assert 'Available: i2acp' in error_message


class TestIntegration:
    """Integration tests for prompt generation module."""
    
    def test_factory_created_generator_produces_valid_output(self):
        """Test that factory-created generator produces valid output."""
        generator = PromptGeneratorFactory.create_generator('i2acp')
        
        # Create test image
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 200
        
        points, labels = generator.generate(image)
        
        # Verify output format and content
        assert points.shape == (2, 2)
        assert labels.shape == (2,)
        assert np.all(labels == 1)
        assert points.dtype in [np.int32, np.int64]  # Integer coordinates
    
    @patch('src.asoct_mcd.prompt_generation.implementations.i2acp.DenoiserFactory')
    def test_denoiser_integration(self, mock_factory):
        """Test integration with denoiser factory."""
        mock_denoiser = Mock()
        mock_denoiser.denoise.return_value = np.ones((100, 100), dtype=np.uint8) * 128
        mock_factory.create_denoiser.return_value = mock_denoiser
        
        generator = I2ACPGenerator()
        image = np.zeros((100, 100), dtype=np.uint8)
        
        # This should not raise an error despite zero image
        # because denoiser returns uniform image
        points, labels = generator.generate(image)
        
        mock_factory.create_denoiser.assert_called_once_with('nlm')
        mock_denoiser.denoise.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])