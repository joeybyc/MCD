"""
Author: B. Chen

Unit tests for threshold methods module.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the modules to test
from src.asoct_mcd.threshold_methods import ThresholdMethod, ThresholdFactory
from src.asoct_mcd.threshold_methods.implementations.thresholding import (
    OtsuThreshold, 
    IsodataThreshold, 
    MeanThreshold
)


class TestThresholdMethod:
    """Test abstract base class ThresholdMethod."""
    
    def test_validate_grayscale_valid_image(self):
        """Test grayscale validation with valid image."""
        # Create a concrete implementation for testing
        class MockThreshold(ThresholdMethod):
            def _calculate_threshold(self, image):
                return 128
        
        mock_threshold = MockThreshold()
        grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        # Should not raise exception
        mock_threshold._validate_grayscale(grayscale_image)
    
    def test_validate_grayscale_invalid_image(self):
        """Test grayscale validation with invalid image."""
        class MockThreshold(ThresholdMethod):
            def _calculate_threshold(self, image):
                return 128
        
        mock_threshold = MockThreshold()
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Input image must be grayscale"):
            mock_threshold._validate_grayscale(color_image)
    
    def test_apply_threshold_basic(self):
        """Test basic threshold application."""
        class MockThreshold(ThresholdMethod):
            def _calculate_threshold(self, image):
                return 128
        
        mock_threshold = MockThreshold()
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        mask = mock_threshold.apply_threshold(image)
        
        assert mask is not None
        assert mask.shape == image.shape
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 255))
    
    def test_apply_threshold_with_lambda_factor(self):
        """Test threshold application with lambda factor."""
        class MockThreshold(ThresholdMethod):
            def _calculate_threshold(self, image):
                return 100
        
        mock_threshold = MockThreshold()
        image = np.full((50, 50), 150, dtype=np.uint8)
        
        # With lambda_factor=0.5, threshold becomes 50, so all pixels (150) should be white
        mask = mock_threshold.apply_threshold(image, lambda_factor=0.5)
        assert np.all(mask == 255)
        
        # With lambda_factor=2.0, threshold becomes 200, so all pixels (150) should be black
        mask = mock_threshold.apply_threshold(image, lambda_factor=2.0)
        assert np.all(mask == 0)
    
    def test_apply_threshold_return_thresh(self):
        """Test threshold application returning threshold value."""
        class MockThreshold(ThresholdMethod):
            def _calculate_threshold(self, image):
                return 100
        
        mock_threshold = MockThreshold()
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        mask, thresh_value = mock_threshold.apply_threshold(
            image, lambda_factor=0.8, return_thresh=True
        )
        
        assert mask is not None
        assert thresh_value == 80.0  # 100 * 0.8


class TestOtsuThreshold:
    """Test Otsu threshold implementation."""
    
    def test_calculate_threshold(self):
        """Test Otsu threshold calculation."""
        # Create a simple bimodal image
        image = np.concatenate([
            np.full((50, 25), 50, dtype=np.uint8),
            np.full((50, 25), 200, dtype=np.uint8)
        ], axis=1)
        
        otsu = OtsuThreshold()
        threshold = otsu._calculate_threshold(image)
        
        # Otsu should find a threshold between the two modes (inclusive)
        assert 50 <= threshold <= 200
        assert isinstance(threshold, (int, float, np.number))
    
    def test_apply_threshold_integration(self):
        """Test full Otsu threshold application."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        otsu = OtsuThreshold()
        mask = otsu.apply_threshold(image)
        
        assert mask.shape == image.shape
        assert np.all((mask == 0) | (mask == 255))


class TestIsodataThreshold:
    """Test Isodata threshold implementation."""
    
    def test_calculate_threshold(self):
        """Test Isodata threshold calculation."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        isodata = IsodataThreshold()
        threshold = isodata._calculate_threshold(image)
        
        assert 0 <= threshold <= 255
        assert isinstance(threshold, float)
    
    def test_apply_threshold_integration(self):
        """Test full Isodata threshold application."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        isodata = IsodataThreshold()
        mask = isodata.apply_threshold(image)
        
        assert mask.shape == image.shape
        assert np.all((mask == 0) | (mask == 255))


class TestMeanThreshold:
    """Test Mean threshold implementation."""
    
    def test_calculate_threshold(self):
        """Test mean threshold calculation."""
        image = np.full((50, 50), 100, dtype=np.uint8)
        
        mean_threshold = MeanThreshold()
        threshold = mean_threshold._calculate_threshold(image)
        
        assert threshold == 100.0
    
    def test_calculate_threshold_mixed_values(self):
        """Test mean threshold with mixed pixel values."""
        image = np.array([[0, 255], [127, 128]], dtype=np.uint8)
        
        mean_threshold = MeanThreshold()
        threshold = mean_threshold._calculate_threshold(image)
        
        expected_mean = (0 + 255 + 127 + 128) / 4
        assert threshold == expected_mean


class TestThresholdFactory:
    """Test threshold factory functionality."""
    
    def test_create_threshold_otsu(self):
        """Test creating Otsu threshold via factory."""
        threshold = ThresholdFactory.create_threshold('otsu')
        assert isinstance(threshold, OtsuThreshold)
    
    def test_create_threshold_isodata(self):
        """Test creating Isodata threshold via factory."""
        threshold = ThresholdFactory.create_threshold('isodata')
        assert isinstance(threshold, IsodataThreshold)
    
    def test_create_threshold_mean(self):
        """Test creating Mean threshold via factory."""
        threshold = ThresholdFactory.create_threshold('mean')
        assert isinstance(threshold, MeanThreshold)
    
    def test_create_threshold_invalid_method(self):
        """Test creating threshold with invalid method type."""
        with pytest.raises(ValueError, match="Unknown threshold method"):
            ThresholdFactory.create_threshold('invalid_method')
    
    def test_get_available_methods(self):
        """Test getting list of available methods."""
        methods = ThresholdFactory.get_available_methods()
        
        assert isinstance(methods, list)
        assert 'otsu' in methods
        assert 'isodata' in methods
        assert 'mean' in methods
        assert len(methods) == 3
    
    def test_register_method(self):
        """Test registering new threshold method."""
        class CustomThreshold(ThresholdMethod):
            def _calculate_threshold(self, image):
                return 42
        
        # Register new method
        ThresholdFactory.register_method('custom', CustomThreshold)
        
        # Check it's available
        methods = ThresholdFactory.get_available_methods()
        assert 'custom' in methods
        
        # Create instance
        threshold = ThresholdFactory.create_threshold('custom')
        assert isinstance(threshold, CustomThreshold)
        
        # Clean up
        ThresholdFactory._methods.pop('custom', None)


class TestThresholdMethodsWithRealImages:
    """Test threshold methods with real image files."""
    
    @pytest.fixture
    def grayscale_image_path(self):
        """Path to grayscale test image."""
        return Path("tests/data/img1_grey.png")
    
    @pytest.fixture
    def color_image_path(self):
        """Path to color test image."""
        return Path("tests/data/img1.png")
    
    @pytest.fixture
    def grayscale_image(self, grayscale_image_path):
        """Load grayscale test image."""
        if not grayscale_image_path.exists():
            pytest.skip(f"Test image not found: {grayscale_image_path}")
        return cv2.imread(str(grayscale_image_path), cv2.IMREAD_GRAYSCALE)
    
    @pytest.fixture
    def color_image(self, color_image_path):
        """Load color test image."""
        if not color_image_path.exists():
            pytest.skip(f"Test image not found: {color_image_path}")
        return cv2.imread(str(color_image_path))
    
    def test_otsu_with_real_grayscale_image(self, grayscale_image):
        """Test Otsu threshold with real grayscale image."""
        otsu = OtsuThreshold()
        
        mask = otsu.apply_threshold(grayscale_image)
        
        assert mask.shape == grayscale_image.shape
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 255))
    
    def test_isodata_with_real_grayscale_image(self, grayscale_image):
        """Test Isodata threshold with real grayscale image."""
        isodata = IsodataThreshold()
        
        mask = isodata.apply_threshold(grayscale_image)
        
        assert mask.shape == grayscale_image.shape
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 255))
    
    def test_mean_with_real_grayscale_image(self, grayscale_image):
        """Test Mean threshold with real grayscale image."""
        mean_threshold = MeanThreshold()
        
        mask = mean_threshold.apply_threshold(grayscale_image)
        
        assert mask.shape == grayscale_image.shape
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 255))
    
    def test_lambda_factor_with_real_image(self, grayscale_image):
        """Test lambda factor functionality with real image."""
        otsu = OtsuThreshold()
        
        # Test with lambda factor 0.83 (from paper)
        mask, thresh = otsu.apply_threshold(
            grayscale_image, lambda_factor=0.83, return_thresh=True
        )
        
        assert mask.shape == grayscale_image.shape
        assert isinstance(thresh, float)
        assert thresh > 0
    
    def test_color_image_raises_error(self, color_image):
        """Test that color images raise appropriate error."""
        if len(color_image.shape) == 2:
            pytest.skip("Color image was loaded as grayscale")
            
        otsu = OtsuThreshold()
        
        with pytest.raises(ValueError, match="Input image must be grayscale"):
            otsu.apply_threshold(color_image)
    
    def test_all_methods_consistency(self, grayscale_image):
        """Test that all threshold methods work consistently."""
        methods = ['otsu', 'isodata', 'mean']
        
        for method_name in methods:
            method = ThresholdFactory.create_threshold(method_name)
            mask = method.apply_threshold(grayscale_image)
            
            # All methods should produce valid binary masks
            assert mask.shape == grayscale_image.shape
            assert mask.dtype == np.uint8
            assert np.all((mask == 0) | (mask == 255))


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_image(self):
        """Test with empty image array."""
        otsu = OtsuThreshold()
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0)
        
        with pytest.raises((ValueError, IndexError)):
            otsu.apply_threshold(empty_image)
    
    def test_single_pixel_image(self):
        """Test with single pixel image."""
        otsu = OtsuThreshold()
        single_pixel = np.array([[128]], dtype=np.uint8)
        
        mask = otsu.apply_threshold(single_pixel)
        assert mask.shape == (1, 1)
    
    def test_uniform_image(self):
        """Test with uniform intensity image."""
        mean_threshold = MeanThreshold()
        uniform_image = np.full((50, 50), 100, dtype=np.uint8)
        
        mask = mean_threshold.apply_threshold(uniform_image)
        
        # All pixels have same intensity, so result depends on threshold comparison
        assert mask.shape == uniform_image.shape
        assert np.all((mask == 0) | (mask == 255))
    
    def test_extreme_lambda_values(self):
        """Test with extreme lambda factor values."""
        otsu = OtsuThreshold()
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        # Very small lambda - should make most pixels white
        mask_small = otsu.apply_threshold(image, lambda_factor=0.001)
        assert np.all((mask_small == 0) | (mask_small == 255))
        
        # Very large lambda - should make most pixels black
        mask_large = otsu.apply_threshold(image, lambda_factor=1000)
        assert np.all((mask_large == 0) | (mask_large == 255))


if __name__ == "__main__":
    pytest.main([__file__])