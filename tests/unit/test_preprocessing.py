"""
Unit tests for asoct_mcd.preprocessing module.
"""

import unittest
import numpy as np
import cv2
from pathlib import Path

from src.asoct_mcd.preprocessing import (
    ConverterFactory,
    DenoiserFactory,
    PreprocessingPipeline,
    ImageConverter,
    ImageDenoiser
)


class TestConverterFactory(unittest.TestCase):
    """Test cases for ConverterFactory."""

    def test_create_grayscale_converter(self):
        """Test creating grayscale converter."""
        converter = ConverterFactory.create_converter('grayscale')
        self.assertIsInstance(converter, ImageConverter)

    def test_create_unknown_converter(self):
        """Test creating unknown converter raises ValueError."""
        with self.assertRaises(ValueError) as context:
            ConverterFactory.create_converter('unknown')
        self.assertIn('Unknown converter', str(context.exception))

    def test_get_available_converters(self):
        """Test getting list of available converters."""
        available = ConverterFactory.get_available_converters()
        self.assertIn('grayscale', available)
        self.assertIsInstance(available, list)

    def test_register_converter(self):
        """Test registering custom converter."""
        class DummyConverter(ImageConverter):
            def convert(self, image):
                return image
        
        ConverterFactory.register_converter('dummy', DummyConverter)
        converter = ConverterFactory.create_converter('dummy')
        self.assertIsInstance(converter, DummyConverter)


class TestDenoiserFactory(unittest.TestCase):
    """Test cases for DenoiserFactory."""

    def test_create_nlm_denoiser(self):
        """Test creating NLM denoiser."""
        denoiser = DenoiserFactory.create_denoiser('nlm')
        self.assertIsInstance(denoiser, ImageDenoiser)

    def test_create_median_denoiser(self):
        """Test creating median denoiser."""
        denoiser = DenoiserFactory.create_denoiser('median')
        self.assertIsInstance(denoiser, ImageDenoiser)

    def test_create_nlm_with_params(self):
        """Test creating NLM denoiser with custom parameters."""
        denoiser = DenoiserFactory.create_denoiser('nlm', h=15, template_window=5)
        self.assertEqual(denoiser.h, 15)
        self.assertEqual(denoiser.template_window, 5)

    def test_create_median_with_params(self):
        """Test creating median denoiser with custom parameters."""
        denoiser = DenoiserFactory.create_denoiser('median', kernel_size=3)
        self.assertEqual(denoiser.kernel_size, 3)

    def test_median_even_kernel_size_raises_error(self):
        """Test that even kernel size raises ValueError for median denoiser."""
        with self.assertRaises(ValueError) as context:
            DenoiserFactory.create_denoiser('median', kernel_size=4)
        self.assertIn('odd', str(context.exception))

    def test_create_unknown_denoiser(self):
        """Test creating unknown denoiser raises ValueError."""
        with self.assertRaises(ValueError) as context:
            DenoiserFactory.create_denoiser('unknown')
        self.assertIn('Unknown denoiser', str(context.exception))

    def test_get_available_denoisers(self):
        """Test getting list of available denoisers."""
        available = DenoiserFactory.get_available_denoisers()
        self.assertIn('nlm', available)
        self.assertIn('median', available)


class TestGrayscaleConverter(unittest.TestCase):
    """Test cases for GrayscaleConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ConverterFactory.create_converter('grayscale')
        # Create test images
        self.color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    def test_convert_color_to_grayscale(self):
        """Test converting color image to grayscale."""
        result = self.converter.convert(self.color_image)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[:2], self.color_image.shape[:2])

    def test_convert_grayscale_unchanged(self):
        """Test that grayscale image remains unchanged."""
        result = self.converter.convert(self.gray_image)
        np.testing.assert_array_equal(result, self.gray_image)


class TestNLMDenoiser(unittest.TestCase):
    """Test cases for NLMDenoiser."""

    def setUp(self):
        """Set up test fixtures."""
        self.denoiser = DenoiserFactory.create_denoiser('nlm')
        self.gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        self.color_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    def test_denoise_grayscale_image(self):
        """Test denoising grayscale image."""
        result = self.denoiser.denoise(self.gray_image)
        self.assertEqual(result.shape, self.gray_image.shape)
        self.assertEqual(result.dtype, self.gray_image.dtype)

    def test_denoise_color_image_raises_error(self):
        """Test that denoising color image raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.denoiser.denoise(self.color_image)
        self.assertIn('grayscale', str(context.exception))

    def test_custom_parameters(self):
        """Test NLM denoiser with custom parameters."""
        custom_denoiser = DenoiserFactory.create_denoiser(
            'nlm', h=20, template_window=9, search_window=25
        )
        result = custom_denoiser.denoise(self.gray_image)
        self.assertEqual(result.shape, self.gray_image.shape)


class TestMedianDenoiser(unittest.TestCase):
    """Test cases for MedianDenoiser."""

    def setUp(self):
        """Set up test fixtures."""
        self.denoiser = DenoiserFactory.create_denoiser('median', kernel_size=5)
        self.gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        self.color_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    def test_denoise_grayscale_image(self):
        """Test denoising grayscale image."""
        result = self.denoiser.denoise(self.gray_image)
        self.assertEqual(result.shape, self.gray_image.shape)
        self.assertEqual(result.dtype, self.gray_image.dtype)

    def test_denoise_color_image_raises_error(self):
        """Test that denoising color image raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.denoiser.denoise(self.color_image)
        self.assertIn('grayscale', str(context.exception))


class TestPreprocessingPipeline(unittest.TestCase):
    """Test cases for PreprocessingPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = PreprocessingPipeline()
        self.color_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

    def test_empty_pipeline(self):
        """Test processing with empty pipeline."""
        result = self.pipeline.process(self.gray_image.copy())
        np.testing.assert_array_equal(result, self.gray_image)

    def test_add_converter(self):
        """Test adding converter to pipeline."""
        converter = ConverterFactory.create_converter('grayscale')
        self.pipeline.add_converter(converter)
        self.assertEqual(len(self.pipeline), 1)

    def test_add_denoiser(self):
        """Test adding denoiser to pipeline."""
        denoiser = DenoiserFactory.create_denoiser('nlm')
        self.pipeline.add_denoiser(denoiser)
        self.assertEqual(len(self.pipeline), 1)

    def test_full_pipeline(self):
        """Test complete pipeline with converter and denoiser."""
        converter = ConverterFactory.create_converter('grayscale')
        denoiser = DenoiserFactory.create_denoiser('median', kernel_size=3)
        
        self.pipeline.add_converter(converter).add_denoiser(denoiser)
        result = self.pipeline.process(self.color_image)
        
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[:2], self.color_image.shape[:2])

    def test_pipeline_chaining(self):
        """Test method chaining in pipeline."""
        converter = ConverterFactory.create_converter('grayscale')
        denoiser1 = DenoiserFactory.create_denoiser('median', kernel_size=3)
        denoiser2 = DenoiserFactory.create_denoiser('nlm')
        
        pipeline = (self.pipeline
                   .add_converter(converter)
                   .add_denoiser(denoiser1)
                   .add_denoiser(denoiser2))
        
        self.assertEqual(len(pipeline), 3)
        self.assertIs(pipeline, self.pipeline)

    def test_clear_pipeline(self):
        """Test clearing pipeline."""
        converter = ConverterFactory.create_converter('grayscale')
        self.pipeline.add_converter(converter)
        self.assertEqual(len(self.pipeline), 1)
        
        self.pipeline.clear()
        self.assertEqual(len(self.pipeline), 0)

    def test_grayscale_only_pipeline(self):
        """Test pipeline with grayscale image and denoiser only."""
        denoiser = DenoiserFactory.create_denoiser('nlm')
        self.pipeline.add_denoiser(denoiser)
        
        result = self.pipeline.process(self.gray_image)
        self.assertEqual(result.shape, self.gray_image.shape)


class TestWithRealImage(unittest.TestCase):
    """Tests with real AS-OCT image data."""

    @classmethod
    def setUpClass(cls):
        """Set up test data directory and sample image."""
        cls.test_data_dir = Path('tests/data')
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample AS-OCT image if it doesn't exist
        cls.sample_image_path = cls.test_data_dir / 'sample_asoct.png'
        if not cls.sample_image_path.exists():
            cls._create_sample_image(cls.sample_image_path)

    @classmethod
    def _create_sample_image(cls, image_path):
        """Create a sample AS-OCT image for testing."""
        # Create realistic AS-OCT image structure
        image = np.zeros((400, 600), dtype=np.uint8)
        
        # Add corneal surface (bright arc at top)
        center_x, center_y = 300, 200
        cv2.ellipse(image, (center_x, center_y - 100), (250, 100), 0, 0, 180, 200, 3)
        
        # Add iris structure (bright regions at bottom)
        cv2.ellipse(image, (center_x, center_y + 150), (200, 50), 0, 0, 180, 180, 2)
        
        # Add some bright spots simulating cells/particles
        for _ in range(10):
            x = np.random.randint(100, 500)
            y = np.random.randint(150, 250)
            cv2.circle(image, (x, y), 2, 255, -1)
        
        # Add realistic noise
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(image_path), image)

    def test_load_real_image(self):
        """Test loading real AS-OCT image."""
        image = cv2.imread(str(self.sample_image_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 2)
        self.assertEqual(image.dtype, np.uint8)

    def test_nlm_denoising_on_real_image(self):
        """Test NLM denoising on real AS-OCT image."""
        image = cv2.imread(str(self.sample_image_path), cv2.IMREAD_GRAYSCALE)
        denoiser = DenoiserFactory.create_denoiser('nlm', h=10, template_window=7, search_window=21)
        
        result = denoiser.denoise(image)
        
        # Verify output properties
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)
        
        # Denoised image should be different from original
        self.assertFalse(np.array_equal(result, image))
        
        # Check that denoising reduces noise (variance should be lower)
        original_var = np.var(image)
        denoised_var = np.var(result)
        self.assertLess(denoised_var, original_var)

    def test_median_denoising_on_real_image(self):
        """Test median denoising on real AS-OCT image."""
        image = cv2.imread(str(self.sample_image_path), cv2.IMREAD_GRAYSCALE)
        denoiser = DenoiserFactory.create_denoiser('median', kernel_size=3)
        
        result = denoiser.denoise(image)
        
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)

    def test_complete_pipeline_on_real_image(self):
        """Test complete preprocessing pipeline on real AS-OCT image."""
        # Load as color image first
        color_image = cv2.imread(str(self.sample_image_path), cv2.IMREAD_COLOR)
        if color_image is None:
            # If image is grayscale, convert to 3-channel for testing
            gray_img = cv2.imread(str(self.sample_image_path), cv2.IMREAD_GRAYSCALE)
            color_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        
        # Create pipeline
        pipeline = PreprocessingPipeline()
        converter = ConverterFactory.create_converter('grayscale')
        denoiser = DenoiserFactory.create_denoiser('nlm', h=10)
        
        pipeline.add_converter(converter).add_denoiser(denoiser)
        
        # Process the image
        result = pipeline.process(color_image)
        
        # Verify output is grayscale
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.dtype, np.uint8)

    def test_pipeline_preserves_image_features(self):
        """Test that preprocessing preserves important image features."""
        image = cv2.imread(str(self.sample_image_path), cv2.IMREAD_GRAYSCALE)
        
        # Create conservative denoising pipeline
        pipeline = PreprocessingPipeline()
        denoiser = DenoiserFactory.create_denoiser('median', kernel_size=3)
        pipeline.add_denoiser(denoiser)
        
        result = pipeline.process(image)
        
        # Check that bright structures are preserved
        # Find bright regions (above threshold)
        threshold = np.percentile(image, 90)
        original_bright = np.sum(image > threshold)
        result_bright = np.sum(result > threshold)
        
        # Should preserve most bright regions
        preservation_ratio = result_bright / original_bright
        self.assertGreater(preservation_ratio, 0.7)  # At least 70% preserved


class TestIntegrationWithSyntheticImage(unittest.TestCase):
    """Integration tests with synthetic AS-OCT image data."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic AS-OCT-like image for testing
        self.test_image = self._create_synthetic_asoct_image()
        
    def _create_synthetic_asoct_image(self):
        """Create a synthetic AS-OCT-like image for testing."""
        # Create a curved structure resembling anterior chamber
        image = np.zeros((400, 600), dtype=np.uint8)
        
        # Add curved bright regions for cornea and iris
        center_x, center_y = 300, 200
        cv2.ellipse(image, (center_x, center_y - 100), (250, 100), 0, 0, 180, 200, 2)
        cv2.ellipse(image, (center_x, center_y + 150), (200, 50), 0, 0, 180, 150, 2)
        
        # Add noise
        noise = np.random.normal(0, 20, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image

    def test_complete_preprocessing_workflow(self):
        """Test complete preprocessing workflow on synthetic AS-OCT image."""
        # Create pipeline
        pipeline = PreprocessingPipeline()
        converter = ConverterFactory.create_converter('grayscale')
        denoiser = DenoiserFactory.create_denoiser('nlm', h=10)
        
        pipeline.add_converter(converter).add_denoiser(denoiser)
        
        # Process the image
        result = pipeline.process(self.test_image)
        
        # Verify output properties
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        
        # Verify denoising effect (result should be different from original)
        self.assertFalse(np.array_equal(result, self.test_image))

    def test_preprocessing_preserves_structure(self):
        """Test that preprocessing preserves image structure."""
        pipeline = PreprocessingPipeline()
        denoiser = DenoiserFactory.create_denoiser('median', kernel_size=3)
        pipeline.add_denoiser(denoiser)
        
        result = pipeline.process(self.test_image)
        
        # Check that bright regions are preserved
        original_mean = np.mean(self.test_image)
        result_mean = np.mean(result)
        
        # Mean intensity should be similar (within reasonable range)
        self.assertLess(abs(original_mean - result_mean), 50)


if __name__ == '__main__':
    unittest.main()