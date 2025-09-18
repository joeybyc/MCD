import cv2
import sys
import os

# Add src to Python path to import your package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from asoct_mcd.preprocessing import ConverterFactory, DenoiserFactory, PreprocessingPipeline

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with sample AS-OCT image."""
    
    # Construct correct path to test data
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, 'data\sample_asoct.png')
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Test image not found at: {image_path}")
    
    print(f"Original image shape: {image.shape}")  # (height, width, channels)
    
    # Create and execute processing pipeline
    pipeline = (PreprocessingPipeline()
               .add_converter(ConverterFactory.create_converter('grayscale'))
               .add_denoiser(DenoiserFactory.create_denoiser('nlm', h=10)))
    
    result = pipeline.process(image)
    
    # Display shape transformation
    print(f"After grayscale conversion: {result.shape}")  # (height, width)
    print(f"Final processed image shape: {result.shape}")
    print(f"Number of pipeline steps: {len(pipeline)}")
    
    # Save result for verification
    output_path = os.path.join(current_dir, 'data\processed_sample.png')
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to: {output_path}")
    
    return result

if __name__ == "__main__":
    test_preprocessing_pipeline()