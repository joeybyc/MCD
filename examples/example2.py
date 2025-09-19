"""
Simple cell detection example.
Usage: python example2.py image1.png
"""

import sys
import os
from pathlib import Path


from asoct_mcd.pipeline import MCDPipelineBuilder
from asoct_mcd.image_processing import draw_rectangles


def detect_cells(image_path):
    """
    Detect cells in AS-OCT image and save result.
    
    Args:
        image_path: Path to input image
    """
    
    print(f"Processing image: {image_path}")
    
    # Create MCD pipeline
    pipeline = MCDPipelineBuilder().build()
    
    # Detect cells
    result = pipeline.detect_cells(image_path)
    
    # Print results
    print(f"\nDetection Results:")
    print(f"Number of cells detected: {result.cell_count}")
    print(f"Cell locations (x, y): {result.cell_locations}")
    
    # Save visualization
    image_name = Path(image_path).stem
    output_path = f"{image_name}_mcd_detection.jpg"
    
    # Draw green boxes around detected cells
    detection_image = draw_rectangles(
        image_path, 
        result.cell_locations, 
        box_size=10, 
        color="#00FF00"  # Green
    )
    
    # Convert to RGB if needed (JPEG doesn't support RGBA)
    if detection_image.mode == 'RGBA':
        detection_image = detection_image.convert('RGB')
    
    # Save as JPEG
    detection_image.save(output_path, "JPEG", quality=95)
    print(f"Detection result saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_detection.py <image_path>")
        print("Example: python simple_detection.py tests/data/image1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        detect_cells(image_path)
    except Exception as e:
        print(f"Error during detection: {e}")
        sys.exit(1)