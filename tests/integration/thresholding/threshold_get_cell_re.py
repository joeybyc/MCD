"""
Integration test using refactored code for threshold-based cell detection.
Tests the pipeline: grayscale -> otsu threshold -> size filtering -> centroid extraction -> visualization
"""

import os
import cv2
import numpy as np
import json
from typing import List, Tuple

# Import refactored modules
import sys
sys.path.append('src')

from asoct_mcd.preprocessing import ConverterFactory, PreprocessingPipeline
from asoct_mcd.threshold_methods import ThresholdFactory
from asoct_mcd.image_processing import filter_objects_by_size, get_centroids, draw_rectangles


def load_image_cv2(image_path: str) -> np.ndarray:
    """Load image using OpenCV."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    return image

def save_intermediate_results(output_dir: str, image_name: str, **results):
    """Save intermediate processing results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays as images
    for name, data in results.items():
        if isinstance(data, np.ndarray):
            if len(data.shape) == 2:  # Grayscale or mask
                cv2.imwrite(os.path.join(output_dir, f"{name}.png"), data)
            elif len(data.shape) == 3:  # Color image
                cv2.imwrite(os.path.join(output_dir, f"{name}.png"), data)
        elif hasattr(data, 'save'):  # PIL Image
            data.save(os.path.join(output_dir, f"{name}.png"))

def main():
    # Configuration
    image_path = "tests/data/image1.png"
    lambda_factor = 0.83  # From your config
    min_size = 1
    max_size = 25
    box_size = 10
    
    # Extract image name for output directory
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"tests/data/{image_name}_re"
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load image and convert to grayscale using refactored code
    print("Step 1: Converting to grayscale...")
    original_image = load_image_cv2(image_path)
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline()
    grayscale_converter = ConverterFactory.create_converter('grayscale')
    pipeline.add_converter(grayscale_converter)
    
    # Apply preprocessing
    grayscale_image = pipeline.process(original_image)
    print(f"Image shape after grayscale conversion: {grayscale_image.shape}")
    
    # Step 2: Apply Otsu threshold with lambda using refactored code
    print("Step 2: Applying Otsu threshold...")
    threshold_method = ThresholdFactory.create_threshold('otsu')
    threshold_mask, thresh_value = threshold_method.apply_threshold(
        grayscale_image, 
        lambda_factor=lambda_factor, 
        return_thresh=True
    )
    print(f"Original threshold value: {thresh_value/lambda_factor:.2f}")
    print(f"Adjusted threshold (lambda={lambda_factor}): {thresh_value:.2f}")
    
    # Step 3: Filter objects by size using refactored code
    print("Step 3: Filtering objects by size...")
    filtered_mask = filter_objects_by_size(threshold_mask, min_size=min_size, max_size=max_size)
    
    # Count objects before and after filtering
    num_objects_before = cv2.connectedComponents(threshold_mask)[0] - 1
    num_objects_after = cv2.connectedComponents(filtered_mask)[0] - 1
    print(f"Objects before size filtering: {num_objects_before}")
    print(f"Objects after size filtering: {num_objects_after}")
    
    # Step 4: Extract centroids using refactored code
    print("Step 4: Extracting centroids...")
    centroids = get_centroids(filtered_mask)
    print(f"Number of centroids found: {len(centroids)}")
    
    # Step 5: Draw bounding boxes using refactored code
    print("Step 5: Drawing bounding boxes...")
    result_image = draw_rectangles(
        image_path, 
        centroids, 
        box_size=box_size, 
        color='#00FF00',
        line_width=2
    )
    
    # Step 6: Save all results
    print("Step 6: Saving results...")
    save_intermediate_results(
        output_dir, image_name,
        grayscale=grayscale_image,
        threshold_mask=threshold_mask,
        filtered_mask=filtered_mask
    )
    
    # Save the result image with boxes
    result_image.save(os.path.join(output_dir, "result_with_boxes.png"))
    
    # Save centroids as JSON for easy comparison
    centroids_data = {
        'centroids': centroids,
        'count': len(centroids),
        'parameters': {
            'lambda_factor': lambda_factor,
            'min_size': min_size,
            'max_size': max_size,
            'threshold_value': float(thresh_value)
        }
    }
    
    with open(os.path.join(output_dir, 'centroids.json'), 'w') as f:
        json.dump(centroids_data, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Centroids: {centroids}")
    print("Processing complete!")

if __name__ == "__main__":
    main()