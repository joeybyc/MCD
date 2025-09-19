"""
Author: B. Chen

Integration test for thresholding methods with visual output comparison.
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from asoct_mcd.threshold_methods import ThresholdFactory


def load_test_image(image_path: str) -> np.ndarray:
    """Load test image in grayscale format."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def save_results(output_dir: str, method_name: str, mask: np.ndarray, 
                threshold_value: float, lambda_factor: float) -> None:
    """Save thresholding results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mask image
    mask_filename = f"{method_name}_lambda_{lambda_factor:.2f}_mask.png"
    cv2.imwrite(os.path.join(output_dir, mask_filename), mask)
    
    # Save threshold info
    info_filename = f"{method_name}_lambda_{lambda_factor:.2f}_info.txt"
    with open(os.path.join(output_dir, info_filename), 'w') as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Lambda factor: {lambda_factor}\n")
        f.write(f"Threshold value: {threshold_value:.4f}\n")
        f.write(f"Mask shape: {mask.shape}\n")
        f.write(f"White pixels: {np.sum(mask == 255)}\n")
        f.write(f"Black pixels: {np.sum(mask == 0)}\n")


def test_all_thresholding_methods():
    """Test all thresholding methods with different lambda factors."""
    # Setup paths
    test_data_dir = Path(__file__).parent.parent / 'data'
    image_path = test_data_dir / 'img1_grey.png'
    
    # Load test image
    image = load_test_image(str(image_path))
    print(f"Loaded image: {image.shape}, dtype: {image.dtype}")
    
    # Get image name for output directory
    image_name = image_path.stem
    output_base_dir = test_data_dir / image_name
    
    # Save original image to output directory
    os.makedirs(output_base_dir, exist_ok=True)
    cv2.imwrite(str(output_base_dir / 'original.png'), image)
    
    # Test parameters
    methods = ThresholdFactory.get_available_methods()
    lambda_factors = [0.1, 0.3, 0.83, 1.0]
    
    print(f"Testing methods: {methods}")
    print(f"Lambda factors: {lambda_factors}")
    
    # Test each method with different lambda factors
    for method_name in methods:
        print(f"\nTesting method: {method_name}")
        threshold_method = ThresholdFactory.create_threshold(method_name)
        
        for lambda_factor in lambda_factors:
            try:
                # Apply thresholding
                mask, threshold_value = threshold_method.apply_threshold(
                    image, lambda_factor=lambda_factor, return_thresh=True
                )
                
                # Save results
                save_results(str(output_base_dir), method_name, mask, 
                           threshold_value, lambda_factor)
                
                print(f"  λ={lambda_factor}: threshold={threshold_value:.4f}, "
                      f"white_pixels={np.sum(mask == 255)}")
                
            except Exception as e:
                print(f"  λ={lambda_factor}: Error - {e}")
    
    print(f"\nResults saved to: {output_base_dir}")


def create_comparison_grid():
    """Create a comparison grid of all results."""
    test_data_dir = Path(__file__).parent.parent / 'data'
    image_name = 'img1_grey'
    output_dir = test_data_dir / image_name
    
    if not output_dir.exists():
        print("No results found. Run test_all_thresholding_methods() first.")
        return
    
    # Load original image
    original = cv2.imread(str(output_dir / 'original.png'), cv2.IMREAD_GRAYSCALE)
    
    # Get all mask files
    mask_files = list(output_dir.glob('*_mask.png'))
    mask_files.sort()
    
    # Create grid layout
    methods = ThresholdFactory.get_available_methods()
    lambda_factors = [0.7, 0.83, 1.0, 1.2]
    
    # Calculate grid size
    rows = len(methods) + 1  # +1 for header with original
    cols = len(lambda_factors) + 1  # +1 for method name
    
    # Create comparison grid
    cell_height, cell_width = original.shape
    grid = np.zeros((rows * cell_height, cols * cell_width), dtype=np.uint8)
    
    # Add original image to top-left
    grid[0:cell_height, 0:cell_width] = original
    
    # Add lambda factor labels (simulate text by creating simple patterns)
    for i, lambda_val in enumerate(lambda_factors):
        col = (i + 1) * cell_width
        # Create a simple pattern to represent lambda value
        pattern = np.ones((cell_height, cell_width), dtype=np.uint8) * int(lambda_val * 100)
        grid[0:cell_height, col:col+cell_width] = pattern
    
    # Add method results
    for method_idx, method_name in enumerate(methods):
        row = (method_idx + 1) * cell_height
        
        for lambda_idx, lambda_val in enumerate(lambda_factors):
            col = (lambda_idx + 1) * cell_width
            
            # Find corresponding mask file
            mask_pattern = f"{method_name}_lambda_{lambda_val:.2f}_mask.png"
            mask_file = output_dir / mask_pattern
            
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                grid[row:row+cell_height, col:col+cell_width] = mask
    
    # Save comparison grid
    cv2.imwrite(str(output_dir / 'comparison_grid.png'), grid)
    print(f"Comparison grid saved to: {output_dir / 'comparison_grid.png'}")


if __name__ == "__main__":
    test_all_thresholding_methods()
    create_comparison_grid()