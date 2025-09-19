"""
Integration test using original (non-refactored) code for threshold-based cell detection.
Tests the pipeline: grayscale -> otsu threshold -> size filtering -> centroid extraction -> visualization
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from typing import List, Tuple
import json

# Original functions extracted from your codebase
def load_image(image_path: str, to_greyscale: bool = True) -> np.ndarray:
    """Load image and optionally convert to grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if to_greyscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def threshold_method(image: np.ndarray, return_thresh: bool = False, lambda_setting: float = 1.0):
    """Apply Otsu thresholding with lambda adjustment."""
    thresh_value = threshold_otsu(image)
    thresh_value = thresh_value * lambda_setting
    _, mask = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    if return_thresh:
        return mask, thresh_value
    else:
        return mask

def filter_objects_by_size(mask: np.ndarray, lower_bound: int = 2, upper_bound: int = 25) -> np.ndarray:
    """Filter objects by size constraints."""
    if mask.dtype != np.uint8:
        raise ValueError("Mask is not of type uint8.")
    
    ret, labels = cv2.connectedComponents(mask)
    filtered_mask = np.zeros_like(mask)
    
    for label in range(1, ret):
        area = np.sum(labels == label)
        if lower_bound <= area <= upper_bound:
            filtered_mask[labels == label] = 255
    
    return filtered_mask

def get_centroids(mask_image: np.ndarray) -> List[Tuple[float, float]]:
    """Extract centroids from binary mask."""
    num_labels, labels = cv2.connectedComponents(mask_image)
    centroids = []
    for i in range(1, num_labels):
        ys, xs = np.where(labels == i)
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
        centroids.append((centroid_x, centroid_y))
    return centroids

def draw_square(image_path: str, cell_centroids: List[Tuple[float, float]], 
               box_size: int = 10, box_outline: str = 'red') -> Image.Image:
    """Draw squares around centroids."""
    half_box_size = box_size // 2
    image_with_cell_boxes = Image.open(image_path)
    draw = ImageDraw.Draw(image_with_cell_boxes)
    
    for x, y in cell_centroids:
        x, y = int(x), int(y)
        box = [x-half_box_size, y-half_box_size, x+half_box_size, y+half_box_size]
        draw.rectangle(box, outline=box_outline, width=2)
    
    return image_with_cell_boxes

def save_intermediate_results(output_dir: str, image_name: str, **results):
    """Save intermediate processing results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save masks
    for name, data in results.items():
        if isinstance(data, np.ndarray):
            cv2.imwrite(os.path.join(output_dir, f"{name}.png"), data)
        elif isinstance(data, Image.Image):
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
    output_dir = f"tests/data/{image_name}_old"
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load and convert to grayscale
    print("Step 1: Converting to grayscale...")
    image = load_image(image_path, to_greyscale=True)
    print(f"Image shape after grayscale conversion: {image.shape}")
    
    # Step 2: Apply Otsu threshold with lambda
    print("Step 2: Applying Otsu threshold...")
    threshold_mask, thresh_value = threshold_method(image, return_thresh=True, lambda_setting=lambda_factor)
    print(f"Threshold value: {thresh_value:.2f} (lambda={lambda_factor})")
    print(f"Adjusted threshold: {thresh_value:.2f}")
    
    # Step 3: Filter objects by size
    print("Step 3: Filtering objects by size...")
    filtered_mask = filter_objects_by_size(threshold_mask, lower_bound=min_size, upper_bound=max_size)
    
    # Count objects before and after filtering
    num_objects_before = cv2.connectedComponents(threshold_mask)[0] - 1
    num_objects_after = cv2.connectedComponents(filtered_mask)[0] - 1
    print(f"Objects before size filtering: {num_objects_before}")
    print(f"Objects after size filtering: {num_objects_after}")
    
    # Step 4: Extract centroids
    print("Step 4: Extracting centroids...")
    centroids = get_centroids(filtered_mask)
    print(f"Number of centroids found: {len(centroids)}")
    
    # Step 5: Draw bounding boxes
    print("Step 5: Drawing bounding boxes...")
    result_image = draw_square(image_path, centroids, box_size=box_size, box_outline='#00FF00')
    
    # Save all results
    print("Step 6: Saving results...")
    save_intermediate_results(
        output_dir, image_name,
        grayscale=image,
        threshold_mask=threshold_mask,
        filtered_mask=filtered_mask,
        result_with_boxes=result_image
    )
    
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