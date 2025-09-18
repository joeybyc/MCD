"""
Author: B. Chen

Image processing operations for cropping and drawing on images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Union


def crop_regions(image_path: str, centroids: List[Tuple[float, float]], 
                crop_size: int = 10) -> List[np.ndarray]:
    """
    Crop square regions around specified centroids from an image.
    
    Args:
        image_path: Path to the input image
        centroids: List of (x, y) coordinates for crop centers
        crop_size: Size of square crop (crop_size x crop_size)
        
    Returns:
        List of cropped image arrays (BGR format)
        
    Raises:
        ValueError: If image cannot be loaded or crop_size is invalid
    """
    if crop_size <= 0:
        raise ValueError("Crop size must be positive")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    height, width = image.shape[:2]
    half_size = crop_size // 2
    cropped_regions = []
    
    for x, y in centroids:
        x, y = int(x), int(y)
        
        # Calculate crop boundaries with image bounds checking
        x1 = max(x - half_size, 0)
        y1 = max(y - half_size, 0)
        x2 = min(x + half_size, width)
        y2 = min(y + half_size, height)
        
        cropped = image[y1:y2, x1:x2]
        
        # Resize to exact crop_size if boundary cropping occurred
        if cropped.shape[:2] != (crop_size, crop_size):
            cropped = cv2.resize(cropped, (crop_size, crop_size))
        
        cropped_regions.append(cropped)
    
    return cropped_regions


def save_cropped_regions(image_path: str, centroids: List[Tuple[float, float]], 
                        output_dir: str, base_filename: str = "crop", 
                        crop_size: int = 10) -> List[str]:
    """
    Crop and save regions around centroids to individual files.
    
    Args:
        image_path: Path to the input image
        centroids: List of (x, y) coordinates for crop centers
        output_dir: Directory to save cropped images
        base_filename: Base name for output files
        crop_size: Size of square crop
        
    Returns:
        List of paths to saved cropped images
        
    Raises:
        ValueError: If image cannot be loaded or output directory creation fails
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    cropped_regions = crop_regions(image_path, centroids, crop_size)
    saved_paths = []
    
    for idx, cropped in enumerate(cropped_regions):
        filename = f"{base_filename}_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cropped)
        saved_paths.append(filepath)
    
    return saved_paths


def draw_rectangles(image_path: str, centroids: List[Tuple[float, float]], 
                   box_size: int = 10, color: str = "#FF0000", 
                   line_width: int = 2) -> Image.Image:
    """
    Draw rectangles around centroids on an image.
    
    Args:
        image_path: Path to the input image
        centroids: List of (x, y) coordinates for rectangle centers
        box_size: Size of rectangles to draw
        color: Color of rectangles in hex format (e.g., '#FF0000' for red)
        line_width: Width of rectangle outline
        
    Returns:
        PIL Image with drawn rectangles
        
    Raises:
        ValueError: If image cannot be loaded or parameters are invalid
    """
    if box_size <= 0 or line_width <= 0:
        raise ValueError("Box size and line width must be positive")
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Cannot load image from {image_path}: {e}")
    
    draw = ImageDraw.Draw(image)
    half_box = box_size // 2
    
    for x, y in centroids:
        x, y = int(x), int(y)
        
        # Calculate rectangle bounds
        left = x - half_box
        top = y - half_box
        right = x + half_box
        bottom = y + half_box
        
        # Draw rectangle
        draw.rectangle([left, top, right, bottom], 
                      outline=color, width=line_width)
    
    return image


def overlay_mask(image_path: str, mask: np.ndarray, output_path: str, 
                color: Tuple[int, int, int] = (255, 0, 0), 
                alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a semi-transparent colored mask on an image and save result.
    
    Args:
        image_path: Path to the input image or numpy array
        mask: Binary mask array (0/255 format)
        output_path: Path to save the overlaid image
        color: Overlay color in BGR format
        alpha: Opacity of overlay (0.0 to 1.0)
        
    Returns:
        Overlaid image array
        
    Raises:
        ValueError: If inputs are invalid or overlay operation fails
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")
    
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
    else:
        image = image_path
    
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("Mask and image must have the same dimensions")
    
    # Create colored overlay
    colored_mask = np.zeros_like(image)
    mask_bool = mask.astype(bool)
    colored_mask[mask_bool] = color
    
    # Blend images
    overlay = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    
    # Save result
    cv2.imwrite(output_path, overlay)
    
    return overlay