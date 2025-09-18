"""
Author: B. Chen

Mask processing operations for binary image masks.
"""

import cv2
import numpy as np
from typing import List, Tuple


def intersect_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Compute intersection of two binary masks.
    
    Args:
        mask1: First binary mask (0/255 format)
        mask2: Second binary mask (0/255 format)
        
    Returns:
        Intersection mask (0/255 format)
        
    Raises:
        ValueError: If masks have different shapes or are not binary
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same dimensions")
    
    if not _is_binary_mask(mask1) or not _is_binary_mask(mask2):
        raise ValueError("Both inputs must be binary masks (0/255 format)")
    
    return cv2.bitwise_and(mask1, mask2)

def union_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Compute union of two binary masks.
    
    Args:
        mask1: First binary mask (0/255 format)
        mask2: Second binary mask (0/255 format)
        
    Returns:
        Union mask (0/255 format)
        
    Raises:
        ValueError: If masks have different shapes or are not binary
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same dimensions")
    
    if not _is_binary_mask(mask1) or not _is_binary_mask(mask2):
        raise ValueError("Both inputs must be binary masks (0/255 format)")
    
    return cv2.bitwise_or(mask1, mask2)


def get_centroids(mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    Extract centroids of all connected components in a binary mask.
    
    Args:
        mask: Binary mask (0/255 format)
        
    Returns:
        List of (centroid_x, centroid_y) tuples
        
    Raises:
        ValueError: If input is not a binary mask
    """
    if not _is_binary_mask(mask):
        raise ValueError("Input must be a binary mask (0/255 format)")
    
    num_labels, labels = cv2.connectedComponents(mask)
    centroids = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        ys, xs = np.where(labels == i)
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        centroids.append((centroid_x, centroid_y))
    
    return centroids


def filter_objects_by_size(mask: np.ndarray, min_size: int = 1, 
                          max_size: int = 25) -> np.ndarray:
    """
    Filter objects in binary mask by size constraints.
    
    Args:
        mask: Binary mask (0/255 format)
        min_size: Minimum object size in pixels
        max_size: Maximum object size in pixels
        
    Returns:
        Filtered binary mask (0/255 format)
        
    Raises:
        ValueError: If input is not a binary mask or size constraints are invalid
    """
    if not _is_binary_mask(mask):
        raise ValueError("Input must be a binary mask (0/255 format)")
    
    if min_size <= 0 or max_size <= 0 or min_size > max_size:
        raise ValueError("Size constraints must be positive and min_size <= max_size")
    
    num_labels, labels = cv2.connectedComponents(mask)
    filtered_mask = np.zeros_like(mask)
    
    for label in range(1, num_labels):  # Skip background (label 0)
        area = np.sum(labels == label)
        if min_size <= area <= max_size:
            filtered_mask[labels == label] = 255
    
    return filtered_mask


def _is_binary_mask(mask: np.ndarray) -> bool:
    """
    Check if array is a binary mask in 0/255 format.
    
    Args:
        mask: Input array to check
        
    Returns:
        True if mask is binary (0/255), False otherwise
    """
    if mask.dtype != np.uint8:
        return False
    
    unique_values = np.unique(mask)
    return len(unique_values) <= 2 and all(val in [0, 255] for val in unique_values)