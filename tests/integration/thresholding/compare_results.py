"""
Comparison script to analyze differences between original and refactored implementations.
"""

import json
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict, Any


def load_centroids_from_json(json_path: str) -> Dict[str, Any]:
    """Load centroids data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_centroid_differences(centroids1: List[Tuple[float, float]], 
                                 centroids2: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Calculate differences between two sets of centroids."""
    if len(centroids1) != len(centroids2):
        return {
            'count_match': False,
            'count_diff': abs(len(centroids1) - len(centroids2)),
            'count1': len(centroids1),
            'count2': len(centroids2),
            'position_analysis': "Cannot compare positions - different counts"
        }
    
    # Sort centroids for consistent comparison
    centroids1_sorted = sorted(centroids1, key=lambda x: (x[0], x[1]))
    centroids2_sorted = sorted(centroids2, key=lambda x: (x[0], x[1]))
    
    differences = []
    max_diff = 0
    total_diff = 0
    
    for (x1, y1), (x2, y2) in zip(centroids1_sorted, centroids2_sorted):
        diff_x = abs(x1 - x2)
        diff_y = abs(y1 - y2)
        euclidean_diff = np.sqrt(diff_x**2 + diff_y**2)
        
        differences.append({
            'centroid1': (x1, y1),
            'centroid2': (x2, y2),
            'diff_x': diff_x,
            'diff_y': diff_y,
            'euclidean_distance': euclidean_diff
        })
        
        max_diff = max(max_diff, euclidean_diff)
        total_diff += euclidean_diff
    
    avg_diff = total_diff / len(differences) if differences else 0
    
    return {
        'count_match': True,
        'count1': len(centroids1),
        'count2': len(centroids2),
        'position_differences': differences,
        'max_euclidean_distance': max_diff,
        'avg_euclidean_distance': avg_diff,
        'total_euclidean_distance': total_diff
    }

def compare_images(img1_path: str, img2_path: str) -> Dict[str, Any]:
    """Compare two images pixel by pixel."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return {'error': f'Cannot load images: {img1_path}, {img2_path}'}
    
    if img1.shape != img2.shape:
        return {
            'shapes_match': False,
            'shape1': img1.shape,
            'shape2': img2.shape
        }
    
    # Calculate differences
    diff = cv2.absdiff(img1, img2)
    non_zero_pixels = np.count_nonzero(diff)
    total_pixels = img1.size
    
    return {
        'shapes_match': True,
        'identical': non_zero_pixels == 0,
        'different_pixels': int(non_zero_pixels),
        'total_pixels': int(total_pixels),
        'difference_percentage': (non_zero_pixels / total_pixels) * 100,
        'max_pixel_difference': int(np.max(diff)),
        'mean_pixel_difference': float(np.mean(diff))
    }

def main():
    image_name = "image1"
    old_dir = f"tests/data/{image_name}_old"
    re_dir = f"tests/data/{image_name}_re"
    
    print("=" * 60)
    print("COMPARISON BETWEEN ORIGINAL AND REFACTORED IMPLEMENTATIONS")
    print("=" * 60)
    
    # Compare centroids
    print("\n1. CENTROID COMPARISON")
    print("-" * 30)
    
    try:
        old_data = load_centroids_from_json(os.path.join(old_dir, 'centroids.json'))
        re_data = load_centroids_from_json(os.path.join(re_dir, 'centroids.json'))
        
        print(f"Original implementation: {old_data['count']} centroids")
        print(f"Refactored implementation: {re_data['count']} centroids")
        
        # Compare parameters
        print(f"\nParameters comparison:")
        old_params = old_data['parameters']
        re_params = re_data['parameters']
        
        for key in old_params:
            if key in re_params:
                print(f"  {key}: {old_params[key]} vs {re_params[key]} - {'✓' if abs(old_params[key] - re_params[key]) < 1e-6 else '✗'}")
            else:
                print(f"  {key}: {old_params[key]} vs [missing] - ✗")
        
        # Compare centroids
        diff_analysis = calculate_centroid_differences(old_data['centroids'], re_data['centroids'])
        
        if diff_analysis['count_match']:
            print(f"\nCentroid positions:")
            print(f"  Count matches: ✓")
            print(f"  Maximum euclidean distance: {diff_analysis['max_euclidean_distance']:.6f}")
            print(f"  Average euclidean distance: {diff_analysis['avg_euclidean_distance']:.6f}")
            
            if diff_analysis['max_euclidean_distance'] < 1e-10:
                print("  Result: IDENTICAL CENTROIDS ✓")
            elif diff_analysis['max_euclidean_distance'] < 1e-3:
                print("  Result: NEARLY IDENTICAL (sub-pixel differences) ✓")
            else:
                print("  Result: SIGNIFICANT DIFFERENCES ✗")
                print("\nDetailed differences:")
                for i, diff in enumerate(diff_analysis['position_differences'][:5]):  # Show first 5
                    print(f"    Centroid {i}: ({diff['centroid1'][0]:.2f}, {diff['centroid1'][1]:.2f}) vs "
                          f"({diff['centroid2'][0]:.2f}, {diff['centroid2'][1]:.2f}) - "
                          f"Distance: {diff['euclidean_distance']:.6f}")
        else:
            print(f"  Count mismatch: {diff_analysis['count1']} vs {diff_analysis['count2']} - ✗")
            
    except Exception as e:
        print(f"Error comparing centroids: {e}")
    
    # Compare intermediate images
    print("\n2. INTERMEDIATE IMAGE COMPARISON")
    print("-" * 30)
    
    image_types = ['grayscale', 'threshold_mask', 'filtered_mask']
    
    for img_type in image_types:
        old_path = os.path.join(old_dir, f"{img_type}.png")
        re_path = os.path.join(re_dir, f"{img_type}.png")
        
        if os.path.exists(old_path) and os.path.exists(re_path):
            comparison = compare_images(old_path, re_path)
            
            print(f"\n{img_type}:")
            if 'error' in comparison:
                print(f"  Error: {comparison['error']}")
            elif comparison['shapes_match']:
                if comparison['identical']:
                    print(f"  Status: IDENTICAL ✓")
                else:
                    print(f"  Status: DIFFERENT ✗")
                    print(f"  Different pixels: {comparison['different_pixels']} ({comparison['difference_percentage']:.4f}%)")
                    print(f"  Max pixel difference: {comparison['max_pixel_difference']}")
                    print(f"  Mean pixel difference: {comparison['mean_pixel_difference']:.6f}")
            else:
                print(f"  Status: SHAPE MISMATCH ✗")
                print(f"  Shapes: {comparison['shape1']} vs {comparison['shape2']}")
        else:
            print(f"\n{img_type}: Missing files ✗")
    
    # Summary
    print("\n3. SUMMARY")
    print("-" * 30)
    try:
        old_count = old_data['count']
        re_count = re_data['count']
        
        if old_count == re_count and diff_analysis.get('max_euclidean_distance', float('inf')) < 1e-6:
            print("✓ IMPLEMENTATIONS ARE EQUIVALENT")
            print("  Both implementations produce identical results.")
        elif old_count == re_count:
            print("~ IMPLEMENTATIONS ARE NEARLY EQUIVALENT")
            print("  Same number of detections with minor position differences.")
        else:
            print("✗ IMPLEMENTATIONS DIFFER")
            print("  Different number of detections or significant position differences.")
            print("  Review the pipeline for potential issues.")
            
    except:
        print("Could not generate summary due to missing data.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()