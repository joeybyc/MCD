#!/usr/bin/env python3
"""
Integration test for complete cell detection pipeline.
Tests single image and batch processing with comprehensive output.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List
import json

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from asoct_mcd.pipeline import create_default_pipeline, PipelineBuilder
from asoct_mcd.threshold_methods import ThresholdFactory
from asoct_mcd.image_processing import draw_rectangles, overlay_mask


def save_visualization(image_path: str, result, output_dir: Path, image_name: str):
    """Save comprehensive visualization of detection results."""
    
    # Load original image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Chamber mask
    cv2.imwrite(str(output_dir / f"{image_name}_chamber_mask.png"), result.chamber_mask)
    
    # 2. Cell mask  
    cv2.imwrite(str(output_dir / f"{image_name}_cell_mask.png"), result.cell_mask)
    
    # 3. Threshold masks (if available)
    if result.threshold_mask is not None:
        cv2.imwrite(str(output_dir / f"{image_name}_threshold_mask.png"), result.threshold_mask)
    
    if result.candidate_mask is not None:
        cv2.imwrite(str(output_dir / f"{image_name}_candidate_mask.png"), result.candidate_mask)
    
    # 4. Cell locations visualization
    if result.cell_locations:
        cell_vis = draw_rectangles(
            image_path, 
            result.cell_locations, 
            box_size=10, 
            color='#00FF00'  # Green for cells
        )
        cell_vis.save(str(output_dir / f"{image_name}_cell_locations.png"))
    
    # 5. Chamber overlay
    overlay_mask(
        image_path,
        result.chamber_mask,
        str(output_dir / f"{image_name}_chamber_overlay.png"),
        color=(255, 0, 0),  # Red
        alpha=0.3
    )
    
    # 6. Cell overlay
    overlay_mask(
        image_path,
        result.cell_mask,
        str(output_dir / f"{image_name}_cell_overlay.png"),
        color=(0, 255, 0),  # Green  
        alpha=0.5
    )
    
    # 7. Prompt points (if available)
    if result.prompt_points is not None:
        prompt_vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        for i, point in enumerate(result.prompt_points):
            x, y = int(point[0]), int(point[1])
            cv2.circle(prompt_vis, (x, y), 8, (0, 255, 255), -1)  # Yellow
            cv2.putText(prompt_vis, f"P{i+1}", (x+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(output_dir / f"{image_name}_prompt_points.png"), prompt_vis)


def create_threshold_comparison(image_path: str, output_dir: Path, image_name: str):
    """Create threshold comparison with and without lambda."""
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold_method = ThresholdFactory.create_threshold('otsu')
    
    # Original threshold
    original_mask, threshold_val = threshold_method.apply_threshold(
        image, lambda_factor=1.0, return_thresh=True
    )
    
    # Lambda adjusted threshold
    lambda_mask, lambda_threshold_val = threshold_method.apply_threshold(
        image, lambda_factor=0.83, return_thresh=True
    )
    
    # Save masks
    cv2.imwrite(str(output_dir / f"{image_name}_otsu_original.png"), original_mask)
    cv2.imwrite(str(output_dir / f"{image_name}_otsu_lambda.png"), lambda_mask)
    
    # Create comparison visualization
    comparison = np.hstack([original_mask, lambda_mask])
    cv2.imwrite(str(output_dir / f"{image_name}_threshold_comparison.png"), comparison)
    
    return {
        'original_threshold': float(threshold_val),
        'lambda_threshold': float(lambda_threshold_val),
        'lambda_factor': 0.83
    }


def test_single_image():
    """Test pipeline on single image."""
    print("\n=== Single Image Test ===")
    
    # Setup paths
    image_path = Path("tests/data/sample_asoct.png")
    output_dir = Path("tests/data/sample_asoct_pipeline_output")
    output_dir.mkdir(exist_ok=True)
    
    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Create pipeline
        print("Creating pipeline...")
        pipeline = create_default_pipeline()
        
        # Process image
        print(f"Processing image: {image_path}")
        result = pipeline.detect_cells(str(image_path), save_intermediates=True)
        
        # Save visualizations
        print("Saving visualizations...")
        save_visualization(str(image_path), result, output_dir, "sample_asoct")
        
        # Create threshold comparison
        print("Creating threshold comparison...")
        threshold_info = create_threshold_comparison(str(image_path), output_dir, "sample_asoct")
        
        # Save summary
        summary = {
            'single_image_test': True,
            'image_path': str(image_path),
            'detection_results': result.get_summary(),
            'threshold_analysis': threshold_info,
            'output_files': {
                'chamber_mask': 'sample_asoct_chamber_mask.png',
                'cell_mask': 'sample_asoct_cell_mask.png', 
                'cell_locations': 'sample_asoct_cell_locations.png',
                'chamber_overlay': 'sample_asoct_chamber_overlay.png',
                'cell_overlay': 'sample_asoct_cell_overlay.png',
                'threshold_original': 'sample_asoct_otsu_original.png',
                'threshold_lambda': 'sample_asoct_otsu_lambda.png',
                'threshold_comparison': 'sample_asoct_threshold_comparison.png'
            }
        }
        
        with open(output_dir / "single_image_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Single image test completed")
        print(f"✓ Detected {result.cell_count} cells")
        print(f"✓ Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_images():
    """Test pipeline on batch of images."""
    print("\n=== Batch Images Test ===")
    
    # Setup paths
    images_dir = Path("tests/data/images")
    output_dir = Path("tests/data/images_pipeline_output")
    output_dir.mkdir(exist_ok=True)
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        print("Creating sample directory structure...")
        images_dir.mkdir(exist_ok=True)
        print(f"Please add test images to: {images_dir}")
        return False
    
    # Find image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    image_paths = [
        p for p in images_dir.iterdir() 
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"❌ No images found in: {images_dir}")
        return False
    
    try:
        # Create pipeline
        print(f"Creating pipeline for {len(image_paths)} images...")
        pipeline = create_default_pipeline()
        
        # Process batch
        print("Processing batch...")
        batch_result = pipeline.detect_batch(
            [str(p) for p in image_paths], 
            save_intermediates=True
        )
        
        # Save individual results
        print("Saving individual results...")
        for result in batch_result.results:
            image_name = Path(result.image_path).stem
            image_output_dir = output_dir / image_name
            image_output_dir.mkdir(exist_ok=True)
            
            save_visualization(result.image_path, result, image_output_dir, image_name)
            create_threshold_comparison(result.image_path, image_output_dir, image_name)
        
        # Save batch summary
        batch_summary = {
            'batch_test': True,
            'images_processed': len(batch_result.results),
            'batch_statistics': batch_result.get_batch_summary(),
            'individual_results': [result.get_summary() for result in batch_result.results]
        }
        
        with open(output_dir / "batch_summary.json", 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"✓ Batch test completed")
        print(f"✓ Processed {batch_result.image_count} images")
        print(f"✓ Total cells detected: {batch_result.total_cells}")
        print(f"✓ Average cells per image: {batch_result.total_cells / batch_result.image_count:.1f}")
        print(f"✓ Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_pipeline():
    """Test custom pipeline configuration."""
    print("\n=== Custom Pipeline Test ===")
    
    try:
        # Create custom pipeline using builder
        custom_pipeline = (PipelineBuilder()
                          .with_threshold_method('otsu')
                          .with_threshold_lambda(0.83)
                          .with_size_bounds(2, 30)
                          .build())
        
        print("✓ Custom pipeline created successfully")
        print(f"  - Threshold method: {custom_pipeline.config.threshold_method}")
        print(f"  - Threshold lambda: {custom_pipeline.config.threshold_lambda}")
        print(f"  - Size bounds: [{custom_pipeline.config.lower_bound}, {custom_pipeline.config.upper_bound}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom pipeline test failed: {e}")
        return False


def main():
    """Run all pipeline tests."""
    print("AS-OCT Cell Detection Pipeline Integration Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Single image
    results.append(test_single_image())
    
    # Test 2: Batch images  
    results.append(test_batch_images())
    
    # Test 3: Custom pipeline
    results.append(test_custom_pipeline())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"Pipeline Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All pipeline tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)