"""
Author: B. Chen

Integration test for segmentation functionality.
Tests the complete segmentation pipeline from image to mask.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import model implementations to trigger registration
import asoct_mcd.model_implementations

from asoct_mcd.model_management import ModelFactory
from asoct_mcd.prompt_generation import PromptGeneratorFactory
from asoct_mcd.segmentation import SegmentorFactory
from asoct_mcd.image_processing import overlay_mask


def test_segment_integration():
    """Test complete segmentation pipeline."""
    # Setup paths
    image_path = "tests/data/image1.png"
    image_name = Path(image_path).stem
    output_dir = Path(f"tests/data/{image_name}_segment_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing segmentation with image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    print(f"Image shape: {image.shape}")
    
    # Step 1: Create prompt generator (I2ACP)
    print("\n1. Creating I2ACP prompt generator...")
    prompt_generator = PromptGeneratorFactory.create_generator(
        'i2acp',
        offset_ratio=0.02,
        area_ratio_threshold=0.65
    )
    
    # Step 2: Generate prompts
    print("2. Generating prompts...")
    points, labels = prompt_generator.generate(image)
    print(f"Generated {len(points)} prompt points: {points}")
    print(f"Labels: {labels}")
    
    # Save prompt visualization
    prompt_vis = image.copy()
    if len(prompt_vis.shape) == 2:
        prompt_vis = cv2.cvtColor(prompt_vis, cv2.COLOR_GRAY2BGR)
    
    for i, (point, label) in enumerate(zip(points, labels)):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive
        cv2.circle(prompt_vis, (int(point[0]), int(point[1])), 5, color, -1)
        cv2.putText(prompt_vis, f"{i}", (int(point[0])+10, int(point[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(str(output_dir / "01_prompt_points.png"), prompt_vis)
    print(f"Saved prompt visualization to: {output_dir / '01_prompt_points.png'}")
    
    # Step 3: Create SAM model wrapper
    print("\n3. Creating SAM model...")
    
    # Create wrapper class for segmentor
    class SAMWrapper:
        def __init__(self):
            self.sam_model = ModelFactory.create_model('sam_vit_b')
        
        def get_mask(self, image, prompts):
            return self.sam_model.segment(image, prompts)
    
    class PromptWrapper:
        def __init__(self, generator):
            self.generator = generator
        
        def generate(self, image):
            return self.generator.generate(image)
    
    # Step 4: Create segmentor
    print("4. Creating zero-shot segmentor...")
    segmentor = SegmentorFactory.create_segmentor(
        'zero_shot',
        model_wrapper=SAMWrapper(),
        prompt_generator=PromptWrapper(prompt_generator)
    )
    
    # Step 5: Perform segmentation
    print("5. Performing segmentation...")
    try:
        chamber_mask = segmentor.segment(image)
        print(f"Segmentation completed. Mask shape: {chamber_mask.shape}")
        print(f"Mask unique values: {np.unique(chamber_mask)}")
        print(f"Mask coverage: {np.sum(chamber_mask > 0) / chamber_mask.size * 100:.2f}%")
        
        # Save chamber mask
        cv2.imwrite(str(output_dir / "02_chamber_mask.png"), chamber_mask)
        print(f"Saved chamber mask to: {output_dir / '02_chamber_mask.png'}")
        
        # Step 6: Create overlay visualization
        print("6. Creating overlay visualization...")
        overlay_image = overlay_mask(
            image_path, 
            chamber_mask, 
            str(output_dir / "03_overlay.png"),
            color=(255, 0, 0),  # Red overlay
            alpha=0.5
        )
        print(f"Saved overlay to: {output_dir / '03_overlay.png'}")
        
        # Step 7: Save original image copy
        cv2.imwrite(str(output_dir / "00_original.png"), image)
        print(f"Saved original image to: {output_dir / '00_original.png'}")
        
        # Step 8: Statistics
        print("\n=== SEGMENTATION RESULTS ===")
        print(f"Image size: {image.shape}")
        print(f"Mask size: {chamber_mask.shape}")
        print(f"Total pixels: {chamber_mask.size}")
        print(f"Segmented pixels: {np.sum(chamber_mask > 0)}")
        print(f"Coverage: {np.sum(chamber_mask > 0) / chamber_mask.size * 100:.2f}%")
        print(f"Prompt points: {len(points)}")
        
        # Save stats
        stats = {
            'image_shape': image.shape,
            'mask_shape': chamber_mask.shape,
            'total_pixels': int(chamber_mask.size),
            'segmented_pixels': int(np.sum(chamber_mask > 0)),
            'coverage_percent': float(np.sum(chamber_mask > 0) / chamber_mask.size * 100),
            'prompt_points': points.tolist(),
            'prompt_labels': labels.tolist()
        }
        
        import json
        with open(output_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to: {output_dir / 'stats.json'}")
        
        print(f"\n✅ Segmentation test completed successfully!")
        print(f"All outputs saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting segmentation integration test...")
    success = test_segment_integration()
    
    if success:
        print("\n🎉 Test passed!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)