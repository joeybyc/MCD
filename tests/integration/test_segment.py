#!/usr/bin/env python3
"""
Integration test for segmentation functionality.
Tests the complete pipeline from image input to mask output.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from asoct_mcd.model_management import ModelRegistry, SAMModelAdapter
from asoct_mcd.prompt_generation import PromptGeneratorFactory
from asoct_mcd.segmentation import SegmentorFactory
from asoct_mcd.image_processing import overlay_mask


def test_segmentation():
    """Test complete segmentation pipeline."""
    
    # Setup paths
    test_data_dir = Path("tests/data")
    image_path = test_data_dir / "sample_asoct.png"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    # Create output directory
    image_name = image_path.stem
    output_dir = test_data_dir / f"{image_name}_segment_output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Testing segmentation with image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Load and prepare image
        print("\n=== Step 1: Loading Image ===")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        
        # Save original image copy
        cv2.imwrite(str(output_dir / "01_original_image.png"), image)
        
        # Step 2: Initialize model registry and ensure models are available
        print("\n=== Step 2: Model Management Setup ===")
        registry = ModelRegistry()
        
        print("Available models:", registry.get_available_models())
        print("Ensuring SAM model is loaded...")
        
        # This will download and load the model if needed
        sam_model = registry.get_model('sam_vit_b', auto_load=True)
        print(f"SAM model loaded: {sam_model.is_loaded}")
        print("Model info:", sam_model.get_model_info())
        
        # Step 3: Create prompt generator
        print("\n=== Step 3: Prompt Generation ===")
        prompt_generator = PromptGeneratorFactory.create_generator('i2acp')
        
        # Generate prompts
        points, labels = prompt_generator.generate(image)
        print(f"Generated {len(points)} prompt points: {points}")
        print(f"Point labels: {labels}")
        
        # Visualize prompt points on image
        prompt_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i, (point, label) in enumerate(zip(points, labels)):
            x, y = int(point[0]), int(point[1])
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, red for negative
            cv2.circle(prompt_vis, (x, y), 5, color, -1)
            cv2.putText(prompt_vis, f"P{i+1}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(output_dir / "02_prompt_points.png"), prompt_vis)
        
        # Step 4: Create segmentation components
        print("\n=== Step 4: Segmentation Setup ===")
        
        # Create model adapter
        sam_adapter = SAMModelAdapter('sam_vit_b')
        
        # Create prompt generator wrapper (adapt to segmentation interface)
        class PromptGeneratorWrapper:
            def __init__(self, generator):
                self.generator = generator
            
            def generate(self, image):
                return self.generator.generate(image)
        
        prompt_wrapper = PromptGeneratorWrapper(prompt_generator)
        
        # Create zero-shot segmentor
        segmentor = SegmentorFactory.create_segmentor(
            'zero_shot',
            model_wrapper=sam_adapter,
            prompt_generator=prompt_wrapper
        )
        
        # Step 5: Perform segmentation
        print("\n=== Step 5: Segmentation ===")
        mask = segmentor.segment(image)
        
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")
        print(f"Mask coverage: {np.sum(mask > 0) / mask.size * 100:.2f}%")
        
        # Save mask
        cv2.imwrite(str(output_dir / "03_chamber_mask.png"), mask)
        
        # Step 6: Create visualization overlay
        print("\n=== Step 6: Creating Overlay Visualization ===")
        overlay_path = str(output_dir / "04_overlay_visualization.png")
        overlay_result = overlay_mask(
            str(image_path),
            mask,
            overlay_path,
            color=(255, 0, 0),  # Red overlay
            alpha=0.5
        )
        
        # Step 7: Save processing summary
        print("\n=== Step 7: Saving Summary ===")
        summary_path = output_dir / "processing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("AS-OCT Segmentation Test Results\n")
            f.write("================================\n\n")
            f.write(f"Input image: {image_path}\n")
            f.write(f"Image dimensions: {image.shape}\n")
            f.write(f"Prompt points: {len(points)}\n")
            f.write(f"Point coordinates: {points.tolist()}\n")
            f.write(f"Mask coverage: {np.sum(mask > 0) / mask.size * 100:.2f}%\n")
            f.write(f"Output files:\n")
            f.write(f"  - 01_original_image.png: Original input image\n")
            f.write(f"  - 02_prompt_points.png: Visualization of I2ACP prompt points\n")
            f.write(f"  - 03_chamber_mask.png: Binary segmentation mask\n")
            f.write(f"  - 04_overlay_visualization.png: Mask overlay on original image\n")
        
        print(f"\n✓ Segmentation test completed successfully!")
        print(f"✓ All outputs saved to: {output_dir}")
        print(f"✓ Check processing_summary.txt for details")
        
        # Step 8: Model cleanup
        print("\n=== Step 8: Cleanup ===")
        model_health = registry.health_check()
        print("Model health before cleanup:", model_health)
        
        # Optional: unload models to free memory
        # registry.unload_all_models()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("AS-OCT Segmentation Integration Test")
    print("=" * 50)
    
    success = test_segmentation()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)