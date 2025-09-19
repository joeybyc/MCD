"""
Integration test for MCD pipeline comparing with traditional threshold methods.

Run from project root directory:
python -m pytest tests/integration/test_all.py -v -s
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from asoct_mcd.pipeline import MCDPipelineBuilder
from asoct_mcd.threshold_methods import ThresholdFactory
from asoct_mcd.image_processing import get_centroids, draw_rectangles, overlay_mask, intersect_masks
from asoct_mcd.prompt_generation import PromptGeneratorFactory


def test_mcd_integration():
    """Test MCD pipeline and compare with threshold methods."""
    
    # Setup paths
    test_dir = Path(__file__).parent.parent
    image_path = test_dir / "data" / "image1.png"
    
    # Create output directory
    image_name = image_path.stem
    output_dir = test_dir / "data" / f"{image_name}_all_output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"Cannot load image from {image_path}"
    
    # 1. MCD Pipeline Detection
    print("Running MCD pipeline...")
    pipeline = MCDPipelineBuilder().build()
    mcd_result = pipeline.detect_cells(str(image_path))
    
    print(f"MCD detected {mcd_result.cell_count} cells")
    print(f"MCD found {mcd_result.candidate_count} candidates")
    
    # 2. Traditional Threshold Methods
    print("Running threshold comparisons...")
    
    # Get threshold method and lambda from MCD config
    threshold_method = ThresholdFactory.create_threshold("otsu")
    lambda_factor = 0.83
    
    # Pure threshold (no lambda)
    threshold_mask_full = threshold_method.apply_threshold(image)
    threshold_mask = intersect_masks(mcd_result.chamber_mask, threshold_mask_full)
    threshold_locations = get_centroids(threshold_mask)
    
    # Threshold with lambda
    threshold_lambda_mask_full = threshold_method.apply_threshold(image, lambda_factor=lambda_factor)
    threshold_lambda_mask = intersect_masks(mcd_result.chamber_mask, threshold_lambda_mask_full)
    threshold_lambda_locations = get_centroids(threshold_lambda_mask)
    
    print(f"Pure threshold detected {len(threshold_locations)} objects")
    print(f"Threshold*lambda detected {len(threshold_lambda_locations)} objects")
    
    # 3. I2ACP Prompt Generation
    print("Generating I2ACP prompts...")
    prompt_generator = PromptGeneratorFactory.create_generator("i2acp")
    prompts, labels = prompt_generator.generate(image)
    
    # 4. Save Results
    print("Saving results...")
    
    # Save detection visualizations
    _save_detection_results(
        str(image_path), output_dir,
        mcd_result.cell_locations, threshold_locations, threshold_lambda_locations
    )
    
    # Save masks
    _save_masks(output_dir, mcd_result.chamber_mask, threshold_mask, threshold_lambda_mask)
    
    # Save mask overlays
    _save_mask_overlays(str(image_path), output_dir, mcd_result.chamber_mask)
    
    # Save prompt visualization
    _save_prompt_visualization(str(image_path), output_dir, prompts)
    
    print(f"All results saved to {output_dir}")
    
    # Assertions
    assert mcd_result.cell_count >= 0
    assert mcd_result.chamber_mask.shape == image.shape
    assert len(prompts) > 0


def _save_detection_results(image_path, output_dir, mcd_locations, threshold_locations, threshold_lambda_locations):
    """Save detection results with colored boxes."""
    
    # MCD results (green boxes)
    mcd_image = draw_rectangles(image_path, mcd_locations, box_size=10, color="#00FF00")
    mcd_image.save(output_dir / "01_mcd_detection.png")
    
    # Pure threshold results (blue boxes)
    threshold_image = draw_rectangles(image_path, threshold_locations, box_size=10, color="#0000FF")
    threshold_image.save(output_dir / "02_threshold_detection.png")
    
    # Threshold*lambda results (red boxes)
    threshold_lambda_image = draw_rectangles(image_path, threshold_lambda_locations, box_size=10, color="#FF0000")
    threshold_lambda_image.save(output_dir / "03_threshold_lambda_detection.png")


def _save_masks(output_dir, chamber_mask, threshold_mask, threshold_lambda_mask):
    """Save binary masks."""
    
    cv2.imwrite(str(output_dir / "04_chamber_mask.png"), chamber_mask)
    cv2.imwrite(str(output_dir / "05_threshold_mask.png"), threshold_mask)
    cv2.imwrite(str(output_dir / "06_threshold_lambda_mask.png"), threshold_lambda_mask)


def _save_mask_overlays(image_path, output_dir, chamber_mask):
    """Save mask overlays on original image."""
    
    # Chamber mask overlay (red)
    overlay_mask(
        image_path, 
        chamber_mask, 
        str(output_dir / "07_chamber_overlay.png"),
        color=(0, 0, 255),  # Red in BGR
        alpha=0.3
    )


def _save_prompt_visualization(image_path, output_dir, prompts):
    """Save I2ACP prompt points visualization."""
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw prompt points (yellow circles)
    for point in prompts:
        x, y = int(point[0]), int(point[1])
        radius = 5
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    fill="yellow", outline="red", width=2)
    
    image.save(output_dir / "08_i2acp_prompts.png")


if __name__ == "__main__":
    test_mcd_integration()