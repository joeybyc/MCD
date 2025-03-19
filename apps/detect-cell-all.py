import os
import argparse
from torchvision import transforms
import torch
import cv2

from lib.model.sam_model import load_sam_predictor
from lib.image_processing.loading import load_image
from lib.chember_segmentation.sam_detection import extract_chamber_mask
from lib.image_processing.thresholding import threshold_method
from lib.image_processing.mask import merge_masks, filter_objects_by_size, extract_objects_by_area, get_centroids
from lib.cell_detection.filter import filter_candidate_cells
from lib.model.spatial_attention_network import Net

from lib.common.config import cfg
from lib.common.drawing import draw_square, overlay_transparent_mask

import shutil
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple



def asoct_cell_detection(image_path, predictor, model, transform, params):
    threshold_lambda = params["threshold_lambda"]
    lower_bound = params["lower_bound"]
    upper_bound = params["upper_bound"]
    device = params["device"]

    # Load image
    image = load_image(image_path)

    #========================== Field-of-Focus Module (Start)==============================================================================================================
    # Extract chamber masks and related information from the image using the predictor
    chamber_mask, mask_image_anterior_segment, point_prompts, prompt_labels, chamber_mask_candidates = extract_chamber_mask(image, predictor, return_all=True)
    #========================== Field-of-Focus Module (End)================================================================================================================

    #========================== Fine-grained Object Detection Module (Start)===============================================================================================

    image_threshold, _ = threshold_method(image, return_thresh=True, lambda_setting=threshold_lambda)
    merge_image_mask = merge_masks(mask_image_anterior_segment, image_threshold)
    detected_objects = filter_objects_by_size(merge_image_mask, lower_bound=lower_bound, upper_bound=upper_bound)
    candidate_cell_mask = extract_objects_by_area(chamber_mask, detected_objects)
    cell_mask = filter_candidate_cells(image, candidate_cell_mask, model, device, transform=transform)
    #========================== Fine-grained Object Detection Module (End)===================================================================================================

    results = {
        'chamber_mask': chamber_mask,
        'mask_image_anterior_segment': mask_image_anterior_segment,
        'point_prompts': point_prompts,
        'chamber_mask_candidates': chamber_mask_candidates,
        'candidate_cell_mask': candidate_cell_mask,
        'cell_mask': cell_mask
    }
    return results


def save_centroids_to_csv(cell_centroids, candidate_cell_centroids, output_path):
    """
    Save cell centroids and candidate cell centroids to a CSV file.
    
    Args:
        cell_centroids (list): List of tuples containing (x, y) coordinates of cell centroids
        candidate_cell_centroids (list): List of tuples containing (x, y) coordinates of candidate cell centroids
        output_path (str): Path where the CSV file will be saved
    """
    import csv
    
    data = [
        ['name', 'points'],
        ['cell_centroids', str(cell_centroids)],
        ['candidate_cell_centroids', str(candidate_cell_centroids)]
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)




# ===================================== Process all images ============================================================
def process_single_image(
    image_path: str,
    output_base: str,
    predictor,
    model,
    transform,
    params: Dict
) -> Optional[pd.DataFrame]:
    """
    Process a single image and update results DataFrame
    
    Args:
        image_path (str): Path to input image
        output_base (str): Base output directory
        predictor: SAM predictor model
        model: Cell detection model
        transform: Image transformation pipeline
        params (dict): Processing parameters
        
    """
    try:
        # Extract image information
        patient_id = os.path.basename(os.path.dirname(image_path))
        # Create output directory structure
        patient_folder = os.path.join(output_base, patient_id)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_folder = os.path.join(patient_folder, image_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Copy original image
        shutil.copy2(image_path, os.path.join(output_folder, os.path.basename(image_path)))
        
        # Process image using existing function
        results = asoct_cell_detection(image_path, predictor, model, transform, params)
        
        # Extract results and save outputs
        chamber_mask = results['chamber_mask']
        point_prompts = results['point_prompts']
        candidate_cell_mask = results['candidate_cell_mask']
        cell_mask = results['cell_mask']
        cell_centroids = get_centroids(cell_mask)
        candidate_cell_centroids = get_centroids(candidate_cell_mask)
        
        # Save visualizations and results
        green_color = '#00FF00'
        red_color = '#FF0000'
        image_with_cell_boxes = draw_square(image_path, cell_centroids, box_outline=green_color)
        image_with_candidate_cell_boxes = draw_square(image_path, candidate_cell_centroids, box_outline=red_color)
        
        # Save outputs
        cv2.imwrite(os.path.join(output_folder, 'chamber_mask.png'), chamber_mask)
        image_with_cell_boxes.save(os.path.join(output_folder, 'image_with_cell_boxes.png'))
        image_with_candidate_cell_boxes.save(os.path.join(output_folder, 'image_with_MiRP_boxes.png'))
        
        # Save centroids
        save_centroids_to_csv(
            cell_centroids,
            candidate_cell_centroids,
            os.path.join(output_folder, 'centroids.csv')
        )
        
        # Create mask overlay
        overlay_transparent_mask(
            image_path,
            chamber_mask,
            os.path.join(output_folder, 'image_with_AC_mask.png'),
            color=(255, 0, 0),
            alpha=0.5
        )
        
        print(f"Successfully processed: {image_path}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_image_directory(
    input_base: str,
    output_base: str,
    predictor,
    model,
    transform,
    params: Dict
) -> None:
    """
    Process all images in the input directory structure
    
    Args:
        input_base (str): Base input directory containing patient folders
        output_base (str): Base output directory for results
        predictor: SAM predictor model
        model: Cell detection model
        transform: Image transformation pipeline
        params (dict): Processing parameters
    """
    # Create output base directory
    os.makedirs(output_base, exist_ok=True)
    
    # Walk through input directory
    for root, dirs, files in os.walk(input_base):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                process_single_image(
                    image_path,
                    output_base,
                    predictor,
                    model,
                    transform,
                    params
                )

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="./configs/default_settings.yaml")
    parser.add_argument("-i", "--input_dir", type=str, default="example_datasets",
                        help="Input directory containing patient folders")
    parser.add_argument("-o", "--output_dir", type=str, default="output/output_MCD",
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Load configuration
    cfg.merge_from_file(args.config)
    
    # Setup device and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SAM model
    print("Loading SAM model...")
    predictor = load_sam_predictor(
        sam_checkpoint=cfg.sam_model.ckpt,
        model_type=cfg.sam_model.model_type
    )
    
    # Load Spatial Attention Network
    print("Loading Spatial Attention Network...")
    L = cfg.cell_classifier.size_L
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((L, L)),
        transforms.ToTensor()
    ])
    
    model = Net(size_img=L)
    model.load_state_dict(torch.load(cfg.cell_classifier.ckpt, map_location=device))
    model.eval()
    
    # Setup parameters
    params = {
        "threshold_lambda": cfg.infer_threshold.threshold_lambda,
        "lower_bound": cfg.infer_threshold.lower_bound,
        "upper_bound": cfg.infer_threshold.upper_bound,
        "device": device
    }
    
    # Process all images
    print("Starting batch processing...")
    process_image_directory(
        args.input_dir,
        args.output_dir,
        predictor,
        model,
        transform,
        params
    )
    print("Batch processing complete!")