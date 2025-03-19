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



if __name__ == "__main__":
    image_path = 'example_datasets/patient1/image1.png'
    base_output = 'output/output_one_image'

    image_name, image_ext = os.path.splitext(os.path.basename(image_path))
    image_ext = image_ext.lstrip('.')
    output_folder = f'{base_output}/{image_name}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    


    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="./configs/default_settings.yaml")
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    sam_checkpoint = cfg.sam_model.ckpt
    model_type = cfg.sam_model.model_type
    model_path = cfg.cell_classifier.ckpt
    # Check if CUDA is available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L = cfg.cell_classifier.size_L
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((L, L)), transforms.ToTensor()])

    params = {
        "threshold_lambda": cfg.infer_threshold.threshold_lambda,
        "lower_bound": cfg.infer_threshold.lower_bound,
        "upper_bound": cfg.infer_threshold.upper_bound,
        "device": device
    }
    print(f"The device is {device}")
    print(f"Loading Vision Foundation Model ...")
    # Using Vit-H SAM for AC area segmentation
    predictor = load_sam_predictor(sam_checkpoint = sam_checkpoint, model_type=model_type)

    # load Spatial Attention Network
    model = Net(size_img=L)  # Initialize model
    # Load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Detecting Cell ...")
    results = asoct_cell_detection(image_path, predictor, model, transform, params)

    chamber_mask = results['chamber_mask']
    point_prompts = results['point_prompts']
    candidate_cell_mask = results['candidate_cell_mask']
    cell_mask = results['cell_mask']
    cell_centroids = get_centroids(cell_mask)
    candidate_cell_centroids = get_centroids(candidate_cell_mask)

    green_color = '#00FF00'
    red_color = '#FF0000'
    image_with_cell_boxes = draw_square(image_path, cell_centroids, box_outline=green_color)
    image_with_candidate_cell_boxes = draw_square(image_path, candidate_cell_centroids, box_outline=red_color)

    csv_output_path = f'{output_folder}/centroids.csv'
    save_centroids_to_csv(cell_centroids, candidate_cell_centroids, csv_output_path)

    # Save the masks as an image file
    cv2.imwrite(f'{output_folder}/chamber_mask.png', chamber_mask)
    image_with_cell_boxes.save(f'{output_folder}/image_with_cell_boxes.png')
    image_with_candidate_cell_boxes.save(f'{output_folder}/image_with_MiRP_boxes.png')

    # Extract masked regions with transparency
    chamber_mask_path = f'{output_folder}/AC_mask.png'

    # Create a semi-transparent blue overlay of the chamber mask
    overlay_transparent_mask(
        image_path,
        chamber_mask,
        f'{output_folder}/image_with_AC_mask.png',
        color=(255, 0, 0),  # Blue in BGR format
        alpha=0.5          # 30% opacity
    )
    print('Finished!')