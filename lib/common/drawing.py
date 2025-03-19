from PIL import Image, ImageDraw
import os
import cv2
import numpy as np
# def draw_square(image_path, cell_centroids, box_size = 10):
#     half_box_size = box_size // 2
#     image_with_cell_boxes = Image.open(image_path)
#     draw = ImageDraw.Draw(image_with_cell_boxes)

#     # Extract each centroid in the list, convert to integers, and draw the square
#     for x, y in cell_centroids:
#         x, y = int(x), int(y)
#         box = [x-half_box_size, y-half_box_size, x+half_box_size, y+half_box_size]  # Creates a box with length 20 centered at (x, y)
#         draw.rectangle(box, outline='red', width=2)
#     return image_with_cell_boxes


def draw_square(image_path, cell_centroids, box_size=10, box_width=2, box_outline='red', cropped_output_folder=None):
    """
    Draw squares around detected cells and optionally save cropped cell images.
    
    Args:
        image_path (str): Path to the input image
        cell_centroids (list): List of (x, y) coordinates for cell centers
        box_size (int): Size of the box to draw (default: 10)
        cropped_output_folder (str): Path to save cropped images. If None, no cropping will be performed
    
    Returns:
        Image: Image with drawn boxes
    """
    half_box_size = box_size // 2
    # Open the image and create a copy for drawing boxes
    original_image = Image.open(image_path)
    image_with_cell_boxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_cell_boxes)

    # If output folder is provided, ensure it exists
    if cropped_output_folder is not None:
        # Create directory if it doesn't exist
        if not os.path.exists(cropped_output_folder):
            os.makedirs(cropped_output_folder)

    # Process each cell centroid
    for idx, (x, y) in enumerate(cell_centroids):
        x, y = int(x), int(y)
        box = [x-half_box_size, y-half_box_size, x+half_box_size, y+half_box_size]
        
        # Draw rectangle on the image
        draw.rectangle(box, outline=box_outline, width=box_width)
        
        # If output folder is provided, save the cropped region
        if cropped_output_folder is not None:
            # Ensure crop box stays within image boundaries
            crop_box = [
                max(0, box[0]),
                max(0, box[1]),
                min(original_image.width, box[2]),
                min(original_image.height, box[3])
            ]
            
            # Crop and save the cell image
            cropped_image = original_image.crop(crop_box)
            crop_path = os.path.join(cropped_output_folder, f'cell_{idx}.png')
            cropped_image.save(crop_path, 'PNG')
            # cropped_image.save(crop_path, 'JPEG')

    return image_with_cell_boxes


def overlay_transparent_mask(image_path, mask, output_path, color=(255, 0, 0), alpha=0.3):
    """
    Overlay a semi-transparent colored mask on the original image.
    
    Args:
        image_path (str or numpy.ndarray): Path to the original image or image array
        mask (numpy.ndarray): Binary mask array where True/1 indicates the region to be colored
        output_path (str): Path where the output image will be saved
        color (tuple): Color of the overlay in BGR format, default is blue (255,0,0)
        alpha (float): Opacity of the overlay, between 0 and 1, higher means more opaque
    
    Returns:
        numpy.ndarray: The resulting image with the colored overlay
    """
    # Load the original image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    
    # Ensure mask is boolean type
    mask = mask.astype(bool)
    
    # Create a colored mask with the same shape as the original image
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = color
    
    # Blend the original image with the colored mask
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    # Save the result
    cv2.imwrite(output_path, overlay)
    
    return overlay