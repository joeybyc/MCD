import numpy as np
import cv2
from lib.image_processing.denoising import nlm_denoising
from lib.image_processing.mask import find_largest_object_mask, select_mask_based_on_points, find_two_largest_objects

def get_centroid_from_mask_image(mask_image):
    """
    Find the centroid of the white pixels in a binary mask image.
    
    Parameters:
    - mask_image: A binary image where the object's pixels are white (255) and the background is black (0).
    Returns:
    - A tuple of (centroid_x, centroid_y) which are the coordinates of the centroid.

    Usage:
    # Assume mask_image is a binary image with white pixels forming the object of interest
    centroid_x, centroid_y = get_centroid_from_mask_image(mask_image)
    """
    # Find the coordinates of the pixels equal to 255
    rows, cols = np.where(mask_image == 255)

    # If no white pixels are found, return None
    if rows.size == 0 or cols.size == 0:
        return None, None

    # Calculate the centroid coordinates
    centroid_x = int(np.mean(cols))
    centroid_y = int(np.mean(rows))

    return centroid_x, centroid_y

def get_centroid_from_image(image, pre_process_method=None, get_mask_method=None, post_process_method=None, return_mask=False):
    """The implementation of the Heuristic Prompt Generation (HPG) algorithm

    Process an image and find the centroid of an object by applying optional pre-processing,
    mask extraction, and post-processing functions.


    Parameters:
    - image: The input image to process (greyscale).
    - pre_process_method: Optional function to pre-process the image.
    - get_mask_method: Optional function to extract the binary mask from the image.
    - post_process_method: Optional function to post-process the binary mask.
    - return_mask: Boolean flag to decide if the binary mask should be returned.
    Returns:
    - A tuple (centroid_x, centroid_y) and optionally the binary mask image.

    Usage:
    # To get the centroid with all default methods
    centroid_x, centroid_y = get_centroid_from_image(image)

    # To get the centroid with a custom get_mask_method and also return the binary mask
    def custom_mask_method(img):
        # Custom mask extraction logic
        pass
    centroid_x, centroid_y, mask = get_centroid_from_image(image, get_mask_method=custom_mask_method, return_mask=True)
    """
    if pre_process_method is not None:
        image = pre_process_method(image)
    if get_mask_method is not None:
        mask_image = get_mask_method(image)
    else:
        # Calculate the average threshold
        threshold = np.mean(image)
        # Create the binary mask
        _, mask_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    if post_process_method is not None:
        mask_image = post_process_method(mask_image)

    centroid_x, centroid_y = get_centroid_from_mask_image(mask_image)

    if centroid_x is None or centroid_y is None:
        raise ValueError("Centroid could not be found. Make sure the object is present in the image.")

    if return_mask:
        return centroid_x, centroid_y, mask_image
    else:
        return centroid_x, centroid_y
    


def get_prompt_point(image, centroid_x, centroid_y, strategy='method1'):
    """
    Generate points around the centroid based on the specified strategy and label them.

    Parameters:
    - image: The input image from which width and height are derived.
    - centroid_x: The x-coordinate of the centroid.
    - centroid_y: The y-coordinate of the centroid.
    - strategy: The strategy to use for generating points. Defaults to 'method1'.
    
    Returns:
    - A tuple of numpy arrays: the first containing points, the second containing labels.

    Usage:
    # Assuming image is an image matrix and centroid_x, centroid_y are known
    points, labels = get_prompt_point(image, centroid_x, centroid_y, strategy='method1')
    """
    # Extract the width and height from the image shape
    height, width = image.shape[:2]

    # Initialize input_point and input_label arrays
    input_point = np.array([])
    input_label = np.array([])

    # Generate points based on the selected strategy
    if strategy == 'method1':
        # Calculate new points by offsetting the centroid by 2% of the image's width
        w1, h1 = [int(centroid_x + width * 0.02), int(centroid_y)]
        w2, h2 = [int(centroid_x - width * 0.02), int(centroid_y)]
        input_point = np.array([[w1, h1], [w2, h2]])
        input_label = np.array([1, 1])
    elif strategy == 'method2':
      # Calculate new points by offsetting the centroid by 2% of the image's width
        w1, h1 = [int(centroid_x), int(centroid_y)]
        input_point = np.array([[w1, h1]])
        input_label = np.array([1])

    return input_point, input_label

def get_chamber_mask(image, predictor, input_point, input_label):
    """
    Apply a prediction model to an image to generate a mask for a specific chamber or region.
    The mask is then converted to a binary image with values 0 and 255.

    Parameters:
    - image: The image to process.
    - predictor: The prediction model that generates the mask.
    - input_point: An array of input points for the predictor.
    - input_label: The labels associated with the input points.
    
    Returns:
    - A binary mask as a numpy array with values 0 and 255.

    """
    # Ensure the image is in the correct format for the predictor (RGB)
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Set the image for the predictor
    predictor.set_image(image)

    # Perform the prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Squeeze the single-dimensional entries from the shape of an array
    mask = np.squeeze(masks)

    # Convert mask to binary format with values 0 and 255
    image_mask = (mask > 0).astype(np.uint8) * 255

    return image_mask


def extract_chamber_mask(image, predictor, return_all=False):
    """The implementation of Chamber Segmentation Module (CSM)
    
    Apply all procedure together to get the chamber mask of an image.

    Parameters:
    - image: The image to process. (image is the greyscale image loaded by cv2)
    - predictor: The SAM model
    - return_all: Boolean flag to decide if the mid-stage data should be returned.
    
    Returns:
    - If return_all=False:
        - chamber_mask: A binary mask as a numpy array with values 0 and 255.
    - If return_all=True:
        - chamber_mask: A binary mask as a numpy array with values 0 and 255.
        - mask_image_anterior_segment: Binary mask of the anterior segment of the eye.
        - point_prompts: The generated prompt points.
        - prompt_labels: The labels corresponding to the prompt points.
        - chamber_mask_candidates: The candidate chamber masks generated by the SAM model.

    """
    # centroid_x, centroid_y, mask_image_anterior_segment = get_centroid_from_image(image, pre_process_method=nlm_denoising, post_process_method=find_largest_object_mask, return_mask=True)
    # replace find_largest_object_mask with post_process_find_two_largest_objects
    centroid_x, centroid_y, mask_image_anterior_segment = get_centroid_from_image(image, pre_process_method=nlm_denoising, post_process_method=find_two_largest_objects, return_mask=True)
    point_prompts, prompt_labels = get_prompt_point(image, centroid_x, centroid_y)
    chamber_mask_candidates = get_chamber_mask(nlm_denoising(image), predictor, point_prompts, prompt_labels)
    chamber_mask = select_mask_based_on_points(chamber_mask_candidates, point_prompts)

    if return_all:
        return chamber_mask, mask_image_anterior_segment, point_prompts, prompt_labels, chamber_mask_candidates
    else:
        return chamber_mask
