from lib.chember_detection.sam_detection import extract_chamber_mask
from lib.image_processing.thresholding import threshold_method
from lib.image_processing.mask import merge_masks, filter_objects_by_size, extract_objects_by_area

def detect_cell(image, predictor, return_all=False, lower_bound=2, upper_bound=25, merge_cornea_flag = True, threshold_alpha = 1.0):
    """
    Detects cells in an image using a given prediction model and a series of image processing steps.

    Parameters:
    - image (numpy.ndarray): The input image in which cells are to be detected.
    - predictor (object): A model or function that predicts or identifies regions of interest in the image.
    - return_all (bool): Flag to determine whether to return additional intermediate processing results. Default is False.

    Returns:
    - numpy.ndarray: The mask image with detected cells, if return_all is False.
    - (optional) various processed images and masks, if return_all is True.
    """

    # Extract chamber masks and related information from the image using the predictor
    chamber_mask, mask_image_cornea, input_point, input_label, chamber_mask_all = extract_chamber_mask(image, predictor, return_all=return_all)

    # Apply isodata thresholding to the image and also return the threshold value
    image_threshold, T = threshold_method(image, return_thresh=True, alpha=threshold_alpha)

    if merge_cornea_flag:
        # Merge the cornea mask with the thresholded image
        merge_image_mask = merge_masks(mask_image_cornea, image_threshold)
    else:
        merge_image_mask = image_threshold

    # Filter objects in the merged mask by size to remove cornea-related regions
    mask_image_no_cornea = filter_objects_by_size(merge_image_mask, lower_bound=lower_bound, upper_bound=upper_bound)

    # Extract the final cell mask by considering only objects within the chamber mask area
    cell_mask = extract_objects_by_area(chamber_mask, mask_image_no_cornea)

    # Return either the cell mask alone or all intermediate results based on the 'return_all' flag
    if return_all:
        return cell_mask, chamber_mask, mask_image_cornea, input_point, input_label, chamber_mask_all
    else:
        return cell_mask