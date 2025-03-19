import cv2
from skimage.filters import threshold_isodata
from skimage.filters import threshold_otsu


def threshold_method(image, return_thresh = False, lambda_setting = 1.0):
    """
    Threshold an image using the isodata method with OpenCV.

    Parameters:
    - image_path: str, path to the greyscale image to be thresholded.
    - return_thresh: Boolean, To decide if the function will return thresh_value.
    Returns:
    - mask: numpy.ndarray, mask of the image with 0 for background and 255 for the object.
    """
    # Use the isodata method to find the optimal threshold value

    thresh_value = threshold_otsu(image)
    thresh_value = thresh_value * lambda_setting
    # Apply the threshold to create a binary mask
    _, mask = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    if return_thresh:
      return mask, thresh_value
    else:
      return mask