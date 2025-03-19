import cv2
def nlm_denoising(image):
    """
    Apply denoising to a grayscale image using the Non-Local Means Denoising algorithm.

    Parameters:
    - image: The input grayscale image to preprocess.
    Returns:
    - The denoised grayscale image.
    """
    # Check if the image is truly a grayscale image (single channel)
    if len(image.shape) != 2:
        raise ValueError("Input image is not a grayscale image. Please provide a grayscale image.")

    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    return denoised_image

def median_denoising(image):
    """
    Apply denoising to a grayscale image using the median denoising algorithm.

    Parameters:
    - image: numpy.ndarray, the input grayscale image to preprocess.

    Returns:
    - denoised_image: numpy.ndarray, the denoised grayscale image.
    """
    # Check if the image is truly a grayscale image (single channel)
    if len(image.shape) != 2:
        raise ValueError("Input image is not a grayscale image. Please provide a grayscale image.")

    # Apply Median Denoising
    denoised_image = cv2.medianBlur(image, 1)

    return denoised_image