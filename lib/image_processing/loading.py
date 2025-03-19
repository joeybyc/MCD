import cv2

def load_image(image_path, to_greyscale=True):
    """
    Loads an image from the specified path and converts it to greyscale if required.

    Parameters:
    - image_path: The path to the image file to load.
    - to_greyscale: A boolean flag that determines if the image should be converted to greyscale.
                         Default value is True, which means convert to greyscale.
    Return:
    - The loaded image. If `to_greyscale` is True, the image will be in greyscale.

    Usage:
    # To load an image in greyscale
    greyscale_image = load_image('path_to_image.jpg', to_greyscale=True)

    # To load an image in its original color
    color_image = load_image('path_to_image.jpg', to_greyscale=False)
    """

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if conversion to greyscale is needed
    if to_greyscale and len(image.shape) == 3:  # If image has 3 channels, it's not greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image

def to_rgb(image):
    """
    Check if the image is greyscale, if so, convert it to RGB format.

    Parameters:
    - image: The input image, which may be greyscale or color.
    Return:
    - The image in RGB format.

    Usage:
    # Assuming 'image' is an image loaded by cv2
    rgb_image = ensure_rgb(image)
    """
    # Check if the image has two dimensions, which means it's greyscale
    if len(image.shape) == 2:
        # Convert the greyscale image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # The image is already in color, so we return it as is
        rgb_image = image

    return rgb_image