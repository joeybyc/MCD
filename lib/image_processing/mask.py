import cv2
import numpy as np
from skimage import measure


def select_mask_based_on_points(mask_image, points):
    """
    Keeps objects in the mask image that contain specified points.

    Parameters:
    - mask_image (numpy.ndarray): Binary mask image with objects marked as 255 and background as 0.
    - points (numpy.ndarray): Array of points in the format [[w1, h1], [w2, h2], ...].

    Returns:
    - numpy.ndarray: Filtered mask image with only the objects containing the points.
    """

    # Find all the contours in the mask image
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the selected contours
    selected_mask = np.zeros_like(mask_image)

    # Iterate through each contour
    for contour in contours:
        # Check if any point is inside the current contour
        for point in points:
            # Convert point to tuple format
            point_tuple = (int(point[0]), int(point[1]))

            # Check if the point is inside the contour
            if cv2.pointPolygonTest(contour, point_tuple, False) >= 0:
                # Draw the contour (fill it) on the selected_mask
                cv2.drawContours(selected_mask, [contour], -1, (255), thickness=cv2.FILLED)
                break  # Break the loop if at least one point is inside the contour

    return selected_mask

def merge_masks(mask1, mask2):
    """
    Merge two binary mask images (0 or 255) into one mask image.

    Parameters:
    - mask1: numpy.ndarray, the first binary mask image.
    - mask2: numpy.ndarray, the second binary mask image.

    Returns:
    - merged_mask: numpy.ndarray, the merged binary mask image.
    """
    # Check if both masks have the same shape
    if mask1.shape != mask2.shape:
        raise ValueError("The masks do not have the same dimensions. Please provide masks with the same dimensions.")

    # Ensure that both masks are of the same type, here assuming unsigned 8-bit integer
    if mask1.dtype != np.uint8 or mask2.dtype != np.uint8:
        raise ValueError("Masks are not of type uint8. Please provide masks with uint8 type.")

    # Merge the masks by applying a bitwise OR operation
    merged_mask = cv2.bitwise_or(mask1, mask2)

    return merged_mask

def filter_objects_by_size(mask, lower_bound=2, upper_bound=25):
    """
    Keep objects in a binary mask image that are within a specified size range.

    Parameters:
    - mask: numpy.ndarray, a binary mask image with objects set to 255 and the background set to 0.
    - lower_bound: int, the lower bound of object size to keep (in pixels).
    - upper_bound: int, the upper bound of object size to keep (in pixels).

    Returns:
    - filtered_mask: numpy.ndarray, the mask image with only the objects within the specified size range.
    """
    # Check if the mask is a binary image of type uint8
    if mask.dtype != np.uint8:
        raise ValueError("Mask is not of type uint8. Please provide a binary mask with uint8 type.")

    # Find connected components
    ret, labels = cv2.connectedComponents(mask)

    # Create an empty mask for the output
    filtered_mask = np.zeros_like(mask)
    # Check each label for its area and filter based on t1 and t2
    for label in range(1, ret):  # start from 1 to exclude background
        area = np.sum(labels == label)
        if lower_bound <= area <= upper_bound:
            filtered_mask[labels == label] = 255

    return filtered_mask


def extract_objects_by_area(area_mask, object_mask):
    """
    Extract objects from object_mask that are within the regions defined by area_mask.

    Parameters:
    - area_mask: numpy.ndarray, a binary mask defining the areas of interest.
    - object_mask: numpy.ndarray, a binary mask with objects that need to be extracted.

    Returns:
    - extracted_mask: numpy.ndarray, a binary mask with objects from object_mask that are within area_mask.
    """
    # Ensure that both masks are binary (0 or 255) and have the same shape
    if not np.all((area_mask == 0) | (area_mask == 255)) or not np.all((object_mask == 0) | (object_mask == 255)):
        raise ValueError("One or both masks are not binary. Please provide binary masks with values 0 or 255.")
    if area_mask.shape != object_mask.shape:
        raise ValueError("Masks do not have the same dimensions. Please provide masks with the same dimensions.")

    # Perform a bitwise AND operation to extract objects within the area defined by area_mask
    extracted_mask = cv2.bitwise_and(object_mask, area_mask)

    return extracted_mask

def find_largest_object_mask(mask_image):
    """
    Post-processing function to isolate the largest object in a mask image.

    Parameters:
    - mask_image: A binary mask image with objects marked in white (255) and the background in black (0).
    Return:
    - A new mask image with only the largest object, including its internal holes.

    Raises:
    ValueError: If no contours are found in the mask, indicating no objects are present.
    """
    # Check if the image is already in binary format (only 0 and 255 values)
    if not np.array_equal(mask_image, mask_image.astype(bool) * 255):
        raise ValueError("The mask image must be a binary image with values 0 and 255.")

    # Label connected regions of an integer array using measure.label from skimage
    labels = measure.label(mask_image, connectivity=2, background=0)

    # Find the largest connected component
    largest_label = 0
    largest_area = 0
    for region in measure.regionprops(labels):
        if region.area > largest_area:
            largest_area = region.area
            largest_label = region.label

    # Check if any object was found
    if largest_label == 0:
        raise ValueError("No objects found in the mask.")

    # Create a mask with the largest connected component
    largest_mask = np.zeros(mask_image.shape, dtype="uint8")
    largest_mask[labels == largest_label] = 255

    return largest_mask



def find_two_largest_objects(mask_image, second_ratio=0.6):
    """
    Post-processing function to isolate the largest and potentially the second-largest object in a mask image.
    Parameters:
    - mask_image: A binary mask image with objects marked in white (255) and the background in black (0).
    - second_ratio: The minimum size ratio of the second largest object to the largest object to include it in the output.
    Returns: 
    - A new mask image with the largest and potentially the second-largest object, including their internal holes.

    Raises:
    ValueError: If no contours are found in the mask, indicating no objects are present.
    """
    if not np.array_equal(mask_image, mask_image.astype(bool) * 255):
        raise ValueError("The mask image must be a binary image with values 0 and 255.")

    labels = measure.label(mask_image, connectivity=2, background=0)

    # Initialize variables to find the two largest components
    largest_label = 0
    second_largest_label = 0
    largest_area = 0
    second_largest_area = 0

    # Iterate through detected regions to find the two largest
    for region in measure.regionprops(labels):
        if region.area > largest_area:
            # Update second largest
            second_largest_area = largest_area
            second_largest_label = largest_label
            # Update largest
            largest_area = region.area
            largest_label = region.label
        elif region.area > second_largest_area:
            second_largest_area = region.area
            second_largest_label = region.label

    if largest_label == 0:
        raise ValueError("No objects found in the mask.")

    largest_mask = np.zeros(mask_image.shape, dtype="uint8")
    largest_mask[labels == largest_label] = 255

    # Check if the second largest object meets the criteria and add it to the mask
    if second_largest_area >= largest_area * second_ratio:
        largest_mask[labels == second_largest_label] = 255

    return largest_mask


# Function to process mask image and find object centroids
def get_centroids(mask_image):
    """
        Finds the centroids of objects in a binary mask image.
        Args:
            mask_image (numpy.ndarray): A binary mask image where objects are marked with non-zero values.
        Returns:
            list: A list of tuples representing the centroids of the objects in the form (centroid_x, centroid_y).
    """
    num_labels, labels = cv2.connectedComponents(mask_image)
    centroids = []
    for i in range(1, num_labels):  # Start from 1 to exclude the background
        ys, xs = np.where(labels == i)
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
        centroids.append((centroid_x, centroid_y))
    return centroids



