"""
Author: B. Chen

Traditional thresholding algorithm implementations.
"""

import numpy as np
from skimage.filters import threshold_otsu, threshold_isodata
from ..interfaces import ThresholdMethod


class OtsuThreshold(ThresholdMethod):
    """Otsu's automatic threshold selection method."""
    
    def _calculate_threshold(self, image: np.ndarray) -> float:
        """Calculate threshold using Otsu's method."""
        return float(threshold_otsu(image))


class IsodataThreshold(ThresholdMethod):
    """Isodata (iterative selection) threshold method."""
    
    def _calculate_threshold(self, image: np.ndarray) -> float:
        """Calculate threshold using Isodata method."""
        return float(threshold_isodata(image))


class MeanThreshold(ThresholdMethod):
    """Simple mean-based thresholding method."""
    
    def _calculate_threshold(self, image: np.ndarray) -> float:
        """Calculate threshold as image mean intensity."""
        return float(np.mean(image))