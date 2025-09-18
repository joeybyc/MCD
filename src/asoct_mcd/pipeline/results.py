"""
Author: B. Chen

Result classes for pipeline outputs.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class CellDetectionResult:
    """Result from cell detection pipeline for a single image."""
    
    image_path: str
    chamber_mask: np.ndarray
    cell_mask: np.ndarray
    cell_locations: List[Tuple[float, float]]
    
    # Optional intermediate results for debugging
    threshold_mask: Optional[np.ndarray] = None
    candidate_mask: Optional[np.ndarray] = None
    prompt_points: Optional[np.ndarray] = None
    
    @property
    def cell_count(self) -> int:
        """Number of detected cells."""
        return len(self.cell_locations)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            'image_path': self.image_path,
            'cell_count': self.cell_count,
            'chamber_coverage': float(np.sum(self.chamber_mask > 0) / self.chamber_mask.size),
            'cell_coverage': float(np.sum(self.cell_mask > 0) / self.cell_mask.size)
        }


@dataclass 
class BatchDetectionResult:
    """Results from batch cell detection."""
    
    results: List[CellDetectionResult]
    
    @property
    def total_cells(self) -> int:
        """Total number of cells across all images."""
        return sum(result.cell_count for result in self.results)
    
    @property
    def image_count(self) -> int:
        """Number of processed images."""
        return len(self.results)
    
    def get_batch_summary(self) -> dict:
        """Get batch processing summary."""
        if not self.results:
            return {'image_count': 0, 'total_cells': 0}
        
        cell_counts = [result.cell_count for result in self.results]
        
        return {
            'image_count': self.image_count,
            'total_cells': self.total_cells,
            'avg_cells_per_image': self.total_cells / self.image_count,
            'max_cells': max(cell_counts),
            'min_cells': min(cell_counts)
        }