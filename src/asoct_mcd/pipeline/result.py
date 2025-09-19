"""
Author: B. Chen

Result data classes for MCD pipeline output.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class CellDetectionResult:
    """Cell detection result containing locations and masks."""
    
    cell_locations: List[Tuple[float, float]]
    chamber_mask: np.ndarray
    candidate_locations: List[Tuple[float, float]] = field(default_factory=list)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cell_count(self) -> int:
        """Number of detected cells."""
        return len(self.cell_locations)
    
    @property
    def candidate_count(self) -> int:
        """Number of candidate regions."""
        return len(self.candidate_locations)
    
    def add_info(self, key: str, value: Any) -> None:
        """Add processing information."""
        self.processing_info[key] = value