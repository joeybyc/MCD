"""
Author: B. Chen

Abstract base classes for model management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Base interface for all models."""
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': getattr(self, 'name', 'unknown'),
            'is_loaded': self.is_loaded
        }


class BaseModelLoader(ABC):
    """Base interface for model loaders."""
    
    @abstractmethod
    def download_if_needed(self, config: Any) -> str:
        """Download model file if needed, return local path."""
        pass