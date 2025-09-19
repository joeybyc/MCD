"""
Author: B. Chen

Factory classes for creating classification components.
"""

from typing import Dict, Type
from .interfaces import BaseClassifier, ClassificationModelWrapper
from .implementations.cell_classifier import CellClassifier


class ClassifierFactory:
    """Factory for creating classifier instances."""
    
    _classifiers: Dict[str, Type[BaseClassifier]] = {
        'cell': CellClassifier,
    }
    
    @classmethod
    def create_classifier(cls, classifier_type: str, **kwargs) -> BaseClassifier:
        """
        Create a classifier instance with injected dependencies.
        
        Args:
            classifier_type: Type of classifier ('cell')
            **kwargs: Dependencies required by the classifier
                     For cell: model_wrapper
            
        Returns:
            Classifier instance
            
        Raises:
            ValueError: If classifier type is not supported or dependencies are missing
        """
        if classifier_type not in cls._classifiers:
            available = ', '.join(cls._classifiers.keys())
            raise ValueError(f"Unknown classifier: {classifier_type}. Available: {available}")
        
        classifier_class = cls._classifiers[classifier_type]
        
        try:
            return classifier_class(**kwargs)
        except TypeError as e:
            raise ValueError(f"Invalid dependencies for {classifier_type}: {e}")
    
    @classmethod
    def register_classifier(cls, name: str, classifier_class: Type[BaseClassifier]) -> None:
        """Register a new classifier type."""
        cls._classifiers[name] = classifier_class
    
    @classmethod
    def get_available_classifiers(cls) -> list:
        """Get list of available classifier types."""
        return list(cls._classifiers.keys())