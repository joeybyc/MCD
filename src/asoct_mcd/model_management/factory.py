"""
Author: B. Chen

Model factory with decorator registration.
"""

from typing import Dict, Type, List
from .interfaces import BaseModel


class ModelFactory:
    """Model factory with decorator registration."""
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register model class."""
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """Create model instance."""
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model names."""
        return list(cls._models.keys())


def register_model(name: str):
    """Decorator for automatic model registration."""
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        ModelFactory.register(name, cls)
        return cls
    return decorator