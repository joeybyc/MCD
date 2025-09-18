"""
Author: B. Chen

Factory classes for creating prompt generator components.
"""

from typing import Dict, Type
from .interfaces import BasePromptGenerator
from .implementations.i2acp import I2ACPGenerator


class PromptGeneratorFactory:
    """Factory for creating prompt generator instances."""
    
    _generators: Dict[str, Type[BasePromptGenerator]] = {
        'i2acp': I2ACPGenerator,
    }
    
    @classmethod
    def create_generator(cls, generator_type: str, **kwargs) -> BasePromptGenerator:
        """
        Create a prompt generator instance.
        
        Args:
            generator_type: Type of generator ('i2acp')
            **kwargs: Parameters for the generator constructor
            
        Returns:
            Prompt generator instance
            
        Raises:
            ValueError: If generator type is not supported
        """
        if generator_type not in cls._generators:
            available = ', '.join(cls._generators.keys())
            raise ValueError(f"Unknown generator: {generator_type}. Available: {available}")
        
        return cls._generators[generator_type](**kwargs)
    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[BasePromptGenerator]) -> None:
        """
        Register a new generator type.
        
        Args:
            name: Name for the generator type
            generator_class: Generator class that implements BasePromptGenerator
        """
        cls._generators[name] = generator_class
    
    @classmethod
    def get_available_generators(cls) -> list:
        """Get list of available generator types."""
        return list(cls._generators.keys())