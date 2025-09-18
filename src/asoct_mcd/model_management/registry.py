"""
Author: B. Chen

Model registry for centralized model management.
"""

from typing import Dict, List, Optional, Type
import threading
from .interfaces import BaseModelWrapper
from .config import ModelConfig, DEFAULT_MODELS
from .implementations.sam_wrapper import SAMWrapper
from .implementations.pytorch_wrapper import PyTorchClassificationWrapper


class ModelRegistry:
    """Singleton registry for managing model instances."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry if not already initialized."""
        if hasattr(self, '_initialized'):
            return
        
        self._models: Dict[str, BaseModelWrapper] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._wrappers: Dict[str, Type[BaseModelWrapper]] = {
            'sam': SAMWrapper,
            'pytorch_classification': PyTorchClassificationWrapper,
        }
        
        # Register default models
        for name, config in DEFAULT_MODELS.items():
            self._configs[name] = config
        
        self._initialized = True
    
    def register_model_config(self, config: ModelConfig) -> None:
        """
        Register a model configuration.
        
        Args:
            config: Model configuration to register
            
        Raises:
            ValueError: If model type is not supported
        """
        if config.model_type not in self._wrappers:
            available = ', '.join(self._wrappers.keys())
            raise ValueError(f"Unsupported model type: {config.model_type}. Available: {available}")
        
        self._configs[config.name] = config
    
    def register_wrapper(self, model_type: str, wrapper_class: Type[BaseModelWrapper]) -> None:
        """
        Register a new model wrapper type.
        
        Args:
            model_type: Type identifier for the wrapper
            wrapper_class: Wrapper class that implements BaseModelWrapper
        """
        self._wrappers[model_type] = wrapper_class
    
    def get_model(self, name: str, auto_load: bool = True) -> BaseModelWrapper:
        """
        Get model instance by name (lazy loading).
        
        Args:
            name: Model name
            auto_load: Whether to automatically load the model if not loaded
            
        Returns:
            Model wrapper instance
            
        Raises:
            KeyError: If model is not registered
            ValueError: If model type is not supported
        """
        # Check if model is already instantiated
        if name in self._models:
            model = self._models[name]
            if auto_load and not model.is_loaded:
                model.load()
            return model
        
        # Check if config exists
        if name not in self._configs:
            available = ', '.join(self._configs.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available}")
        
        config = self._configs[name]
        
        # Get wrapper class
        if config.model_type not in self._wrappers:
            available = ', '.join(self._wrappers.keys())
            raise ValueError(f"Unsupported model type: {config.model_type}. Available: {available}")
        
        wrapper_class = self._wrappers[config.model_type]
        
        # Create and store model instance
        model = wrapper_class(config)
        self._models[name] = model
        
        # Auto-load if requested
        if auto_load:
            model.load()
        
        return model
    
    def preload_models(self, model_names: List[str]) -> None:
        """
        Preload multiple models.
        
        Args:
            model_names: List of model names to preload
        """
        for name in model_names:
            try:
                self.get_model(name, auto_load=True)
                print(f"Preloaded model: {name}")
            except Exception as e:
                print(f"Failed to preload model {name}: {e}")
    
    def unload_model(self, name: str) -> None:
        """
        Unload a specific model from memory.
        
        Args:
            name: Model name to unload
        """
        if name in self._models:
            self._models[name].unload()
    
    def unload_all_models(self) -> None:
        """Unload all models from memory."""
        for model in self._models.values():
            model.unload()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return [name for name, model in self._models.items() if model.is_loaded]
    
    def get_available_models(self) -> List[str]:
        """Get list of all registered model names."""
        return list(self._configs.keys())
    
    def get_model_info(self, name: str) -> Dict:
        """
        Get information about a specific model.
        
        Args:
            name: Model name
            
        Returns:
            Model information dictionary
        """
        if name in self._models:
            return self._models[name].get_model_info()
        elif name in self._configs:
            config = self._configs[name]
            return {
                'name': config.name,
                'type': config.model_type,
                'is_loaded': False,
                'checkpoint_path': config.local_path
            }
        else:
            raise KeyError(f"Model '{name}' not found")
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all loaded models.
        
        Returns:
            Dictionary mapping model names to health status
        """
        return {name: model.health_check() for name, model in self._models.items()}