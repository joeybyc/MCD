"""
Author: B. Chen

PyTorch model wrapper implementation for classification models.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torchvision import transforms

from ..interfaces import ClassificationModelWrapper
from ..config import ModelConfig
from ..loaders import ModelLoader


class SpatialAttention(nn.Module):
    """Spatial attention module from original code."""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.conv(attention)
        return x * attention_map


class SpatialAttentionNetwork(nn.Module):
    """Spatial Attention Network from original code."""
    
    def __init__(self, size_img=20):
        super(SpatialAttentionNetwork, self).__init__()
        self.size_img = size_img
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.attention = SpatialAttention()
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class PyTorchClassificationWrapper(ClassificationModelWrapper):
    """Wrapper for PyTorch classification models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize PyTorch classification wrapper.
        
        Args:
            config: Model configuration
        """
        super().__init__(config.name, config.device)
        self.config = config
        self.size_img = config.model_params.get('size_img', 20)
        self.loader = ModelLoader()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size_img, self.size_img)),
            transforms.ToTensor()
        ])
    
    def load(self) -> None:
        """Load PyTorch model into memory."""
        if self.is_loaded:
            return
        
        # Ensure model file is available
        model_path = self.loader.prepare_model(self.config)
        
        try:
            # Initialize spatial attention network
            self.model = SpatialAttentionNetwork(size_img=self.size_img)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"Successfully loaded PyTorch model: {self.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model {self.name}: {e}")
    
    def unload(self) -> None:
        """Release PyTorch model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict classification for single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 for noise, 1 for cell)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.name} is not loaded")
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            
        return predicted.item()
    
    def predict_batch(self, images: List[np.ndarray]) -> List[int]:
        """
        Predict classification for batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of classification results (0 for noise, 1 for cell)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.name} is not loaded")
        
        if not images:
            return []
        
        # Preprocess images
        processed_images = []
        for image in images:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed_images.append(self.transform(image))
        
        # Stack into batch tensor
        batch_tensor = torch.stack(processed_images).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            _, predictions = torch.max(outputs.data, 1)
            
        return predictions.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get PyTorch model metadata information."""
        return {
            'name': self.name,
            'type': 'pytorch_classification',
            'size_img': self.size_img,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'checkpoint_path': self.config.local_path
        }