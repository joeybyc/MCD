"""
Author: B. Chen

Spatial attention network wrapper implementation.
"""

import cv2
import numpy as np
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms

from ...model_management import BaseModel, HTTPModelLoader, register_model
from .config import SpatialAttentionConfig


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
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
    """Spatial attention network for cell classification."""
    
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


@register_model("spatial_attention_network")
class SpatialAttentionWrapper(BaseModel):
    """Spatial attention network wrapper."""
    
    def __init__(self, config: SpatialAttentionConfig = None):
        """Initialize spatial attention wrapper."""
        self.config = config or SpatialAttentionConfig()
        self.loader = HTTPModelLoader(self.config.cache_dir)
        self._model = None
        self.name = self.config.name
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.size_img, self.config.size_img)),
            transforms.ToTensor()
        ])
    
    def load(self) -> None:
        """Load spatial attention model into memory."""
        if self.is_loaded:
            return
        
        model_path = self.loader.download_if_needed(self.config)
        
        try:
            self._model = SpatialAttentionNetwork(size_img=self.config.size_img)
            state_dict = torch.load(model_path, map_location=self.config.resolve_device())
            self._model.load_state_dict(state_dict)
            self._model.to(self.config.resolve_device())
            self._model.eval()
            
            print(f"Loaded spatial attention model: {self.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load spatial attention model: {e}")
    
    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self.config.resolve_device() == 'cuda':
            torch.cuda.empty_cache()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def classify(self, image: np.ndarray) -> int:
        """
        Classify single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Classification result (0 for noise, 1 for cell)
        """
        if not self.is_loaded:
            self.load()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Transform and predict
        image_tensor = self.transform(image).unsqueeze(0).to(self.config.resolve_device())
        
        with torch.no_grad():
            output = self._model(image_tensor)
            _, predicted = torch.max(output.data, 1)
        
        return predicted.item()
    
    def classify_batch(self, images: List[np.ndarray]) -> List[int]:
        """
        Classify batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of classification results
        """
        if not self.is_loaded:
            self.load()
        
        if not images:
            return []
        
        # Preprocess images
        processed_images = []
        for image in images:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed_images.append(self.transform(image))
        
        # Stack and predict
        batch_tensor = torch.stack(processed_images).to(self.config.resolve_device())
        
        with torch.no_grad():
            outputs = self._model(batch_tensor)
            _, predictions = torch.max(outputs.data, 1)
        
        return predictions.tolist()