"""
Author: B. Chen

Model loading utilities with download capabilities.
"""

import os
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from .config import ModelConfig


class ModelDownloader:
    """Handles model download with progress tracking."""
    
    @staticmethod
    def download_model(config: ModelConfig, force_download: bool = False) -> str:
        """
        Download model if not exists locally.
        
        Args:
            config: Model configuration
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded model file
            
        Raises:
            RuntimeError: If download fails
        """
        local_path = Path(config.local_path)
        
        # Create parent directory if not exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if local_path.exists() and not force_download:
            return str(local_path)
        
        print(f"Downloading {config.name} from {config.checkpoint_url}")
        
        try:
            response = requests.get(config.checkpoint_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as file, tqdm(
                desc=f"Downloading {config.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = file.write(chunk)
                    pbar.update(size)
            
            print(f"Successfully downloaded {config.name} to {local_path}")
            return str(local_path)
            
        except Exception as e:
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Failed to download {config.name}: {e}")
    
    @staticmethod
    def verify_model_file(path: str) -> bool:
        """
        Verify model file exists and is readable.
        
        Args:
            path: Path to model file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            file_path = Path(path)
            return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0
        except Exception:
            return False


class ModelLoader:
    """Centralized model loading logic."""
    
    def __init__(self):
        """Initialize model loader."""
        self.downloader = ModelDownloader()
    
    def ensure_model_available(self, config: ModelConfig) -> str:
        """
        Ensure model is available locally, download if needed.
        
        Args:
            config: Model configuration
            
        Returns:
            Path to local model file
            
        Raises:
            RuntimeError: If model cannot be made available
        """
        # Check if local file exists and is valid
        if self.downloader.verify_model_file(config.local_path):
            return config.local_path
        
        # Download model
        return self.downloader.download_model(config)
    
    def prepare_model(self, config: ModelConfig) -> str:
        """
        Prepare model for loading (download if necessary).
        
        Args:
            config: Model configuration
            
        Returns:
            Path to prepared model file
        """
        return self.ensure_model_available(config)