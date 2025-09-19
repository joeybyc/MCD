"""
Author: B. Chen

Model loading utilities with download capabilities and integrity checking.
"""

import os
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Any
from tqdm import tqdm

from .interfaces import BaseModelLoader


class HTTPModelLoader(BaseModelLoader):
    """HTTP model downloader with caching and integrity checking."""
    
    def __init__(self, cache_dir: str = './models'):
        """Initialize loader with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_if_needed(self, config: Any) -> str:
        """Download model if not exists locally or corrupted."""
        local_path = self.cache_dir / config.local_filename
        
        # Check if file exists and is valid
        if local_path.exists():
            if self._validate_model_file(local_path, config):
                return str(local_path)
            else:
                print(f"Corrupted model file detected: {local_path}")
                self._safe_remove(local_path)
        
        return self._download_with_recovery(config.checkpoint_url, local_path)
    
    def _validate_model_file(self, local_path: Path, config: Any) -> bool:
        """Validate model file integrity."""
        try:
            # Basic file size check (non-zero)
            if local_path.stat().st_size == 0:
                return False
            
            # Try to load the model file to check if it's valid
            # This is the most reliable way to check integrity
            import torch
            try:
                torch.load(local_path, map_location='cpu')
                return True
            except Exception:
                return False
                
        except Exception:
            return False
    
    def _safe_remove(self, file_path: Path) -> None:
        """Safely remove file if it exists."""
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"Removed corrupted file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove corrupted file {file_path}: {e}")
    
    def _download_with_recovery(self, url: str, local_path: Path) -> str:
        """Download file with atomic operation and recovery."""
        print(f"Downloading model from {url}")
        
        # Create temporary file in same directory for atomic move
        temp_dir = local_path.parent
        temp_file = None
        
        try:
            # Use temporary file in same directory
            with tempfile.NamedTemporaryFile(
                dir=temp_dir, 
                delete=False, 
                suffix='.tmp'
            ) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Download to temporary file
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = temp_file.write(chunk)
                        pbar.update(size)
            
            # Atomic move: rename temporary file to final location
            shutil.move(str(temp_path), str(local_path))
            print(f"Downloaded successfully to {local_path}")
            
            return str(local_path)
            
        except Exception as e:
            # Clean up temporary file on failure
            if temp_file and Path(temp_file.name).exists():
                try:
                    Path(temp_file.name).unlink()
                except:
                    pass
            
            # Clean up partial download
            if local_path.exists():
                self._safe_remove(local_path)
            
            raise RuntimeError(f"Download failed: {e}")