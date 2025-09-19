"""
Author: B. Chen

Model loading utilities with download capabilities.
"""

import os
import requests
from pathlib import Path
from typing import Any
from tqdm import tqdm

from .interfaces import BaseModelLoader


class HTTPModelLoader(BaseModelLoader):
    """HTTP model downloader with caching."""
    
    def __init__(self, cache_dir: str = './models'):
        """Initialize loader with cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_if_needed(self, config: Any) -> str:
        """Download model if not exists locally."""
        local_path = self.cache_dir / config.local_filename
        
        if local_path.exists():
            return str(local_path)
        
        return self._download(config.checkpoint_url, local_path)
    
    def _download(self, url: str, local_path: Path) -> str:
        """Download file with progress bar."""
        print(f"Downloading model from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = file.write(chunk)
                    pbar.update(size)
            
            print(f"Downloaded to {local_path}")
            return str(local_path)
            
        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download failed: {e}")