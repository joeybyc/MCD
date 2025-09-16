#!/usr/bin/env python3
"""
Project Structure Setup Script for ASOCT-MCD Package
Automatically creates the directory structure for PyPI package
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create the complete directory structure for the ASOCT-MCD package"""
    
    # Define the project structure
    structure = {
        # Main package directory
        "src/asoct_mcd": [
            "__init__.py",
        ],
        
        # Core detection logic
        "src/asoct_mcd/core": [
            "__init__.py",
            "detector.py",
            "pipeline.py"
        ],
        
        # Model management
        "src/asoct_mcd/models": [
            "__init__.py", 
            "model_manager.py",
            "sam_model.py",
            "attention_network.py"
        ],
        
        # Image processing module
        "src/asoct_mcd/image_processing": [
            "__init__.py",
            "loading.py", 
            "denoising.py",
            "mask_operations.py"
        ],
        
        # Thresholding algorithms (strategy pattern)
        "src/asoct_mcd/image_processing/thresholding": [
            "__init__.py",
            "base.py",
            "otsu_threshold.py", 
            "isodata_threshold.py"
        ],
        
        # Segmentation module (strategy pattern)
        "src/asoct_mcd/segmentation": [
            "__init__.py",
            "base.py"
        ],
        
        # Zero-shot segmentation
        "src/asoct_mcd/segmentation/zero_shot": [
            "__init__.py",
            "sam_segmenter.py",
            "foundation_base.py"
        ],
        
        # Pre-trained segmentation (reserved)
        "src/asoct_mcd/segmentation/pretrained": [
            "__init__.py",
            "unet_segmenter.py"
        ],
        
        # Cell detection module (strategy pattern)
        "src/asoct_mcd/detection": [
            "__init__.py",
            "base.py", 
            "attention_detector.py",
            "cell_filter.py"
        ],
        
        # Utilities
        "src/asoct_mcd/utils": [
            "__init__.py",
            "visualization.py",
            "io_utils.py",
            "metrics.py"
        ],
        
        # Configuration management (builder pattern)
        "src/asoct_mcd/config": [
            "__init__.py",
            "settings.py",
            "defaults.py", 
            "registry.py",
            "factory.py"
        ],
        
        # Command line interface
        "src/asoct_mcd/cli": [
            "__init__.py",
            "commands.py"
        ],
        
        # Tests directory
        "tests": [
            "__init__.py",
            "conftest.py"
        ],
        
        # Unit tests
        "tests/unit": [
            "__init__.py",
            "test_core.py",
            "test_image_processing.py", 
            "test_segmentation.py",
            "test_detection.py",
            "test_config.py"
        ],
        
        # Integration tests
        "tests/integration": [
            "__init__.py",
            "test_end_to_end.py",
            "test_cli.py"
        ],
        
        # Test data
        "tests/data": [
            ".gitkeep"
        ],
        
        # Documentation
        "docs": [
            "__init__.py",
            "index.md",
            "installation.md",
            "quickstart.md",
            "api.md"
        ],
        
        # Examples
        "examples": [
            "__init__.py",
            "basic_usage.py",
            "batch_processing.py",
            "custom_components.py"
        ],
        
        # Configuration files
        "configs": [
            "default.yaml",
            "production.yaml",
            "development.yaml"
        ],
        
        # Scripts
        "scripts": [
            "download_models.py",
            "setup_dev_env.py"
        ]
    }
    
    # Root level files that need to be created
    root_files = [
        "pyproject.toml",
        "setup.py", 
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        ".gitignore",
        ".github/workflows/ci.yml",
        "requirements.txt",
        "requirements-dev.txt"
    ]
    
    print("Creating ASOCT-MCD project structure...")
    
    # Create directories and files
    for directory, files in structure.items():
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        for file in files:
            file_path = dir_path / file
            if not file_path.exists():
                file_path.touch()
                print(f"  Created file: {file_path}")
    
    # Create root level files
    for file in root_files:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.touch()
            print(f"Created root file: {file_path}")
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Update pyproject.toml with package metadata")
    print("2. Configure setup.py")
    print("3. Update README.md")
    print("4. Set up .gitignore")
    print("5. Begin migrating existing code")


def create_initial_content():
    """Create initial content for key configuration files"""
    
    # Create .gitignore content
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Models and data
*.pth
*.pt
*.ckpt
models/
data/
datasets/
*.h5
*.hdf5

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    # Create basic README.md
    readme_content = """# ASOCT-MCD: Minuscule Cell Detection in AS-OCT Images

A Python package for detecting minuscule cells in Anterior Segment Optical Coherence Tomography (AS-OCT) medical images.

## Installation

```bash
pip install asoct-mcd
```

## Quick Start

```python
from asoct_mcd import CellDetector

# Initialize detector
detector = CellDetector()

# Detect cells in a single image
result = detector.detect("path/to/image.png")

# Process multiple images
results = detector.detect_batch("path/to/images/")
```

## Features

- Zero-shot cell detection using SAM (Segment Anything Model)
- Spatial attention networks for fine-grained classification
- Configurable processing pipeline
- Batch processing capabilities
- Command-line interface
- Extensible architecture with custom components

## Requirements

- Python >= 3.9
- PyTorch >= 1.12.0
- OpenCV >= 4.8.0
- See requirements.txt for full list

## Development Status

This package is currently under active development. APIs may change between versions.

## License

MIT License - see LICENSE file for details.
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Created initial .gitignore and README.md")


if __name__ == "__main__":
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create the structure
    create_directory_structure()
    create_initial_content()
    
    print(f"\nProject structure created in: {os.getcwd()}")
    print("You can now start migrating your existing code to the new structure!")