#!/usr/bin/env python3
"""
Setup script for ASOCT-MCD package
This is a fallback setup.py for compatibility with older tools
Main configuration is in pyproject.toml
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

# Basic requirements (fallback if requirements.txt doesn't exist)
requirements = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0", 
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "scikit-image>=0.19.0",
    "scikit-learn>=1.1.0",
    "matplotlib>=3.6.0",
    "pandas>=1.5.0",
    "pyyaml>=6.0",
    "yacs>=0.1.8",
    "segment-anything>=1.0",
    "pillow>=9.0.0",
    "click>=8.0.0",
    "tqdm>=4.64.0",
    "requests>=2.28.0",
    "packaging>=21.0",
]

# Try to read from requirements.txt if it exists
requirements_from_file = read_requirements("requirements.txt")
if requirements_from_file:
    requirements = requirements_from_file

setup(
    name="asoct-mcd",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Minuscule Cell Detection in AS-OCT Medical Images using SAM and Spatial Attention Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joeybyc/MCD",
    project_urls={
        "Bug Tracker": "https://github.com/joeybyc/MCD/issues",
        "Documentation": "https://github.com/joeybyc/MCD#readme",
        "Source Code": "https://github.com/joeybyc/MCD",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "asoct-mcd=asoct_mcd.cli.commands:main",
        ],
    },
    package_data={
        "asoct_mcd.config": ["*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
)