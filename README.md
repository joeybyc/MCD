# ASOCT-MCD: Minuscule Cell Detection in AS-OCT Images

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
