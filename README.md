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

```

## Requirements

- Python >= 3.9
- PyTorch >= 1.12.0
- See requirements.txt for full list

## Development Status

This package is currently under active development. APIs may change between versions.

## License

MIT License - see LICENSE file for details.
