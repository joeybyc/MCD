# ASOCT-MCD: Minuscule Cell Detection in AS-OCT Images

A Python package for detecting minuscule cells in Anterior Segment Optical Coherence Tomography (AS-OCT) medical images.

## Installation

```bash
pip install asoct-mcd
```

## Basic Usage

```python
from asoct_mcd.pipeline import MCDPipelineBuilder

# Create pipeline with default settings
pipeline = MCDPipelineBuilder().build()

# Detect cells in image
result = pipeline.detect_cells("image.png")

# Print results
print(f"Detected {result.cell_count} cells")
print(f"Cell locations: {result.cell_locations}")
```

## Custom Configuration

```python
# Using dictionary configuration
config = {
    "threshold": {"lambda_factor": 0.9, "method": "isodata"},
}

pipeline = MCDPipelineBuilder().from_dict(config).build()
result = pipeline.detect_cells("image.png")
```

```python
# Using YAML configuration
pipeline = MCDPipelineBuilder().from_yaml("your_config.yaml").build()
result = pipeline.detect_cells("image.png")
```

### Showcase
#### Original image
![](readme_images/asoct_image.png)
#### Anterior Chamber Mask
![](readme_images/chamber_mask.png)
#### Cell Dection by MCD
![](readme_images/asoct_image_with_cell_boxes.png)


## Model Storage and Management
Default Model Storage Locations
The ASOCT-MCD package automatically downloads and manages pre-trained models. Models are cached locally following standard ML library conventions:
Default Cache Directories
Linux/macOS:
```
~/.cache/asoct_mcd/models/
├── sam_vit_b_01ec64.pth              # SAM ViT-B segmentation model (~375MB)
└── spatial_attention_network.pth     # Cell classification model (~1MB)
```
Windows:
```
C:\Users\<username>\.cache\asoct_mcd\models\
├── sam_vit_b_01ec64.pth              # SAM ViT-B segmentation model (~375MB)
└── spatial_attention_network.pth     # Cell classification model (~1MB)
```

## Requirements
- Python >= 3.9
- See requirements.txt for full list

## Citation

arXiv: https://arxiv.org/abs/2503.12249

To cite MCD in publications, please use:

```bibtex
@article{chen2025minuscule,
      title={Minuscule Cell Detection in AS-OCT Images with Progressive Field-of-View Focusing}, 
      author={Boyu Chen, Ameenat L. Solebo, Daqian Shi, Jinge Wu, Paul Taylor},
      year={2025},
      journal={arXiv preprint arXiv:2503.12249}
}

```
## Acknowledgements
Thanks to the support of AWS Doctoral Scholarship in Digital Innovation, awarded through the UCL Centre for Digital Innovation. We thank them for their generous support.
![](readme_images/AWS.png)
![](readme_images/CDI.png)
