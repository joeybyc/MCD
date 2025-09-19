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
result = pipeline.detect_cells("path/to/image.png")

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
