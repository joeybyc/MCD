
from asoct_mcd.pipeline import MCDPipelineBuilder

# Create pipeline with default settings
pipeline = MCDPipelineBuilder().build()

# Detect cells in image
result = pipeline.detect_cells("image.png")

# Print results
print(f"Detected {result.cell_count} cells")
print(f"Cell locations: {result.cell_locations}")