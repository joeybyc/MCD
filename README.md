# Minuscule Cell Detection Framework (MCD)
This repository contains the code for MCD, an anterior chamber cell detector.
MCD can detect the extremely small cells inside the AS-OCT images and thoese cell represent less than 0.005% are of the image.
## Download
```
git clone https://github.com/joeybyc/MCD.git
cd MCD
```

## Installation
1. Create a new conda environment with Python 3.9:
```
conda create -n MCD python=3.9
```
2. Activate the conda environment:
```
conda activate MCD # sometimes could be `source activate MCD`
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
## Checkpoints
MCD requires a pre-trained ViT model. Download the model from the [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) link.
The link is provided in the [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) repository. Put the pre-trained model into the **models** folder.

You can also use
```
wget -P ./models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

```
to download the ViT model.

## Getting Started

### Process one image
To process an image, segment the AC area, and detect cells, run the following command:
```
python -m apps.detect-cell
```
By default, this command will process the image located at `/example_datasets/patient1/image1.png`. The output will be saved in the `output/output_one_image` directory.

### Showcase
After running `python -m apps.detect-cell`, intermediate stage images will also be generated. Below are examples of the generated images for the sample input image.
#### Original image
![](example_datasets/patient1/image1.png)
#### Anterior Chamber Mask
![](output/output_one_image/image1/image_with_AC_mask.png)
#### Cell Dection by MCD
![](output/output_one_image/image1/image_with_cell_boxes.png)

### Process images in a folder
To process a list of images, run the following command:
```
python -m apps.detect-cell-all
```
By default, this command will process all the images in `/example_datasets`. The output will be saved in the `output/output_datasets` directory.

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
![](AWS.png)
![](CDI.png)
