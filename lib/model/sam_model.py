from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor


# def load_sam_predictor(sam_checkpoint = "", model_type = "", device = None):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     sam2 = build_sam2(model_type, sam_checkpoint)
#     # sam2.half()
#     # sam2.to(device=device)
#     predictor = SAM2ImagePredictor(sam2)
#     return predictor


def load_sam_predictor(sam_checkpoint = "models/sam_vit_h_4b8939.pth", model_type = "vit_h", device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"The device is {device}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor
