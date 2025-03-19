from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.sam_model = CN()


_C.sam_model.ckpt = "models/sam_vit_h_4b8939.pth"
_C.sam_model.model_type = "vit_h"

# _C.sam_model.ckpt = "models/sam_vit_l_0b3195.pth"
# _C.sam_model.model_type = "vit_l"

# _C.sam_model.ckpt = "models/sam_vit_b_01ec64.pth"
# _C.sam_model.model_type = "vit_b"

_C.dataset = CN()


_C.cell_classifier = CN()
_C.cell_classifier.size_L = 20
_C.cell_classifier.ckpt = "models/spatial_attention_network.pth"

cfg = _C    # users can `from config import cfg`
