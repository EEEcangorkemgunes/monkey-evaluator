import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch import cuda

output_dir = os.path.join("models", "output_kidney_patches")
model_weights = os.path.join(output_dir, "model_final.pth")


def Predictor():
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(output_dir, "config.yaml"))
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor
