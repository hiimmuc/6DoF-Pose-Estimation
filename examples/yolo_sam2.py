
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "src/checkpoints/SAM2/sam2.1_hiera_tiny.pt"
model_cfg = "src/checkpoints/SAM2/configs/sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)