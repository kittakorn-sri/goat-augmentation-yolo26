"""Print params and GFLOPs (at IMGSZ) for the compared models, without rerunning Phase 5.
Usage (inside the goat_v2 folder):  python gflops_check.py"""
from config import IMGSZ, MODEL, COMPARE_MODELS
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
for w in [MODEL] + list(COMPARE_MODELS):
    m = YOLO(w)
    p = sum(x.numel() for x in m.model.parameters()) / 1e6
    print(f"{w:12s} params={p:.3f} M  GFLOPs@{IMGSZ}={get_flops(m.model, IMGSZ):.2f}")
