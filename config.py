# ==================== config.py ====================
# Shared configuration for all phases (v2)
#
# Changes vs. v1
#  - API key is read from the environment only (never hard-coded)
#  - SEED is now passed to model.train(seed=...) (v1 trained with Ultralytics' default seed 0)
#  - Phases 1-3 select on the VALID split only; the TEST split is touched only by 4_stepwise_evaluation.py
#  - Albumentations is always passed explicitly (no-op transform when none is wanted)
#  - cwd is pinned to this folder so relative run paths (runs/detect/...) are stable

import os
import random
import logging
from datetime import datetime

import numpy as np
import torch
import albumentations as A

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INVOCATION_DIR = os.getcwd()          # where the command was typed (relative --source paths are resolved against it)
os.chdir(BASE_DIR)

# ==========================================
# General settings
# ==========================================
# Seeds are fixed BEFORE any result is seen and are all reported in the paper. Do not change them after looking at scores.
# Selection seeds (phases 1-3) and evaluation seeds (phases 4-6) are disjoint on purpose: the configuration was selected because it
# scored well under the selection seeds, so re-evaluating it with the same seed would be optimistically biased.
SPLIT_SEED = 42                                   # data split only (0a_prepare_dataset.py)
SELECTION_SEEDS = [7, 11, 13, 17, 19]                # training seeds of EVERY run in phases 1-3 (same 5 seeds for all configs
                                                  # -> comparisons are paired by seed)
EVALUATION_SEEDS = [101, 202, 303, 404, 505]           # phases 4-6 (final evaluation): repeated runs of the frozen configuration
SEED = SELECTION_SEEDS[0]                            # alias used by set_seed()
LR_SELECTION_SEED = SELECTION_SEEDS[0]               # phase 3: learning rates are screened with this one seed

assert len({SPLIT_SEED, *SELECTION_SEEDS, *EVALUATION_SEEDS}) == 1 + len(SELECTION_SEEDS) + len(EVALUATION_SEEDS), \
    "all seeds must be distinct"

# ==========================================
# Selection rules used in phases 1-3 (valid split only). Record the rules actually used in the paper.
# (Mosaic, MixUp and CutMix are independent techniques, so no mixing-group restriction is needed.)
#   Phase 1: candidates = every technique whose mean seed-paired gain over Baseline is > 0.
#   Phase 2: greedy forward selection; at each step the candidate with the highest mean paired gain is added if that
#            gain is > 0 (ACCEPT_RULE = "mean"); ACCEPT_RULE = "lower_bound" additionally requires the one-sided lower
#            confidence bound of the gain to be > 0. No limit on the set size (MAX_SET_SIZE = None).
#   Phase 3: an optimizer replaces `auto` only if the one-sided lower bound of its seed-paired gain vs auto is > 0.
# ==========================================
SELECTION_METRIC = "val_mAP50_95"
ACCEPT_RULE = "mean"                              # phase 2 acceptance: "mean" = accept the best candidate if its mean seed-paired gain
                                                  # over the current set is > 0; "lower_bound" = also require the one-sided
                                                  # lower confidence bound (ACCEPT_CONFIDENCE) to be > 0
ACCEPT_CONFIDENCE = 0.95                          # one-sided; used by "lower_bound", by phase 3, and always reported
SHORTLIST_MAX = None                              # phase 1 -> phase 2: None = ALL techniques with mean paired gain > 0 vs Baseline; an int caps the list
MAX_SET_SIZE = None                               # phase 2: None = no limit (stops when no candidate meets the acceptance rule or none are left)
MIXING_GROUP = ()                                 # techniques of which at most one may be chosen (empty: no restriction)

# Models for the comparison requested by the reviewer (same split, same seeds, pretrained weights for all)
COMPARE_MODELS = ["yolo26s.pt", "yolo11n.pt"]       # core
COMPARE_MODELS_OPTIONAL = ["rtdetr-l.pt"]           # only after checking which augmentations/optimizers it supports
EPOCHS = int(os.environ.get("EPOCHS", 600))   # EPOCHS=2 python ... for a smoke test
PATIENCE = 100
IMGSZ = 512
BATCH = 16
MODEL = "yolo26n.pt"
# 0 = mosaic / mixup / cutmix stay on for the whole run (Ultralytics would switch them off together for the last N epochs).
# Phase 1 runs stopped by early stopping at <= 212 epochs of 600, so a value of 10 was never reached; 0 states that explicitly.
CLOSE_MOSAIC_EPOCHS = 0
CLS_PW = 1.0                # verify what cls_pw does in your installed Ultralytics before describing it in the paper

# ==========================================
# Dataset
# ==========================================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")   # export ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE = "goat-qxgzi"
ROBOFLOW_PROJECT = "goat-twi81"
ROBOFLOW_VERSION = 4
ROBOFLOW_FORMAT = "yolo26"      # check this is a valid export format name for your Roboflow project
HASH_THRESHOLD = 6              # max Hamming distance (of 64-bit pHash) for two images to be treated as near-duplicates
# Optional stricter grouping with image embeddings (catches the same scene/animal in another pose or crop).
# None = off. Choose the value from the similarity diagnostics of 0b_verify_split.py and manual review of image pairs,
# BEFORE any training, and record it. Never tune it using model scores.
EMBED_BACKEND = "dinov2"        # "dinov2" (needs transformers + a model download) or "ultralytics" (yolo11n-cls)
EMBED_GROUP_THRESHOLD = None    # e.g. 0.90 (cosine similarity); similarity scales differ between backends

SPLIT_DIR = os.path.join(BASE_DIR, "single_split_data_v2")
DATA_YAML = os.path.join(SPLIT_DIR, "data.yaml")            # train / valid  (used in phases 1-3)
DATA_TEST_YAML = os.path.join(SPLIT_DIR, "data_test.yaml")  # train / test   (used ONLY in phases 4-6)

# ==========================================
# Logging
# ==========================================
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def get_logger(phase_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{phase_name}_{timestamp}.log")
    logger = logging.getLogger(phase_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    return logger


# ==========================================
# Augmentation configs
#   "albu" is always a list. An empty list means "no Albumentations";
#   common.build_aug_config() then substitutes a no-op transform so Ultralytics
#   cannot fall back to its default Albumentations set.
# ==========================================
ZERO_AUG = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
    "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
    "cutmix": 0.0,
}

augmentation_configs = {
    "Baseline":    {"builtin": {**ZERO_AUG}, "albu": []},
    "Flip":        {"builtin": {**ZERO_AUG, "fliplr": 0.5}, "albu": []},
    "Rotate":      {"builtin": {**ZERO_AUG, "degrees": 15.0}, "albu": []},
    "Scale":       {"builtin": {**ZERO_AUG, "scale": 0.2}, "albu": []},
    "Shear":       {"builtin": {**ZERO_AUG, "shear": 10.0}, "albu": []},
    "Translate":   {"builtin": {**ZERO_AUG, "translate": 0.2}, "albu": []},
    "Perspective": {"builtin": {**ZERO_AUG, "perspective": 0.0005}, "albu": []},
    "HSV":         {"builtin": {**ZERO_AUG, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4}, "albu": []},
    # Mosaic, MixUp and CutMix are separate techniques (MixUp and CutMix do NOT switch mosaic on)
    "Mosaic":      {"builtin": {**ZERO_AUG, "mosaic": 1.0}, "albu": []},
    "MixUp":       {"builtin": {**ZERO_AUG, "mixup": 0.5}, "albu": []},
    "CutMix":      {"builtin": {**ZERO_AUG, "cutmix": 0.5}, "albu": []},
    "Grayscale":          {"builtin": {**ZERO_AUG}, "albu": [A.ToGray(p=0.3)]},
    "CLAHE":              {"builtin": {**ZERO_AUG}, "albu": [A.CLAHE(clip_limit=4.0, p=0.3)]},
    "Blur":               {"builtin": {**ZERO_AUG}, "albu": [A.Blur(blur_limit=5, p=0.2)]},
    "GaussNoise":         {"builtin": {**ZERO_AUG}, "albu": [A.GaussNoise(std_range=(0.01, 0.05), p=0.2)]},
    "BrightnessContrast": {"builtin": {**ZERO_AUG}, "albu": [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3)]},
}

# ==========================================
# Optimizers: every optimizer gets the same small learning-rate grid
# (selected on the valid split), plus "auto" as a reference.
# Trim the grids if the compute budget is tight (6 optimizers x 3 LR = 18 runs).
# ==========================================
OPTIMIZER_CONFIGS = ["SGD", "MuSGD", "AdamW", "Adam", "NAdam", "RMSProp"]

OPTIMIZER_LR_GRID = {                              # 5 values per optimizer, roughly log-spaced (x2 steps)
    "SGD":     [0.02, 0.01, 0.005, 0.002, 0.001],
    "MuSGD":   [0.02, 0.01, 0.005, 0.002, 0.001],
    "AdamW":   [0.004, 0.002, 0.001, 0.0005, 0.00025],
    "Adam":    [0.004, 0.002, 0.001, 0.0005, 0.00025],
    "NAdam":   [0.004, 0.002, 0.001, 0.0005, 0.00025],
    "RMSProp": [0.002, 0.001, 0.0005, 0.0002, 0.0001],
}


# ==========================================
# Utilities
# ==========================================
def calc_f1(p, r):
    return 2 * p * r / (p + r + 1e-16)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def print_config():
    print("=" * 70)
    print(f"SPLIT_SEED={SPLIT_SEED}  SELECTION_SEEDS={SELECTION_SEEDS}  EVALUATION_SEEDS={EVALUATION_SEEDS}")
    print(f"accept: one-sided {ACCEPT_CONFIDENCE:.0%} lower bound > 0 on {SELECTION_METRIC}; shortlist={"all with gain>0" if SHORTLIST_MAX is None else f"top {SHORTLIST_MAX}"}; set size<={"no limit" if MAX_SET_SIZE is None else MAX_SET_SIZE}")
    print(f"EPOCHS={EPOCHS}  PATIENCE={PATIENCE}")
    print(f"CLS_PW={CLS_PW}  MODEL={MODEL}  IMGSZ={IMGSZ}  BATCH={BATCH}")
    print(f"Optimizers: {OPTIMIZER_CONFIGS}")
    print(f"Split dir: {SPLIT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
