# ==================== 0a_prepare_dataset.py ====================
# Build a leakage-aware train / valid / test split from a dataset in YOLO layout.
# Source: a folder you already downloaded (--source DIR, or env DATASET_DIR; Roboflow is NOT contacted),
#         or, without --source, version ROBOFLOW_VERSION downloaded through the Roboflow API.
#
#  - near-duplicates are grouped (perceptual hash + Roboflow ".rf." augmentation siblings)
#  - groups are never split across partitions (StratifiedGroupKFold)
#  - background (no-label) images are a stratum of their own
#  - files are copied with a unique prefix (no silent overwrite by identical basenames)
#
# Split proportions come from the fold sizes: test = 1/7 (14.3%), valid = 1/6 of the rest (14.3%), train = 71.4%.
# Run: python 0a_prepare_dataset.py --source /path/to/downloaded_dataset
#      python 0a_prepare_dataset.py            (Roboflow API; needs ROBOFLOW_API_KEY)

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    INVOCATION_DIR, SPLIT_SEED, SPLIT_DIR, DATA_YAML, DATA_TEST_YAML, ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION, ROBOFLOW_FORMAT,
    HASH_THRESHOLD, EMBED_BACKEND, EMBED_GROUP_THRESHOLD, set_seed, get_logger,
)
from similarity import phash_matrix, duplicate_pairs, embed_images, similarity_pairs

logger = get_logger("phase0")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------------
# Pure functions (unit-testable)
# ------------------------------------------------------------------
def union_groups(n, pairs):
    """Union-find. Returns an int group id per item."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])


def group_split(strata, groups, seed):
    """Returns (train_idx, valid_idx, test_idx); groups never cross partitions."""
    strata = np.asarray(strata)
    groups = np.asarray(groups)
    idx = np.arange(len(strata))
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    trainval, test = next(outer.split(idx, strata, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr, va = next(inner.split(trainval, strata[trainval], groups[trainval]))
    return trainval[tr], trainval[va], test


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def label_path_for(img_path):
    p = Path(img_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def read_classes(label_file):
    if not label_file.exists():
        return None                      # missing label file
    with open(label_file) as f:
        return [int(line.split()[0]) for line in f if line.strip()]


def find_data_yaml(base):
    base = Path(base)
    direct = base / "data.yaml"
    if direct.exists():
        return direct
    hits = sorted(base.rglob("data.yaml"))
    return hits[0] if hits else None


def class_names_from_yaml(path):
    with open(path) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def find_dataset_images(base):
    """Images inside a folder called 'images' (Roboflow / YOLO layout), at any depth, any extension case."""
    return sorted(str(p) for p in Path(base).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS and p.parent.name.lower() == "images")


def rf_base(name):
    m = re.match(r"(.+?)\.rf\.", name)
    return m.group(1) if m else None


# ------------------------------------------------------------------
def main():
    set_seed(SPLIT_SEED)
    if os.path.exists(DATA_YAML) and os.path.exists(DATA_TEST_YAML):
        logger.info(f"Split already exists at {SPLIT_DIR}; delete it to rebuild.")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.environ.get("DATASET_DIR"),
                    help="folder of an already downloaded dataset in YOLO layout (no Roboflow call)")
    ap.add_argument("--names", default=None,
                    help="comma-separated class names in label-index order, e.g. CAE,Orf (used if there is no data.yaml)")
    args = ap.parse_args()

    if args.source:
        base = args.source if os.path.isabs(args.source) else os.path.join(INVOCATION_DIR, args.source)
        yaml_path = find_data_yaml(base)
        if yaml_path is not None:
            class_names = class_names_from_yaml(yaml_path)
        elif args.names:
            class_names = [n.strip() for n in args.names.split(",") if n.strip()]
        else:
            logger.error(f"No data.yaml found under {base}. Pass the class names with --names CAE,Orf "
                         "(in label-index order).")
            sys.exit(1)
        logger.info(f"Local dataset: {base} (Roboflow not contacted); classes: {class_names}")
    else:
        if not ROBOFLOW_API_KEY:
            logger.error("Set ROBOFLOW_API_KEY, or pass --source with a downloaded dataset folder.")
            sys.exit(1)
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        version = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT).version(ROBOFLOW_VERSION)
        dataset = version.download(ROBOFLOW_FORMAT)
        base = dataset.location
        class_names = class_names_from_yaml(find_data_yaml(base))
        logger.info(f"Downloaded to {base}; classes: {class_names}")

    # ---- collect images (case-insensitive extensions) ----
    images = find_dataset_images(base)
    if not images:
        logger.error(f"No images found under {base}. Expected folders named 'images' with a sibling 'labels' "
                     "folder (YOLO layout, e.g. train/images + train/labels).")
        sys.exit(1)
    per_folder = Counter(Path(p).parent.parent.name for p in images)
    logger.info(f"Images found: {len(images)}  (by original folder: {dict(per_folder)})")

    # ---- strata: majority class, or -1 for background ----
    strata, missing = [], 0
    for p in images:
        cls = read_classes(label_path_for(p))
        if cls is None:
            missing += 1
            strata.append(-1)
        elif len(cls) == 0:
            strata.append(-1)
        else:
            cnt = Counter(cls)
            top = max(cnt.values())
            strata.append(min(c for c, v in cnt.items() if v == top))   # deterministic tie-break
    strata = np.array(strata)
    logger.info(f"Images with MISSING label file (treated as background): {missing}")
    logger.info(f"Background images: {(strata == -1).sum()}")

    # ---- duplicate groups ----
    logger.info("Computing perceptual hashes ...")
    H = phash_matrix(images)
    pairs = duplicate_pairs(H, HASH_THRESHOLD)
    logger.info(f"pHash pairs (Hamming <= {HASH_THRESHOLD}): {len(pairs)}")
    if EMBED_GROUP_THRESHOLD is not None:
        logger.info(f"Computing embeddings ({EMBED_BACKEND}) ...")
        E = embed_images(images, EMBED_BACKEND)
        epairs = similarity_pairs(E, EMBED_GROUP_THRESHOLD)
        logger.info(f"Embedding pairs (cosine >= {EMBED_GROUP_THRESHOLD}): {len(epairs)}")
        pairs.extend(epairs)
    by_base = {}
    for i, p in enumerate(images):
        b = rf_base(Path(p).name)
        if b:
            by_base.setdefault(b, []).append(i)
    for members in by_base.values():
        pairs.extend((members[0], m) for m in members[1:])
    groups = union_groups(len(images), pairs)
    sizes = Counter(groups)
    logger.info(f"Related-image pairs: {len(pairs)}; groups: {len(sizes)}; "
                f"images in multi-image groups: {sum(v for v in sizes.values() if v > 1)}")
    biggest = max(sizes.values())
    logger.info(f"Largest group: {biggest} images ({100 * biggest / len(images):.1f}% of all images)")
    if biggest > 0.10 * len(images):
        logger.warning("A single group holds >10% of the images (chaining). The split may be badly unbalanced; "
                       "raise the similarity threshold or review the pairs before using this split.")

    # ---- split ----
    tr, va, te = group_split(strata, groups, SPLIT_SEED)
    assert not (set(groups[tr]) & set(groups[va]) | set(groups[tr]) & set(groups[te]) | set(groups[va]) & set(groups[te]))

    # ---- copy ----
    manifest = []
    for split_name, idxs in (("train", tr), ("valid", va), ("test", te)):
        os.makedirs(f"{SPLIT_DIR}/{split_name}/images", exist_ok=True)
        os.makedirs(f"{SPLIT_DIR}/{split_name}/labels", exist_ok=True)
        for i in idxs:
            src = Path(images[i])
            uid = f"{i:05d}_{src.name}"
            shutil.copy(src, f"{SPLIT_DIR}/{split_name}/images/{uid}")
            lbl = label_path_for(src)
            if lbl.exists():
                shutil.copy(lbl, f"{SPLIT_DIR}/{split_name}/labels/{Path(uid).stem}.txt")
            manifest.append({"split": split_name, "file": uid, "orig": str(src),
                             "stratum": int(strata[i]), "group": int(groups[i])})
    pd.DataFrame(manifest).to_csv(os.path.join(SPLIT_DIR, "split_manifest.csv"), index=False)

    # ---- yaml ----
    for path, val in ((DATA_YAML, "valid/images"), (DATA_TEST_YAML, "test/images")):
        with open(path, "w") as f:
            yaml.dump({"path": os.path.abspath(SPLIT_DIR), "train": "train/images",
                       "val": val, "names": class_names}, f)

    # ---- report (numbers needed by the reviewers) ----
    logger.info("\nSplit report (images / instances per class):")
    names = class_names if isinstance(class_names, list) else [class_names[k] for k in sorted(class_names)]
    for split_name in ("train", "valid", "test"):
        files = list(Path(f"{SPLIT_DIR}/{split_name}/labels").glob("*.txt"))
        inst, with_cls = Counter(), Counter()
        for lf in files:
            cls = read_classes(lf) or []
            inst.update(cls)
            for c in set(cls):
                with_cls[c] += 1
        n_img = len(list(Path(f"{SPLIT_DIR}/{split_name}/images").glob("*")))
        logger.info(f"  {split_name}: images={n_img}, no-label images={n_img - sum(1 for lf in files if read_classes(lf))}, "
                    + ", ".join(f"{names[c]}: {inst[c]} inst / {with_cls[c]} img" for c in range(len(names))))
    logger.info("Done. Manifest: " + os.path.join(SPLIT_DIR, "split_manifest.csv"))


if __name__ == "__main__":
    main()
