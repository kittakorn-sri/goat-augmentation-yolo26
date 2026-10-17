# ==================== 0b_verify_split.py ====================
# Evidence about possible leakage between train and valid/test, for images whose animal / farm / source cannot be identified.
#
# What it produces (folder: split_verification/):
#   nearest_train_neighbors.csv   per valid/test image: nearest TRAIN image by embedding cosine and by pHash distance
#   similarity_summary.csv        distribution of nearest-train similarity for valid, test and train (leave-one-out)
#   review_template.csv           most similar cross-split pairs with an empty column for MANUAL labelling
#   top_pairs_page*.jpg           side-by-side images of those pairs for visual inspection
#   stratified_eval.csv           (only with --weights) test accuracy on the low- vs high-similarity halves of the test set
#
# This does NOT prove that no related images cross partitions: embedding similarity is evidence of visual relatedness,
# not of "same animal". Report what it finds and keep the limitation in the paper.
#
# Run: python 0b_verify_split.py [--backend dinov2|ultralytics] [--top 36] [--weights path/to/best.pt]

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from config import SPLIT_DIR, DATA_YAML, EMBED_BACKEND, HASH_THRESHOLD
from similarity import phash_matrix, embed_images, nearest_neighbors

OUT_DIR = "split_verification"
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]       # reporting grid only; not a decision rule


# ------------------------------------------------------------------
# Pure analysis (unit-testable)
# ------------------------------------------------------------------
def nn_table(manifest, E, H=None):
    """Nearest TRAIN neighbour of every valid/test image."""
    split = manifest["split"].to_numpy()
    tr = np.where(split == "train")[0]
    rows = []
    for sp in ("valid", "test"):
        q = np.where(split == sp)[0]
        if len(q) == 0:
            continue
        j, s = nearest_neighbors(E[q], E[tr])
        for qi, jj, ss in zip(q, j, s):
            d = int((H[tr] != H[qi]).sum(axis=1).min()) if H is not None else None
            rows.append({"split": sp, "file": manifest["file"].iloc[qi],
                         "nearest_train": manifest["file"].iloc[tr[jj]],
                         "cos_sim": float(ss), "phash_min_dist": d,
                         "group": int(manifest["group"].iloc[qi]),
                         "nearest_group": int(manifest["group"].iloc[tr[jj]])})
    return pd.DataFrame(rows)


def similarity_summary(manifest, E, nn):
    split = manifest["split"].to_numpy()
    tr = np.where(split == "train")[0]
    _, s_tr = nearest_neighbors(E[tr], E[tr], exclude_self=True)
    series = {"train (leave-one-out)": s_tr,
              **{f"{sp} -> train": nn.loc[nn["split"] == sp, "cos_sim"].to_numpy() for sp in ("valid", "test")
                 if (nn["split"] == sp).any()}}
    rows = []
    for name, v in series.items():
        row = {"comparison": name, "n_images": len(v), "median": float(np.median(v)),
               "p90": float(np.percentile(v, 90)), "max": float(np.max(v))}
        for t in THRESHOLDS:
            row[f"n_cos>={t:.2f}"] = int((v >= t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def contact_sheets(nn, img_dir_of, top, out_dir, per_page=12, thumb=224):
    from PIL import Image, ImageDraw
    pairs = nn.sort_values("cos_sim", ascending=False).head(top).reset_index(drop=True)
    for page, start in enumerate(range(0, len(pairs), per_page), 1):
        chunk = pairs.iloc[start:start + per_page]
        sheet = Image.new("RGB", (2 * thumb * 2, (thumb + 18) * ((len(chunk) + 1) // 2)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (_, r) in enumerate(chunk.iterrows()):
            col, row = k % 2, k // 2
            x0, y0 = col * 2 * thumb, row * (thumb + 18)
            for m, (fname, spl) in enumerate(((r["file"], r["split"]), (r["nearest_train"], "train"))):
                with Image.open(os.path.join(img_dir_of(spl), fname)) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    sheet.paste(im, (x0 + m * thumb, y0 + 18))
            draw.text((x0 + 2, y0 + 2), f"{r['split']} | train   cos={r['cos_sim']:.3f}  phash={r['phash_min_dist']}", fill="black")
        sheet.save(os.path.join(out_dir, f"top_pairs_page{page}.jpg"), quality=90)


# ------------------------------------------------------------------
# Optional: does accuracy depend on how similar a test image is to the training set?
# (test has few images, so each half is small and the result is noisy)
# ------------------------------------------------------------------
def count_instances(label_dir, files):
    n = 0
    for f in files:
        lf = os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lf):
            with open(lf) as fh:
                n += sum(1 for line in fh if line.strip())
    return n


def stratified_eval(weights, nn, out_dir):
    from common import load_model
    with open(DATA_YAML) as f:
        names = yaml.safe_load(f)["names"]
    t = nn[nn["split"] == "test"].sort_values("cos_sim").reset_index(drop=True)
    half = len(t) // 2
    parts = {"low_similarity_half": t.iloc[:half], "high_similarity_half": t.iloc[half:]}
    rows = []
    for name, part in parts.items():
        lst = os.path.join(out_dir, f"{name}.txt")
        with open(lst, "w") as f:
            f.write("\n".join(os.path.abspath(os.path.join(SPLIT_DIR, "test", "images", x)) for x in part["file"]))
        ypath = os.path.join(out_dir, f"{name}.yaml")
        with open(ypath, "w") as f:
            yaml.dump({"path": os.path.abspath(SPLIT_DIR), "train": "train/images",
                       "val": os.path.abspath(lst), "names": names}, f)
        m = load_model(weights).val(data=ypath, split="val", conf=0.001, iou=0.7, plots=False,
                                    project=os.path.join(out_dir, "val"), name=name, exist_ok=True)
        rows.append({"subset": name, "n_images": len(part),
                     "n_instances": count_instances(os.path.join(SPLIT_DIR, "test", "labels"), part["file"]),
                     "median_cos_sim": float(part["cos_sim"].median()),
                     "mAP50_95": float(m.box.map), "mAP50": float(m.box.map50),
                     "Precision": float(m.box.mp), "Recall": float(m.box.mr)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=EMBED_BACKEND)
    ap.add_argument("--top", type=int, default=36, help="number of most-similar cross-split pairs to show")
    ap.add_argument("--weights", default=None, help="best.pt for the similarity-stratified evaluation")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd.read_csv(os.path.join(SPLIT_DIR, "split_manifest.csv"))
    path_of = lambda i: os.path.join(SPLIT_DIR, manifest["split"].iloc[i], "images", manifest["file"].iloc[i])
    paths = [path_of(i) for i in range(len(manifest))]

    print("Perceptual hashes ...")
    H = phash_matrix(paths)
    print(f"Embeddings ({args.backend}) ...")
    E = embed_images(paths, args.backend)

    nn = nn_table(manifest, E, H)
    nn.to_csv(os.path.join(OUT_DIR, "nearest_train_neighbors.csv"), index=False)
    summ = similarity_summary(manifest, E, nn)
    summ.to_csv(os.path.join(OUT_DIR, "similarity_summary.csv"), index=False)
    print("\nNearest-train similarity (cosine):\n" + summ.to_string(index=False))

    close_hash = nn[nn["phash_min_dist"] <= HASH_THRESHOLD]
    print(f"\nvalid/test images with a TRAIN image within pHash distance {HASH_THRESHOLD}: {len(close_hash)} "
          "(should be 0 if grouping worked)")
    same_group = nn[nn["group"] == nn["nearest_group"]]
    print(f"valid/test images whose nearest train image is in the SAME group: {len(same_group)} (must be 0)")

    img_dir_of = lambda split: os.path.join(SPLIT_DIR, split, "images")
    top = nn.sort_values("cos_sim", ascending=False).head(args.top).copy()
    top["same_animal_or_source (y/n/unsure)"] = ""
    top.to_csv(os.path.join(OUT_DIR, "review_template.csv"), index=False)
    contact_sheets(nn, img_dir_of, args.top, OUT_DIR)
    print(f"\nReview the pairs in {OUT_DIR}/top_pairs_page*.jpg and fill review_template.csv "
          "(report how many pairs you judged to be the same animal/source).")

    if args.weights:
        res = stratified_eval(args.weights, nn, OUT_DIR)
        res.to_csv(os.path.join(OUT_DIR, "stratified_eval.csv"), index=False)
        print("\nTest accuracy by similarity to the training set (small subsets, noisy):\n" + res.to_string(index=False))


if __name__ == "__main__":
    main()
