# ==================== 6_bootstrap_test.py ====================
# Phase 6: how much do the TEST results depend on which test images happened to be sampled?
#
# For every evaluated configuration (models of phases 4/5, one entity per table:model) the saved best.pt of each
# evaluation seed is validated ONCE on the test split with Ultralytics val() (conf=0.001, IoU=0.7; per-image statistics cached). Then test images are
# resampled with replacement as CLUSTERS (the `group` column of split_manifest.csv, so related images are drawn
# together) and mAP is recomputed for every resample. The SAME resamples are used for every model, so
# differences between models are paired.
#
# Reported statistic of an entity = mean over its seeds of the mAP on the resample.
#   -> the interval covers the sampling of test images only; training variability across seeds is reported by
#      4_stepwise_evaluation.py / 5_compare_models.py (mean +- SD +- CI over seeds). Report both.
#
# Comparisons made automatically (when the entities exist):
#   frozen:yolo26n  vs  default:yolo26n   (tuned configuration vs Baseline; Baseline taken from Phase 4)
#   every other model vs YOLO26n within the same table (frozen / default)
#   frozen:yolo26n  vs  phase4:cls_pw0   (class-weight ablation of phase 4)
#   phase4:rungK+1 vs phase4:rungK  (ladder steps)
#
# Per-image TP flags / confidences / classes are recorded from the Ultralytics validator (same val() call as phase 4/5), so the
# full-test-set mAP reproduces the reported value; only the resampling and the AP aggregation are done here.
# Self-check: the full-test-set value is compared with the reported one; a difference above --tol stops the script.
# The AP aggregation (101-point interpolated AP, classes without ground truth in a resample are skipped) mirrors Ultralytics'
# ap_per_class; matching of detections to ground truth is NOT re-implemented (taken from the validator).
#
# Run: python 6_bootstrap_test.py [--n-boot 1000] [--boot-seed 777] [--tol 0.02]

import argparse
import json
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from config import SPLIT_DIR, DATA_TEST_YAML, IMGSZ, MODEL, get_logger
from common import load_csv, load_model

CONF, IOU = 0.001, 0.7                                 # as in common.evaluate_on_test (recorded in bootstrap_meta.json)
IOUV = np.linspace(0.5, 0.95, 10)
CACHE_DIR = "bootstrap_cache"
PHASE4_CSV = "results_final_test.csv"
COMPARE_CSV = "results_compare_test.csv"
OUT_RUNS = "bootstrap_run_level.csv"
OUT_CI = "bootstrap_ci.csv"
OUT_DIFF = "bootstrap_paired_differences.csv"
OUT_META = "bootstrap_meta.json"

logger = get_logger("phase6")
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------
# Detection metrics (numpy re-implementation of the Ultralytics evaluation)
# ------------------------------------------------------------------
def compute_ap(recall, precision):
    """101-point interpolated AP, as in Ultralytics."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def map_from_weights(P, w):
    """mAP50-95 and mAP50 when image i is counted w[i] times (w = ones -> the plain test set)."""
    ngt = w @ P["G"]
    aps = np.full((len(ngt), len(IOUV)), np.nan)
    for c in range(len(ngt)):
        if ngt[c] <= 0:
            continue
        imgs, tp = P["per_class"][c]
        if len(imgs) == 0:
            aps[c] = 0.0
            continue
        wd = w[imgs][:, None]
        tpc = np.cumsum(tp * wd, axis=0)
        fpc = np.cumsum((~tp) * wd, axis=0)
        recall = tpc / ngt[c]
        prec = tpc / np.maximum(tpc + fpc, 1e-16)
        for k in range(len(IOUV)):
            aps[c, k] = compute_ap(recall[:, k], prec[:, k])
    if np.all(np.isnan(aps)):
        return np.nan, np.nan
    return float(np.nanmean(aps)), float(np.nanmean(aps[:, 0]))


def bootstrap_weights(clusters, n_boot, rng):
    """(n_boot, n_images) counts: clusters are drawn with replacement, every image of a drawn cluster is counted."""
    uniq, cl = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    W = np.zeros((n_boot, len(clusters)))
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, k, size=k), minlength=k)[cl]
    return W


def mean_sd_ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    return float(np.nanmean(x)), float(np.nanstd(x, ddof=1)), float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))


# ------------------------------------------------------------------
# Data access
# ------------------------------------------------------------------
def validator_run(weights, model_name, cache_name, test_df):
    """Per-image statistics recorded from the Ultralytics validator itself (the same val() call as
    common.evaluate_on_test), so the full-test-set mAP equals the reported value. Cached on disk."""
    from ultralytics.utils import metrics as um
    from config import DATA_TEST_YAML
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{cache_name}_val.pkl")
    files = test_df["file"].tolist()
    meta = {"files": files, "weights": str(weights), "mtime": os.path.getmtime(weights), "kind": "validator-v1"}
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved["meta"] == meta:
            return saved["stats"]
    store = {}
    orig = um.DetMetrics.update_stats

    def hook(self, stat):
        store[stat["im_name"]] = {"tp": np.asarray(stat["tp"]).astype(bool).copy(),
                                  "conf": np.asarray(stat["conf"], float).copy(),
                                  "pred_cls": np.asarray(stat["pred_cls"]).astype(int).copy(),
                                  "target_cls": np.asarray(stat["target_cls"]).astype(int).copy()}
        return orig(self, stat)

    um.DetMetrics.update_stats = hook
    try:
        model = load_model(str(weights), family_hint=model_name)
        model.val(data=DATA_TEST_YAML, split="val", conf=0.001, iou=0.7, plots=False, verbose=False,
                  project=os.path.join(CACHE_DIR, "val_tmp"), name=cache_name, exist_ok=True)
    finally:
        um.DetMetrics.update_stats = orig
    missing = [f for f in files if f not in store]
    if missing:
        raise RuntimeError(f"{cache_name}: validator gave no statistics for {len(missing)} test images, e.g. {missing[:3]}")
    stats = [store[f] for f in files]
    with open(cache, "wb") as f:
        pickle.dump({"meta": meta, "stats": stats}, f)
    return stats


def prepare_run_stats(stats, n_cls):
    """Same output as prepare_run(), built from the validator's own per-image TP flags."""
    img_idx, conf, cls, tp = [], [], [], []
    G = np.zeros((len(stats), n_cls))
    for i, st in enumerate(stats):
        for c in st["target_cls"]:
            G[i, int(c)] += 1
        n = len(st["conf"])
        img_idx.append(np.full(n, i)); conf.append(st["conf"]); cls.append(st["pred_cls"]); tp.append(st["tp"].reshape(n, len(IOUV)))
    img_idx = np.concatenate(img_idx); conf = np.concatenate(conf)
    cls = np.concatenate(cls).astype(int); tp = np.concatenate(tp)
    per_class = []
    for c in range(n_cls):
        sel = np.nonzero(cls == c)[0]
        order = sel[np.argsort(-conf[sel], kind="stable")]
        per_class.append((img_idx[order], tp[order]))
    return {"G": G, "per_class": per_class}


def stem(model_name):
    return os.path.splitext(os.path.basename(str(model_name)))[0]


def collect_entities():
    """label -> list of runs (one per seed) found in the phase-4 / phase-5 result files."""
    ents = {}
    def add(label, r, model_name):
        w = Path(str(r["save_dir"])) / "weights" / "best.pt"
        ents.setdefault(label, []).append({"label": label, "seed": int(r["seed"]), "weights": w, "model": model_name,
                                           "reported": float(r["test_mAP50_95"]), "run": str(r["Run"])})
    cmp = load_csv(COMPARE_CSV)
    if not cmp.empty:
        for _, r in cmp.sort_values(["table", "model", "seed"]).iterrows():
            add(f"{r['table']}:{stem(r['model'])}", r, r["model"])
    p4 = load_csv(PHASE4_CSV)
    if not p4.empty and "role" in p4:
        for role, label in (("baseline", f"default:{stem(MODEL)}"), ("final", f"frozen:{stem(MODEL)}")):
            if label not in ents:
                for _, r in p4[p4["role"] == role].sort_values("seed").iterrows():
                    add(label, r, MODEL)
        # every Phase 4 rung (ladder) and the cls_pw ablation as their own entities
        for _, r in p4[p4["role"].notna()].sort_values(["variant", "seed"]).iterrows():
            add(f"phase4:{r['variant']}", r, MODEL)
    return ents


def comparison_pairs(ents):
    ref = stem(MODEL)
    pairs = []
    if f"frozen:{ref}" in ents and f"default:{ref}" in ents:
        pairs.append((f"frozen:{ref}", f"default:{ref}", "tuned configuration vs Baseline"))
    for table in ("frozen", "default"):
        r = f"{table}:{ref}"
        for lab in ents:
            if lab.startswith(table + ":") and lab != r and r in ents:
                pairs.append((lab, r, f"{lab} vs {r}"))
    rungs = sorted(l for l in ents if l.startswith("phase4:rung"))
    for a, b in zip(rungs[:-1], rungs[1:]):
        pairs.append((b, a, "ladder step"))
    if "phase4:cls_pw0" in ents and f"frozen:{ref}" in ents:
        pairs.append((f"frozen:{ref}", "phase4:cls_pw0", "cls_pw 1.0 vs 0.0"))
    return pairs


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=777)
    ap.add_argument("--tol", type=float, default=0.02, help="flag runs whose numpy mAP50-95 differs from val() by more")
    ap.add_argument("--allow-mismatch", action="store_true", help="continue even if the self-check fails (results NOT usable)")
    args = ap.parse_args()

    manifest = pd.read_csv(os.path.join(SPLIT_DIR, "split_manifest.csv"))
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    test_df["image_path"] = [os.path.join(SPLIT_DIR, "test", "images", f) for f in test_df["file"]]
    test_df["label_path"] = [os.path.join(SPLIT_DIR, "test", "labels", os.path.splitext(f)[0] + ".txt") for f in test_df["file"]]
    if "group" in test_df:
        clusters = test_df["group"].to_numpy()
    else:
        logger.warning("split_manifest.csv has no 'group' column: every image is treated as its own cluster.")
        clusters = np.arange(len(test_df))
    with open(DATA_TEST_YAML) as f:
        names = yaml.safe_load(f)["names"]
    n_cls = len(names)
    logger.info(f"Test images: {len(test_df)}; clusters: {len(np.unique(clusters))}; classes: {n_cls}")

    ents = collect_entities()
    if not ents:
        logger.error(f"No runs found: run 4_stepwise_evaluation.py / 5_compare_models.py first ({PHASE4_CSV}, {COMPARE_CSV}).")
        sys.exit(1)
    logger.info("Entities: " + ", ".join(f"{k} ({len(v)} seeds)" for k, v in ents.items()))

    # ---- predictions, ground truth (image sizes taken from the predictor), plain-test-set mAP (self-check) ----
    prepared, run_rows = {}, []
    for label, runs in ents.items():
        for r in runs:
            if not r["weights"].exists():
                logger.error(f"Missing weights: {r['weights']}")
                sys.exit(1)
            logger.info(f"Validating {r['run']} (per-image statistics from the Ultralytics validator) ...")
            stats = validator_run(r["weights"], r["model"], r["run"], test_df)
            P = prepare_run_stats(stats, n_cls)
            prepared[r["run"]] = P
            m5095, m50 = map_from_weights(P, np.ones(len(test_df)))
            diff = m5095 - r["reported"]
            run_rows.append({"entity": label, "run": r["run"], "seed": r["seed"], "numpy_mAP50_95": m5095,
                             "numpy_mAP50": m50, "reported_val_mAP50_95": r["reported"], "difference": diff,
                             "flag": abs(diff) > args.tol})
            if abs(diff) > args.tol:
                logger.warning(f"{r['run']}: numpy mAP50-95 {m5095:.4f} vs val() {r['reported']:.4f} (|diff| > {args.tol})")
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(OUT_RUNS, index=False)
    logger.info(f"Self-check: mean |numpy - val()| = {runs_df['difference'].abs().mean():.4f}, "
                f"max = {runs_df['difference'].abs().max():.4f}, flagged runs = {int(runs_df['flag'].sum())}")

    if runs_df["flag"].any() and not args.allow_mismatch:
        logger.error(f"Self-check FAILED for {int(runs_df['flag'].sum())} of {len(runs_df)} runs (see {OUT_RUNS}). "
                     "The numpy mAP does not reproduce val(); the bootstrap results would not be usable. Stopped. "
                     "(--allow-mismatch continues anyway, for debugging only.)")
        sys.exit(2)

    # ---- cluster bootstrap with common resamples ----
    rng = np.random.default_rng(args.boot_seed)
    W = bootstrap_weights(clusters, args.n_boot, rng)
    boot = {}
    for label, runs in ents.items():
        for r in runs:
            P = prepared[r["run"]]
            vals = np.array([map_from_weights(P, W[b]) for b in range(args.n_boot)])
            boot[r["run"]] = {"mAP50_95": vals[:, 0], "mAP50": vals[:, 1]}
        logger.info(f"bootstrap done: {label}")

    ci_rows = []
    for label, runs in ents.items():
        for metric, col in (("mAP50_95", "numpy_mAP50_95"), ("mAP50", "numpy_mAP50")):
            point = float(runs_df[runs_df["entity"] == label][col].mean())
            b = np.mean([boot[r["run"]][metric] for r in runs], axis=0)
            mean, sd, lo, hi = mean_sd_ci(b)
            ci_rows.append({"entity": label, "metric": metric, "n_seeds": len(runs), "point_mean_over_seeds": point,
                            "boot_mean": mean, "boot_sd": sd, "ci95_low": lo, "ci95_high": hi})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT_CI, index=False)

    diff_rows = []
    for a, b_, why in comparison_pairs(ents):
        ra = {r["seed"]: r for r in ents[a]}
        rb = {r["seed"]: r for r in ents[b_]}
        seeds = sorted(set(ra) & set(rb))
        if not seeds:
            logger.warning(f"{a} vs {b_}: no common seeds, skipped")
            continue
        for metric, col in (("mAP50_95", "numpy_mAP50_95"), ("mAP50", "numpy_mAP50")):
            d_boot = np.mean([boot[ra[s]["run"]][metric] - boot[rb[s]["run"]][metric] for s in seeds], axis=0)
            pa = runs_df.set_index("run").loc[[ra[s]["run"] for s in seeds], col].to_numpy()
            pb = runs_df.set_index("run").loc[[rb[s]["run"] for s in seeds], col].to_numpy()
            mean, sd, lo, hi = mean_sd_ci(d_boot)
            diff_rows.append({"comparison": why, "A": a, "B": b_, "metric": metric, "n_paired_seeds": len(seeds),
                              "point_diff": float(np.mean(pa - pb)), "boot_mean_diff": mean, "boot_sd": sd,
                              "ci95_low": lo, "ci95_high": hi, "boot_frac_diff_le_0": float(np.mean(d_boot <= 0)),
                              "clear_difference": bool(lo > 0 or hi < 0)})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUT_DIFF, index=False)

    with open(OUT_META, "w") as f:
        json.dump({"n_boot": args.n_boot, "boot_seed": args.boot_seed, "n_test_images": len(test_df),
                   "n_clusters": int(len(np.unique(clusters))), "conf": CONF, "iou": IOU, "imgsz": IMGSZ,
                   "resampling_unit": "cluster (group column of split_manifest.csv)"}, f, indent=2)

    logger.info("\nTest mAP50-95 (mean over seeds; 95% interval from resampling test clusters):\n" +
                ci[ci["metric"] == "mAP50_95"].round(4).to_string(index=False))
    if not diff.empty:
        logger.info("\nPaired differences, mAP50-95 (A - B, same resamples and same seeds):\n" +
                    diff[diff["metric"] == "mAP50_95"].round(4).to_string(index=False))
    logger.info("\nReading guide: an interval for a difference that includes 0 means the difference cannot be separated "
                "from test-image sampling variability. These intervals ignore training variability (see the seed-based "
                "results) and cover one split only. With few test images and clusters the intervals are wide and the "
                "percentile bootstrap is approximate.")


if __name__ == "__main__":
    main()
