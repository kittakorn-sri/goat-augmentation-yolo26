# ==================== 3_train_optimizer.py ====================
# Phase 3: optimizer comparison with the phase-2 augmentation set, on the VALID split.
# Stage A: every optimizer x learning rate of config.OPTIMIZER_LR_GRID is trained with ONE seed (LR_SELECTION_SEED);
#          the best learning rate per optimizer is picked on valid. (single-seed screening, noisy by design)
# Stage B: the chosen (optimizer, lr) is trained with the remaining selection seeds -> all selection seeds.
# "auto" (the setting of phases 1-2, already trained on all selection seeds) is the reference: another optimizer
# replaces it only if the one-sided lower confidence bound of its seed-paired gain is above 0.
# Run: python 3_train_optimizer.py

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import (SELECTION_SEEDS, LR_SELECTION_SEED, OPTIMIZER_CONFIGS, OPTIMIZER_LR_GRID, ACCEPT_CONFIDENCE,
                    SELECTION_METRIC, set_seed, print_config, get_logger)
from common import train_run, read_run_stats, get_row, append_row, load_csv, paired_summary

PROJECT = "goat_v2_phase3"
PHASE1_CSV = "results_phase1_val.csv"
PHASE2_CSV = "results_phase2_val.csv"
RESULTS_CSV = "results_phase3_val.csv"           # one row per (optimizer, lr, seed)
SUMMARY_CSV = "phase3_summary_val.csv"
SELECTED_JSON = "phase3_selected.json"

set_seed(SELECTION_SEEDS[0])
print_config()
logger = get_logger("phase3")


def run_opt(methods, opt, lr, seed):
    key = f"opt_{opt}_lr{lr:g}_s{seed}"
    row = get_row(RESULTS_CSV, "Run", key)
    if row is not None:
        return row
    logger.info(f"\n=== {key} ===")
    save_dir = train_run(PROJECT, key, methods, opt, lr, seed)
    row = {"Run": key, "Optimizer": opt, "lr0": lr, "seed": seed,
           "stage": "lr_screen" if seed == LR_SELECTION_SEED else "final", **read_run_stats(save_dir)}
    append_row(RESULTS_CSV, "Run", row)
    logger.info(f"[{key}] valid mAP50-95 = {row['val_mAP50_95']:.4f} (epochs run {row['epochs_run']})")
    return row


def main():
    if not os.path.exists("phase2_selected.json"):
        logger.error("Run 2_test_combinations_forward.py first.")
        sys.exit(1)
    with open("phase2_selected.json") as f:
        methods = json.load(f)["methods"]
    logger.info(f"Augmentation set from phase 2: {methods}")

    # reference = phase-2 runs of exactly this set with optimizer=auto (all selection seeds)
    if methods == ["Baseline"]:
        d1 = load_csv(PHASE1_CSV)
        ref = d1[d1["Method"] == "Baseline"].set_index("seed")
    else:
        d2 = load_csv(PHASE2_CSV)
        ref = d2[d2["Combination"] == "+".join(methods)].set_index("seed")

    # Stage A: learning-rate screen with one seed
    best_lr = {}
    for opt in OPTIMIZER_CONFIGS:
        scores = {lr: run_opt(methods, opt, lr, LR_SELECTION_SEED)[SELECTION_METRIC] for lr in OPTIMIZER_LR_GRID[opt]}
        best_lr[opt] = max(scores, key=scores.get)
        logger.info(f"{opt}: lr screen (seed {LR_SELECTION_SEED}) {scores} -> {best_lr[opt]}")

    # Stage B: all selection seeds for the chosen lr
    rows_summary, per_opt = [], {}
    for opt in OPTIMIZER_CONFIGS:
        recs = [run_opt(methods, opt, best_lr[opt], s) for s in SELECTION_SEEDS]
        g = pd.DataFrame(recs).set_index("seed")
        per_opt[opt] = g
        ps = paired_summary(g[SELECTION_METRIC], ref[SELECTION_METRIC], ACCEPT_CONFIDENCE)
        rows_summary.append({"Optimizer": opt, "lr0": best_lr[opt], "n_seeds": len(g),
                             "mean_val_mAP50_95": g[SELECTION_METRIC].mean(), "sd_val_mAP50_95": g[SELECTION_METRIC].std(ddof=1),
                             "mean_val_Precision": g["val_Precision"].mean(), "mean_val_Recall": g["val_Recall"].mean(),
                             "mean_val_F1": g["val_F1"].mean(), "mean_epochs_run": g["epochs_run"].mean(),
                             **{f"vs_auto_{k}": v for k, v in ps.items()}})
        logger.info(f"  {opt} (lr0={best_lr[opt]:g}): mean {g[SELECTION_METRIC].mean():.4f} (SD {g[SELECTION_METRIC].std(ddof=1):.4f}); "
                    f"gain vs auto {ps['mean_diff']:+.4f}, lower bound {ps['lb_one_sided']:+.4f}, "
                    f"p(one-sided) {ps['p_one_sided']:.3f} -> {'PASS' if ps['passes'] else 'fail'}")
    summary = pd.DataFrame(rows_summary).sort_values("mean_val_mAP50_95", ascending=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    cols = ["Optimizer", "lr0", "mean_val_mAP50_95", "sd_val_mAP50_95", "vs_auto_mean_diff", "vs_auto_lb_one_sided",
            "vs_auto_p_one_sided", "mean_val_Precision", "mean_val_Recall"]
    logger.info(f"\nauto (reference): mean valid mAP50-95 = {ref[SELECTION_METRIC].mean():.4f}")
    logger.info("Optimizers (best lr each; differences paired by seed vs auto):\n" + summary[cols].round(4).to_string(index=False))

    top = summary.iloc[0]
    if bool(top["vs_auto_passes"]):
        out = {"methods": methods, "optimizer": str(top["Optimizer"]), "lr0": float(top["lr0"]),
               "val_mAP50_95": float(top["mean_val_mAP50_95"]), "vs_auto": {k.replace("vs_auto_", ""): top[k] for k in top.index if k.startswith("vs_auto_")}}
        logger.info(f"\nSelected: {out['optimizer']} lr0={out['lr0']} (beats auto at the pre-set criterion)")
    else:
        out = {"methods": methods, "optimizer": "auto", "lr0": None, "val_mAP50_95": float(ref[SELECTION_METRIC].mean()),
               "vs_auto": None}
        logger.info(f"\nNo optimizer beat auto at the pre-set criterion -> keep optimizer=auto "
                    f"(best alternative: {top['Optimizer']}, gain {top['vs_auto_mean_diff']:+.4f}, "
                    f"lower bound {top['vs_auto_lb_one_sided']:+.4f})")
    with open(SELECTED_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
