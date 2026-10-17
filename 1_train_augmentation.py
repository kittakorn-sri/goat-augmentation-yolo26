# ==================== 1_train_augmentation.py ====================
# Phase 1: every augmentation individually, each trained with the SAME 5 selection seeds (config.SELECTION_SEEDS).
# Scores are VALID-split scores; the test split is not used.
# Each technique is compared with Baseline PAIRED BY SEED (mean difference, SD, one-sided lower bound, p).
# Output: the shortlist for phase 2 = every technique whose mean seed-paired difference vs Baseline is > 0
# (config.SHORTLIST_MAX = None; an integer would cap the list at the best N).
# With 15 comparisons these numbers are exploratory screening, not confirmed effects; confirmation is phase 4.
# Run: python 1_train_augmentation.py   (after 0a_prepare_dataset.py)

import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import (SELECTION_SEEDS, ACCEPT_CONFIDENCE, SHORTLIST_MAX, SELECTION_METRIC,
                    augmentation_configs, set_seed, print_config, get_logger)
from common import train_run, read_run_stats, has_row, append_row, load_csv, paired_summary

PROJECT = "goat_v2_phase1"
RESULTS_CSV = "results_phase1_val.csv"          # one row per (method, seed)
SUMMARY_CSV = "phase1_summary_val.csv"          # one row per method
SHORTLIST_JSON = "phase1_shortlist.json"

set_seed(SELECTION_SEEDS[0])
print_config()
logger = get_logger("phase1")


def main():
    for name in augmentation_configs:
        for s in SELECTION_SEEDS:
            run = f"{name}_s{s}"
            if has_row(RESULTS_CSV, "Run", run):
                logger.info(f"[{run}] done -> skip")
                continue
            logger.info(f"\n=== {run} (optimizer=auto) ===")
            save_dir = train_run(PROJECT, run, [name], "auto", None, s)
            row = {"Run": run, "Method": name, "seed": s, **read_run_stats(save_dir)}
            append_row(RESULTS_CSV, "Run", row)
            logger.info(f"[{run}] valid mAP50-95 = {row['val_mAP50_95']:.4f} "
                        f"(epochs run {row['epochs_run']}, best {row['best_epoch']}, lr_peak {row['lr_peak']:.5f})")

    df = load_csv(RESULTS_CSV)
    piv = df.pivot_table(index="seed", columns="Method", values=SELECTION_METRIC)
    base = piv["Baseline"]
    rows = []
    for m in augmentation_configs:
        g = df[df["Method"] == m]
        row = {"Method": m, "n_seeds": len(g),
               "mean_val_mAP50_95": g["val_mAP50_95"].mean(), "sd_val_mAP50_95": g["val_mAP50_95"].std(ddof=1),
               "mean_val_Precision": g["val_Precision"].mean(), "mean_val_Recall": g["val_Recall"].mean(),
               "mean_val_F1": g["val_F1"].mean(), "mean_epochs_run": g["epochs_run"].mean()}
        if m != "Baseline":
            ps = paired_summary(piv[m], base, ACCEPT_CONFIDENCE)
            row.update({f"vs_baseline_{k}": v for k, v in ps.items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    cand = summary[(summary["Method"] != "Baseline") & (summary["vs_baseline_mean_diff"] > 0)] \
        .sort_values("vs_baseline_mean_diff", ascending=False)
    shortlist = cand["Method"].tolist() if SHORTLIST_MAX is None else cand["Method"].head(SHORTLIST_MAX).tolist()
    with open(SHORTLIST_JSON, "w") as f:
        json.dump({"shortlist": shortlist, "n_seeds": len(SELECTION_SEEDS), "seeds": SELECTION_SEEDS,
                   "rule": f"mean seed-paired difference vs Baseline > 0 on {SELECTION_METRIC}"
                           + ("" if SHORTLIST_MAX is None else f"; top {SHORTLIST_MAX}")}, f, indent=2)

    cols = ["Method", "mean_val_mAP50_95", "sd_val_mAP50_95", "vs_baseline_mean_diff", "vs_baseline_sd_diff",
            "vs_baseline_lb_one_sided", "vs_baseline_p_one_sided", "mean_val_Precision", "mean_val_Recall"]
    logger.info("\nPhase 1 summary (valid split; differences paired by seed vs Baseline):\n" +
                summary[cols].sort_values("vs_baseline_mean_diff", ascending=False, na_position="last").round(4).to_string(index=False))
    logger.info(f"\nShortlist for phase 2: {shortlist}")
    logger.info("lb_one_sided > 0 means 'better than Baseline' at the pre-set confidence; with 15 comparisons treat these as "
                "screening results. Precision/recall are shown for context (not used for selection).")


if __name__ == "__main__":
    main()
