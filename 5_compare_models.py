# ==================== 5_compare_models.py ====================
# Phase 5: model comparison requested by the reviewer ("same protocol").
#
# Same for every model: split (v2), imgsz / batch / epochs / patience, pretrained (COCO) weights,
# EVALUATION_SEEDS, one test evaluation per run (conf=0.001, IoU=0.7), identical metrics.
#
# Table "frozen":  augmentation + optimizer chosen in phases 1-3 for YOLO26n, applied to every model.
#                  (YOLO26n rows are reused from phase 4, role final / baseline, when present.)
#                  Caveat: that configuration was tuned on YOLO26n, so the other models are slightly disadvantaged.
# Table "default": Baseline augmentation + optimizer=auto for every model (--default-table).
#                  If both tables lead to the same conclusion, the conclusion is more robust.
#
# RT-DETR always uses optimizer=auto (its own default). If the frozen optimizer is not available for a model
# (NotImplementedError), that run falls back to auto and the fallback is recorded in `optimizer_used`.
#
# Usage:
#   python 5_compare_models.py                     # frozen table, models = config.COMPARE_MODELS
#   python 5_compare_models.py --default-table     # also the default-settings table
#   python 5_compare_models.py --include-optional  # add config.COMPARE_MODELS_OPTIONAL (RT-DETR)
#   EPOCHS=2 python 5_compare_models.py --seeds 101 --models yolo11n.pt --skip-benchmark   # smoke test

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from config import (EVALUATION_SEEDS, CLS_PW, MODEL, COMPARE_MODELS, COMPARE_MODELS_OPTIONAL,
                    LOGS_DIR, SEED, set_seed, print_config, get_logger)
from common import (train_run, has_row, append_row, load_csv, evaluate_on_test,
                    benchmark_model, dump_transform_source, family_of, read_run_stats)

PROJECT = "goat_v2_compare"
EVAL_DIR = "goat_v2_compare_eval"
PHASE4_CSV = "results_final_test.csv"
PHASE4_PERCLASS_CSV = "results_final_test_per_class.csv"
RESULTS_CSV = "results_compare_test.csv"
PERCLASS_CSV = "results_compare_test_per_class.csv"
BENCH_CSV = "results_compare_benchmark.csv"
SUMMARY_CSV = "results_compare_summary.csv"
PAIRED_CSV = "results_compare_paired_vs_reference.csv"

METRICS = ["test_mAP50_95", "test_mAP50", "test_Precision", "test_Recall", "test_F1_of_means",
           "val_mAP50_95", "epochs_run", "best_epoch"]

set_seed(SEED)
print_config()
logger = get_logger("phase5")


# ------------------------------------------------------------------
# Statistics (pure functions)
# ------------------------------------------------------------------
def mean_sd_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half


def summarize(df):
    rows = []
    for (table, model), g in df.groupby(["table", "model"]):
        for mt in METRICS:
            if mt not in g:
                continue
            mean, sd, lo, hi = mean_sd_ci(g[mt])
            rows.append({"table": table, "model": model, "metric": mt, "n_seeds": len(g),
                         "mean": mean, "sd": sd, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def paired_vs_reference(df, ref_model, metric="test_mAP50_95"):
    """Per-seed paired differences (model - reference). Covers training variability on one fixed test set only."""
    rows = []
    for table, g in df.groupby("table"):
        piv = g.pivot_table(index="seed", columns="model", values=metric)
        if ref_model not in piv.columns:
            continue
        for m in piv.columns:
            if m == ref_model:
                continue
            d = (piv[m] - piv[ref_model]).dropna()
            n = len(d)
            if n < 2:
                continue
            mean, sd, lo, hi = mean_sd_ci(d)
            rows.append({"table": table, "model": m, "reference": ref_model, "metric": metric,
                         "n_pairs": n, "mean_diff": mean, "sd_diff": sd, "ci95_low": lo, "ci95_high": hi,
                         "clear_difference": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Running
# ------------------------------------------------------------------
def stem(model_name):
    return os.path.splitext(os.path.basename(model_name))[0]


def save_per_class(pc, run, table, model):
    pc = pc.copy()
    pc["run"], pc["table"], pc["model"] = run, table, model
    old = load_csv(PERCLASS_CSV)
    if not old.empty:
        old = old[old["run"] != run]
    pd.concat([old, pc], ignore_index=True).to_csv(PERCLASS_CSV, index=False)


def reuse_phase4(run, table, model_name, seed):
    """YOLO26n rows come from phase 4 so the reference is not trained twice:
    frozen table  <- Phase 4 rung with role 'final'; default table <- rung with role 'baseline'
    (the Baseline rung is the default-settings YOLO26n: no augmentation set, auto optimizer, same cls_pw)."""
    p4 = load_csv(PHASE4_CSV)
    if p4.empty or "role" not in p4:
        return False
    role = {"frozen": "final", "default": "baseline"}.get(table)
    hit = p4[(p4["role"] == role) & (p4["seed"] == seed)]
    if hit.empty:
        return False
    h = hit.iloc[0].to_dict()
    if float(h["cls_pw"]) != float(CLS_PW):
        return False
    keep = {k: h[k] for k in h if k.startswith(("test_", "val_")) or k in ("epochs_run", "best_epoch", "lr_peak", "save_dir")}
    append_row(RESULTS_CSV, "Run", {"Run": run, "table": table, "model": model_name, "seed": seed,
                                    "optimizer_used": h["optimizer"], "lr0": h["lr0"], "cls_pw": h["cls_pw"],
                                    "reused_from": h["Run"], **keep})
    pc4 = load_csv(PHASE4_PERCLASS_CSV)
    if not pc4.empty:
        sub = pc4[pc4["run"] == h["Run"]].drop(columns=["run", "variant"], errors="ignore")
        if not sub.empty:
            save_per_class(sub, run, table, model_name)
    logger.info(f"[{run}] reused from phase 4 ({h['Run']})")
    return True


def run_one(table, model_name, seed, methods, optimizer, lr0):
    run = f"{table}_{stem(model_name)}_s{seed}"
    if has_row(RESULTS_CSV, "Run", run):
        logger.info(f"[{run}] done -> skip")
        return
    if model_name == MODEL and reuse_phase4(run, table, model_name, seed):
        return

    opt, lr = optimizer, lr0
    if family_of(model_name) == "rtdetr":
        opt, lr = "auto", None                   # RT-DETR keeps its own default optimizer
    used_opt, used_lr, run_dir_name = opt, lr, run
    logger.info(f"\n=== {run} (optimizer={opt}, lr0={lr}, cls_pw={CLS_PW}) ===")
    try:
        save_dir = train_run(PROJECT, run, methods, opt, lr, seed, cls_pw=CLS_PW, model_name=model_name)
    except NotImplementedError as e:             # optimizer name unknown to this model/version
        if opt == "auto":
            raise
        logger.warning(f"[{run}] optimizer {opt} unavailable ({e}); falling back to auto")
        used_opt, used_lr, run_dir_name = "auto", None, run + "_autofallback"
        save_dir = train_run(PROJECT, run_dir_name, methods, "auto", None, seed, cls_pw=CLS_PW, model_name=model_name)

    weights = os.path.join(str(save_dir), "weights", "best.pt")
    overall, per_class = evaluate_on_test(weights, run, EVAL_DIR, family_hint=model_name)
    append_row(RESULTS_CSV, "Run", {"Run": run, "table": table, "model": model_name, "seed": seed,
                                    "optimizer_used": used_opt, "lr0": used_lr, "cls_pw": CLS_PW,
                                    **overall, **read_run_stats(save_dir)})
    save_per_class(per_class, run, table, model_name)
    logger.info(f"[{run}] test mAP50-95 = {overall['test_mAP50_95']:.4f}")


def run_benchmarks(models):
    df = load_csv(RESULTS_CSV)
    for m in models:
        if has_row(BENCH_CSV, "model", m):
            continue
        rows = df[df["model"] == m]
        if rows.empty:
            continue
        weights = os.path.join(str(rows.iloc[0]["save_dir"]), "weights", "best.pt")
        logger.info(f"\nBenchmark {m} ({weights})")
        b = benchmark_model(weights, family_hint=m)
        append_row(BENCH_CSV, "model", {"model": m, **b})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="override config.COMPARE_MODELS")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--default-table", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override config.EVALUATION_SEEDS")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    seeds = args.seeds or list(EVALUATION_SEEDS)
    others = args.models if args.models is not None else list(COMPARE_MODELS)
    if args.include_optional:
        others += [m for m in COMPARE_MODELS_OPTIONAL if m not in others]
    models = [MODEL] + [m for m in others if m != MODEL]
    logger.info(f"Models: {models}\nSeeds: {seeds}")
    logger.warning("CHECK before reporting: (1) does cls_pw have any effect for the non-YOLO26 models in your installed "
                   "version? It is passed identically to all models; if a model ignores it, say so in the paper. "
                   "(2) RT-DETR: read the dumped transform source to see which augmentations it really builds.")

    # evidence about the installed Ultralytics augmentation pipeline (answers 'which augmentations are active?')
    for fam in sorted({family_of(m) for m in models}):
        path = dump_transform_source(fam, os.path.join(LOGS_DIR, f"transform_source_{fam}.txt"))
        logger.info(f"Transform source for '{fam}' written to {path}")

    if not os.path.exists("phase3_selected.json"):
        logger.error("Run phases 1-3 first (phase3_selected.json missing).")
        sys.exit(1)
    with open("phase3_selected.json") as f:
        sel = json.load(f)
    methods, optimizer, lr0 = sel["methods"], sel["optimizer"], sel["lr0"]
    logger.info(f"Frozen configuration (tuned on {MODEL}): augmentation={methods}, optimizer={optimizer}, lr0={lr0}")

    for model_name in models:
        for s in seeds:
            run_one("frozen", model_name, s, methods, optimizer, lr0)
    if args.default_table:
        for model_name in models:
            for s in seeds:
                run_one("default", model_name, s, ["Baseline"], "auto", None)

    df = load_csv(RESULTS_CSV)
    if df.empty:
        return
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    paired = paired_vs_reference(df, MODEL)
    paired.to_csv(PAIRED_CSV, index=False)

    if not args.skip_benchmark:
        run_benchmarks(models)

    logger.info("\nTest metrics over seeds (mean, SD, 95% CI):\n" +
                summary[summary["metric"].isin(["test_mAP50_95", "test_Precision", "test_Recall"])].to_string(index=False))
    if not paired.empty:
        logger.info(f"\nPaired difference vs {MODEL} (test mAP50-95):\n" + paired.to_string(index=False))
    bench = load_csv(BENCH_CSV)
    if not bench.empty:
        cols = [c for c in ["model", "params_M", "GFLOPs", "weights_MB", "gpu", "gpu_infer_ms", "gpu_wall_ms_mean",
                            "cpu_infer_ms", "cpu_wall_ms_mean"] if c in bench]
        logger.info("\nSize and speed (batch 1, this machine only):\n" + bench[cols].to_string(index=False))
    logger.info("\nReading guide: CIs and paired differences cover training variability on ONE split and ONE test set "
                "(68 instances). If a CI for the difference includes 0, do not call one model better. "
                "Latency is measured on this machine; it does not establish edge-device or concurrent-user performance.")


if __name__ == "__main__":
    main()
