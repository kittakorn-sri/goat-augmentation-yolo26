# ==================== 4_stepwise_evaluation.py ====================
# Phase 4: stepwise improvement (ladder) on the TEST split (phases 4-6 are the only ones that touch it).
#
# Everything chosen on the VALID split in phases 1-3 is frozen. The stepwise improvement from Baseline to the final
# configuration is reported on the test split with the SAME evaluation seeds for every step:
#
#   rung 0  Baseline                         all augmentation off, optimizer=auto
#   rung 1  best single augmentation         highest mean valid mAP50-95 among the single techniques of phase 1, optimizer=auto
#   rung 2  Phase 2 set                      phase2_selected.json, optimizer=auto
#   rung 3  Phase 2 set + optimizer          phase3_selected.json  (= the FINAL configuration)
#   ablation  final configuration with cls_pw = 0.0  (compared with cls_pw = config.CLS_PW; reported, not used for selection)
#
# The rungs are DESCRIPTIVE. They are fixed by rules that use only valid-split outputs of phases 1-3 and are written to
# final_plan.json BEFORE any test run. Every rung is reported whatever its test score is; nothing here may be used to
# change the final configuration. A rung whose configuration is identical to an earlier rung is not run again
# (e.g. phase 3 chose "auto", or the phase 2 set is one technique that is also the best single technique); it is marked
# `same_as` in the plan.
#
# Valid split in this phase: only used by Ultralytics to pick best.pt and for early stopping; no decision is made with it.
#
# Output (all seeds = config.EVALUATION_SEEDS, one test evaluation per run):
#   final_plan.json                    the pre-declared rungs and the role of each (baseline / intermediate / final)
#   results_final_test.csv             one row per (variant, seed); variants rung0..rung3 and cls_pw0
#   results_final_test_per_class.csv   per-class AP / P / R / F1 per run
#   results_final_test_summary.csv     mean, SD, 95% CI over seeds per variant and metric
#   results_final_test_steps.csv       seed-paired differences between consecutive rungs, first -> final, cls_pw 1.0 vs 0.0
#   results_final_test_per_class_summary.csv
#
# The intervals cover training variability on ONE fixed test set. They do not cover test-image sampling
# (see 6_bootstrap_test.py). A difference whose CI includes 0 must not be described as an improvement.
#
# Run: python 4_stepwise_evaluation.py             (after phases 1-3)
#      EPOCHS=2 python 4_stepwise_evaluation.py    # smoke test only; delete its outputs afterwards

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from config import EVALUATION_SEEDS, CLS_PW, SEED, set_seed, print_config, get_logger
from common import train_run, read_run_stats, has_row, append_row, load_csv, evaluate_on_test

PROJECT = "goat_v2_final"
EVAL_DIR = "goat_v2_final_eval"
PLAN_JSON = "final_plan.json"
RESULTS_CSV = "results_final_test.csv"
PERCLASS_CSV = "results_final_test_per_class.csv"
SUMMARY_CSV = "results_final_test_summary.csv"
STEPS_CSV = "results_final_test_steps.csv"
PERCLASS_SUMMARY_CSV = "results_final_test_per_class_summary.csv"

PHASE1_SUMMARY = "phase1_summary_val.csv"
PHASE2_JSON = "phase2_selected.json"
PHASE3_JSON = "phase3_selected.json"

ABLATION = "cls_pw0"
METRICS = ["test_mAP50_95", "test_mAP50", "test_Precision", "test_Recall", "test_F1_of_means",
           "val_mAP50_95", "epochs_run", "best_epoch"]
STEP_METRICS = ["test_mAP50_95", "test_mAP50", "test_Precision", "test_Recall"]

set_seed(SEED)
logger = get_logger("phase4")


# ------------------------------------------------------------------
# Pure functions (unit-testable)
# ------------------------------------------------------------------
def config_key(methods, optimizer, lr0):
    """Identity of a configuration; lr0 is irrelevant for optimizer='auto'."""
    lr = None if optimizer == "auto" or lr0 is None else round(float(lr0), 10)
    return (tuple(sorted(methods)), optimizer, lr)


def build_rungs(phase1_summary, phase2_methods, optimizer, lr0):
    """The four pre-declared rungs from VALID-split outputs only, with duplicate detection and roles."""
    single = phase1_summary[phase1_summary["Method"] != "Baseline"]
    best = single.sort_values("mean_val_mAP50_95", ascending=False).iloc[0]["Method"]
    rungs = [
        {"id": 0, "label": "Baseline", "methods": ["Baseline"], "optimizer": "auto", "lr0": None},
        {"id": 1, "label": f"Best single augmentation ({best})", "methods": [best], "optimizer": "auto", "lr0": None},
        {"id": 2, "label": "Phase 2 set (" + "+".join(phase2_methods) + ")", "methods": list(phase2_methods),
         "optimizer": "auto", "lr0": None},
        {"id": 3, "label": "Phase 2 set + optimizer", "methods": list(phase2_methods),
         "optimizer": optimizer, "lr0": None if optimizer == "auto" else lr0},
    ]
    base_key = config_key(["Baseline"], "auto", None)
    final_key = config_key(rungs[3]["methods"], rungs[3]["optimizer"], rungs[3]["lr0"])
    seen = {}
    for r in rungs:
        k = config_key(r["methods"], r["optimizer"], r["lr0"])
        r["variant"] = f"rung{r['id']}"
        r["same_as"] = seen.get(k)
        seen.setdefault(k, r["id"])
        if r["same_as"] is not None:
            r["role"] = None                    # not run; its result is that of rung `same_as`
        elif k == base_key:
            r["role"] = "baseline"
        elif k == final_key:
            r["role"] = "final"
        else:
            r["role"] = "intermediate"
    return rungs


def mean_sd_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half


def summarize(df):
    rows = []
    for (variant, label, role), g in df.groupby(["variant", "label", "role"], sort=True):
        for mt in METRICS:
            if mt not in g:
                continue
            mean, sd, lo, hi = mean_sd_ci(g[mt])
            rows.append({"variant": variant, "label": label, "role": role, "metric": mt, "n_seeds": len(g),
                         "mean": mean, "sd": sd, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def step_table(df, rungs):
    """Seed-paired differences (to - from): consecutive distinct rungs, first -> last distinct rung, and the cls_pw ablation."""
    run_rungs = [r for r in rungs if r["same_as"] is None]
    labels = {r["variant"]: r["label"] for r in run_rungs}
    labels[ABLATION] = "Final configuration with cls_pw = 0.0"
    pairs = [(a["variant"], b["variant"], "step") for a, b in zip(run_rungs[:-1], run_rungs[1:])]
    if len(run_rungs) > 2:
        pairs.append((run_rungs[0]["variant"], run_rungs[-1]["variant"], "cumulative"))
    final_variant = next((r["variant"] for r in run_rungs if r["role"] == "final"), None)
    if final_variant is not None:
        pairs.append((ABLATION, final_variant, "cls_pw ablation (cls_pw 1.0 minus 0.0)"))
    rows = []
    for a, b, kind in pairs:
        for mt in STEP_METRICS:
            piv = df.pivot_table(index="seed", columns="variant", values=mt)
            if a not in piv.columns or b not in piv.columns:
                continue
            d = (piv[b] - piv[a]).dropna()
            if len(d) < 2:
                continue
            mean, sd, lo, hi = mean_sd_ci(d)
            rows.append({"kind": kind, "from": a, "to": b, "from_label": labels.get(a, a), "to_label": labels.get(b, b),
                         "metric": mt, "n_pairs": len(d), "mean_diff": mean, "sd_diff": sd,
                         "ci95_low": lo, "ci95_high": hi, "clear_difference": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Plan (pre-declaration)
# ------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def declare_plan():
    for p in (PHASE1_SUMMARY, PHASE2_JSON, PHASE3_JSON):
        if not os.path.exists(p):
            logger.error(f"{p} missing: run phases 1-3 first.")
            sys.exit(1)
    p3 = load_json(PHASE3_JSON)
    rungs = build_rungs(pd.read_csv(PHASE1_SUMMARY), load_json(PHASE2_JSON)["methods"], p3["optimizer"], p3["lr0"])
    if os.path.exists(PLAN_JSON):
        old = load_json(PLAN_JSON)["rungs"]
        same = len(old) == len(rungs) and all(
            config_key(o["methods"], o["optimizer"], o["lr0"]) == config_key(n["methods"], n["optimizer"], n["lr0"])
            for o, n in zip(old, rungs))
        if not same:
            logger.error(f"{PLAN_JSON} already exists with DIFFERENT rungs. The plan is fixed before test runs; "
                         "if phases 1-3 were genuinely rerun, back up and delete the old plan and its results.")
            sys.exit(1)
        return rungs
    with open(PLAN_JSON, "w") as f:
        json.dump({"seeds": list(EVALUATION_SEEDS), "cls_pw": CLS_PW, "rungs": rungs}, f, indent=2)
    logger.info(f"Plan written to {PLAN_JSON} before any test evaluation.")
    return rungs


# ------------------------------------------------------------------
def save_per_class(pc, run):
    pc = pc.copy()
    pc["run"] = run
    old = load_csv(PERCLASS_CSV)
    if not old.empty:
        old = old[old["run"] != run]
    pd.concat([old, pc], ignore_index=True).to_csv(PERCLASS_CSV, index=False)


def run_one(entry, seed):
    """entry: dict with variant, label, role, rung (int or None), methods, optimizer, lr0, cls_pw."""
    run = f"{entry['variant']}_s{seed}"
    if has_row(RESULTS_CSV, "Run", run):
        logger.info(f"[{run}] done -> skip")
        return
    logger.info(f"\n=== {run}: {entry['label']} (optimizer={entry['optimizer']}, lr0={entry['lr0']}, cls_pw={entry['cls_pw']}) ===")
    save_dir = train_run(PROJECT, run, entry["methods"], entry["optimizer"], entry["lr0"], seed, cls_pw=entry["cls_pw"])
    overall, per_class = evaluate_on_test(os.path.join(str(save_dir), "weights", "best.pt"), run, EVAL_DIR)
    append_row(RESULTS_CSV, "Run", {"Run": run, "variant": entry["variant"], "rung": entry["rung"],
                                    "label": entry["label"], "role": entry["role"], "seed": seed,
                                    "methods": "+".join(entry["methods"]), "optimizer": entry["optimizer"],
                                    "lr0": entry["lr0"], "cls_pw": entry["cls_pw"],
                                    **overall, **read_run_stats(save_dir)})
    save_per_class(per_class, run)
    logger.info(f"[{run}] test mAP50-95 = {overall['test_mAP50_95']:.4f}")


def main():
    print_config()
    rungs = declare_plan()
    for r in rungs:
        logger.info(f"rung {r['id']}: {r['label']}  methods={r['methods']} optimizer={r['optimizer']} lr0={r['lr0']} "
                    f"role={r['role']}" + (f"  (same as rung {r['same_as']}; not run again)" if r["same_as"] is not None else ""))

    entries = [{**r, "rung": r["id"], "cls_pw": CLS_PW} for r in rungs if r["same_as"] is None]
    final = next(e for e in entries if e["role"] == "final")
    entries.append({"variant": ABLATION, "label": "Final configuration with cls_pw = 0.0", "role": "ablation",
                    "rung": None, "methods": final["methods"], "optimizer": final["optimizer"],
                    "lr0": final["lr0"], "cls_pw": 0.0})
    for e in entries:
        for s in EVALUATION_SEEDS:
            run_one(e, s)

    df = load_csv(RESULTS_CSV)
    if df.empty:
        return
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    steps = step_table(df, rungs)
    steps.to_csv(STEPS_CSV, index=False)

    pc = load_csv(PERCLASS_CSV)
    if not pc.empty:
        pc["variant"] = pc["run"].str.rsplit("_s", n=1).str[0]
        pc_sum = pc.groupby(["variant", "class"])[["AP50", "AP50_95", "Precision", "Recall", "F1"]] \
            .agg(["mean", "std"]).round(4)
        pc_sum.to_csv(PERCLASS_SUMMARY_CSV)
        logger.info("\nPer-class (mean, SD over seeds):\n" + pc_sum.to_string())

    logger.info("\nTest metrics over seeds (mean, SD, 95% CI):\n" +
                summary[summary["metric"].isin(["test_mAP50_95", "test_Precision", "test_Recall"])].round(4).to_string(index=False))
    if not steps.empty:
        logger.info("\nSeed-paired differences:\n" + steps[steps["metric"] == "test_mAP50_95"].round(4).to_string(index=False))
    logger.info("\nReading guide: the rungs were fixed before testing and must not be used to change the final configuration. "
                "Intervals cover training variability on one fixed test set only (not test-image sampling: see "
                "6_bootstrap_test.py); a CI for a difference that includes 0 is not evidence of improvement.")


if __name__ == "__main__":
    main()
