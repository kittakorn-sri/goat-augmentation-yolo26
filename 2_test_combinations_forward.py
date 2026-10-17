# ==================== 2_test_combinations_forward.py ====================
# Phase 2: forward selection on the VALID split, starting from Baseline (no technique).
# Every combination is trained with the SAME selection seeds as phase 1. At each step the candidate with the
# largest mean seed-paired gain over the CURRENT set is tested; it is accepted only if the one-sided lower
# confidence bound of that gain is above 0 (config.ACCEPT_CONFIDENCE). Otherwise the search stops.
# Limits (config): candidates = phase-1 shortlist, MAX_SET_SIZE (None = unlimited), at most one MIXING_GROUP technique.
# Phase 4 later confirms the final configuration with new seeds on the test split.
# Run: python 2_test_combinations_forward.py

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import (SELECTION_SEEDS, ACCEPT_RULE, ACCEPT_CONFIDENCE, MAX_SET_SIZE, MIXING_GROUP, SELECTION_METRIC,
                    set_seed, print_config, get_logger)
from common import train_run, read_run_stats, get_row, append_row, load_csv, paired_summary

PROJECT = "goat_v2_phase2"
PHASE1_CSV = "results_phase1_val.csv"
RESULTS_CSV = "results_phase2_val.csv"           # one row per (combination, seed)
STEPS_CSV = "phase2_steps_val.csv"               # every candidate tested at every step
SELECTED_JSON = "phase2_selected.json"

set_seed(SELECTION_SEEDS[0])
print_config()
logger = get_logger("phase2_forward")


def combo_scores(methods):
    """Per-seed rows for a combination; missing seeds are trained (single techniques reuse the phase-1 runs)."""
    name = "+".join(methods)
    rows = []
    for s in SELECTION_SEEDS:
        run = f"{name}_s{s}"
        row = get_row(RESULTS_CSV, "Run", run)
        if row is None:
            if len(methods) == 1:
                p1 = get_row(PHASE1_CSV, "Run", f"{methods[0]}_s{s}")
                row = {"Run": run, "Combination": name, "Methods": str(methods), "num_methods": 1, "seed": s,
                       **{k: v for k, v in p1.items() if k not in ("Run", "Method", "seed")}}
            else:
                logger.info(f"\n=== {run} ===")
                save_dir = train_run(PROJECT, run, methods, "auto", None, s)
                row = {"Run": run, "Combination": name, "Methods": str(methods), "num_methods": len(methods),
                       "seed": s, **read_run_stats(save_dir)}
            append_row(RESULTS_CSV, "Run", row)
        rows.append(row)
    return pd.DataFrame(rows).set_index("seed")


def accept(ps):
    """Acceptance decision for the best candidate of a step (ps = paired_summary of its gain)."""
    if ACCEPT_RULE == "mean":
        return bool(ps["mean_diff"] > 0)
    if ACCEPT_RULE == "lower_bound":
        return bool(ps["passes"])
    raise ValueError(f"unknown ACCEPT_RULE: {ACCEPT_RULE}")


def main():
    if not os.path.exists("phase1_shortlist.json"):
        logger.error("Run 1_train_augmentation.py first.")
        sys.exit(1)
    with open("phase1_shortlist.json") as f:
        shortlist = json.load(f)["shortlist"]
    df1 = load_csv(PHASE1_CSV)
    base_rows = df1[df1["Method"] == "Baseline"].set_index("seed")
    cur_scores = base_rows[SELECTION_METRIC]
    rule_text = ("mean paired gain over the current set > 0" if ACCEPT_RULE == "mean"
                 else f"one-sided {ACCEPT_CONFIDENCE:.0%} lower bound of the paired gain > 0")
    logger.info(f"Candidates (phase-1 shortlist): {shortlist}\n"
                f"Rule: best candidate by mean gain; accept if {rule_text}; "
                f"max set size {'unlimited' if MAX_SET_SIZE is None else MAX_SET_SIZE}; at most one of {MIXING_GROUP}")

    current, cur_rows = [], base_rows
    step = 1
    while MAX_SET_SIZE is None or len(current) < MAX_SET_SIZE:
        cands = [m for m in shortlist if m not in current]
        if any(c in MIXING_GROUP for c in current):
            cands = [m for m in cands if m not in MIXING_GROUP]
        if not cands:
            logger.info("No candidates left.")
            break
        logger.info(f"\n--- Step {step}: current {current or ['Baseline']}; candidates {cands}")
        tested = {}
        for m in cands:
            rows = combo_scores(current + [m])
            ps = paired_summary(rows[SELECTION_METRIC], cur_scores, ACCEPT_CONFIDENCE)
            tested[m] = (ps, rows)
        best = max(tested, key=lambda m: tested[m][0]["mean_diff"])
        for m, (ps, rows) in tested.items():
            append_row(STEPS_CSV, "Key", {
                "Key": f"{step}|{'+'.join(current) or 'Baseline'}|{m}", "step": step,
                "current": "+".join(current) or "Baseline", "candidate": m,
                "mean_val_mAP50_95": rows[SELECTION_METRIC].mean(),
                "mean_val_Precision": rows["val_Precision"].mean(), "mean_val_Recall": rows["val_Recall"].mean(),
                **{f"gain_{k}": v for k, v in ps.items()},
                "best_at_step": m == best, "accepted": bool(m == best and accept(ps))})
            logger.info(f"  +{m}: gain {ps['mean_diff']:+.4f} (SD {ps['sd_diff']:.4f}), "
                        f"lower bound {ps['lb_one_sided']:+.4f}, p={ps['p_one_sided']:.3f}")
        ps, rows = tested[best]
        if not accept(ps):
            logger.info(f"Best candidate {best}: mean gain {ps['mean_diff']:+.4f}, lower bound {ps['lb_one_sided']:+.4f} "
                        f"-> does not meet the '{ACCEPT_RULE}' rule; stop.")
            break
        current.append(best)
        cur_scores, cur_rows = rows[SELECTION_METRIC], rows
        logger.info(f"Accepted {best} -> {current}")
        step += 1

    selected = current or ["Baseline"]
    out = {"methods": selected, "val_mAP50_95": float(cur_scores.mean()), "n_seeds": len(SELECTION_SEEDS),
           "mean_val_Precision": float(cur_rows["val_Precision"].mean()),
           "mean_val_Recall": float(cur_rows["val_Recall"].mean()),
           "acceptance_rule": ACCEPT_RULE,
           "rule": (f"mean seed-paired gain over the current set > 0" if ACCEPT_RULE == "mean"
                    else f"one-sided {ACCEPT_CONFIDENCE:.0%} lower bound of seed-paired gain > 0") + f"; max set size {'unlimited' if MAX_SET_SIZE is None else MAX_SET_SIZE}"}
    with open(SELECTED_JSON, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSelected (valid split): {selected}  mean valid mAP50-95 = {out['val_mAP50_95']:.4f}")
    if selected == ["Baseline"]:
        logger.info("No technique met the acceptance rule; the paper should say that no augmentation improved on Baseline.")


if __name__ == "__main__":
    main()
