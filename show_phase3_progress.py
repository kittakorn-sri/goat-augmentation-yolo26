"""Print the Phase-3 progress table from results_phase3_val.csv (safe to run while Phase 3 is running).
Usage: python show_phase3_progress.py      (run inside the goat_v2 folder)"""
import json, pandas as pd, numpy as np
from scipy import stats
d = pd.read_csv("results_phase3_val.csv")
methods = json.load(open("phase2_selected.json"))["methods"]
ref = pd.read_csv("results_phase2_val.csv")
ref = ref[ref["Combination"] == "+".join(methods)].set_index("seed")["val_mAP50_95"]
print(f"auto reference mean = {ref.mean():.4f} (n={len(ref)})\n")
print("lr screen (seed 7):")
for o, g in d[d.stage == "lr_screen"].groupby("Optimizer", sort=False):
    g = g.sort_values("lr0", ascending=False)
    print(f"  {o:8s} " + "  ".join(f"{l:g}:{v:.4f}" for l, v in zip(g.lr0, g.val_mAP50_95)))
print("\nfinal (5 seeds at chosen lr0):")
for o, g in d.groupby("Optimizer", sort=False):
    best = g[g.stage == "lr_screen"].sort_values("val_mAP50_95").iloc[-1]["lr0"]
    f = d[(d.Optimizer == o) & (d.lr0 == best)].drop_duplicates("seed").set_index("seed")["val_mAP50_95"]
    n = len(f)
    line = f"  {o:8s} lr0={best:g}  n={n}/5  mean={f.mean():.4f}"
    diff = (f - ref.reindex(f.index)).dropna()
    if len(diff) >= 2:
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        lb = diff.mean() - stats.t.ppf(0.95, len(diff) - 1) * se
        line += f"  gain={diff.mean():+.4f}  lower bound={lb:+.4f}  {'PASS' if lb > 0 else 'fail'}"
    print(line)
