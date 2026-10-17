# ==================== common.py ====================
# Helpers shared by phases 1-7 (previously duplicated across files)

import copy
import gc
import glob
import inspect
import os
import platform
import time
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from scipy import stats
from ultralytics import YOLO, RTDETR

from config import (
    augmentation_configs, ZERO_AUG, calc_f1, CLOSE_MOSAIC_EPOCHS,
    DATA_YAML, DATA_TEST_YAML, SPLIT_DIR, MODEL, EPOCHS, PATIENCE, IMGSZ, BATCH, CLS_PW,
)


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------
def build_aug_config(methods):
    """Merge several augmentation methods. Albumentations is ALWAYS a non-empty list:
    if no method needs it, a no-op transform (p=0) is used so Ultralytics cannot
    substitute its default Albumentations set."""
    builtin = dict(ZERO_AUG)
    albu = []
    for m in methods:
        cfg = augmentation_configs[m]          # KeyError on typo is intentional
        for k, v in cfg["builtin"].items():
            if v != 0:
                builtin[k] = v
        albu.extend(copy.deepcopy(cfg["albu"]))
    if not albu:
        albu = [A.Blur(p=0.0)]
    return {"builtin": builtin, "albu": albu}


# ------------------------------------------------------------------
# Locating runs (Ultralytics places relative projects under runs/detect/)
# ------------------------------------------------------------------
def find_run_dir(project, name):
    for root in (Path("runs") / "detect" / project, Path(project)):
        d = root / name
        if d.exists():
            return d
    return None


# ------------------------------------------------------------------
# Model families (RT-DETR needs its own class)
# ------------------------------------------------------------------
def family_of(model_name):
    return "rtdetr" if "rtdetr" in os.path.basename(str(model_name)).lower() else "yolo"


def load_model(weights, family_hint=None):
    """family_hint: the model NAME (e.g. 'rtdetr-l.pt'); needed when `weights` is a checkpoint path like last.pt."""
    cls = RTDETR if family_of(family_hint or weights) == "rtdetr" else YOLO
    return cls(str(weights))


def train_run(project, name, methods, optimizer, lr0, seed, cls_pw=CLS_PW, data=DATA_YAML,
              model_name=MODEL):
    """Train one configuration; returns the run directory.
    - If the run finished earlier (TRAIN_DONE marker) it is not retrained.
    - If it was interrupted (last.pt exists, no marker) it is resumed.
    - `lr0` is ignored for optimizer='auto' (Ultralytics chooses it)."""
    existing = find_run_dir(project, name)
    if existing is not None and (existing / "TRAIN_DONE").exists():
        return existing

    last = existing / "weights" / "last.pt" if existing is not None else None
    model = None
    if last is not None and last.exists():
        try:
            model = load_model(last, family_hint=model_name)
        except Exception as e:           # killed while last.pt was being written -> truncated checkpoint
            bad = existing.with_name(existing.name + "_corrupt")
            k = 1
            while bad.exists():
                k += 1
                bad = existing.with_name(f"{existing.name}_corrupt{k}")
            print(f"[{name}] last.pt is not loadable ({type(e).__name__}); moving the run to {bad.name} and retraining from scratch")
            existing.rename(bad)
        else:
            model.train(resume=True)
    if model is None:
        aug = build_aug_config(methods)
        kwargs = dict(
            data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            optimizer=optimizer, patience=PATIENCE,
            seed=seed, deterministic=True,
            close_mosaic=CLOSE_MOSAIC_EPOCHS if any(aug["builtin"].get(k, 0) > 0 for k in ("mosaic", "mixup", "cutmix")) else 0,
            project=project, name=name, exist_ok=True,
            cls_pw=cls_pw, augmentations=aug["albu"],
            **aug["builtin"],
        )
        if optimizer != "auto" and lr0 is not None:
            kwargs["lr0"] = lr0
        model = load_model(model_name)
        model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    (save_dir / "TRAIN_DONE").write_text("done")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir


# ------------------------------------------------------------------
# Validation-split statistics taken from the run's own results.csv
# (row = epoch with best fitness = 0.1*mAP50 + 0.9*mAP50-95, i.e. the epoch saved as best.pt)
# ------------------------------------------------------------------
def read_run_stats(save_dir):
    df = pd.read_csv(Path(save_dir) / "results.csv")
    df.columns = df.columns.str.strip()
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    i = fitness.idxmax()
    p = float(df.loc[i, "metrics/precision(B)"])
    r = float(df.loc[i, "metrics/recall(B)"])
    return {
        "val_mAP50_95": float(df.loc[i, "metrics/mAP50-95(B)"]),
        "val_mAP50": float(df.loc[i, "metrics/mAP50(B)"]),
        "val_Precision": p,
        "val_Recall": r,
        "val_F1": float(calc_f1(p, r)),
        "epochs_run": int(len(df)),
        "best_epoch": int(df.loc[i, "epoch"]),
        "lr_peak": float(df["lr/pg0"].max()),   # ~ lr0 actually used (0.00167 => auto chose AdamW for nc=2)
        "save_dir": str(save_dir),
    }


# ------------------------------------------------------------------
# CSV bookkeeping (one row per key; re-running replaces the row)
# ------------------------------------------------------------------
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def has_row(path, key_col, key):
    df = load_csv(path)
    return (not df.empty) and (key in set(df[key_col].astype(str)))


def get_row(path, key_col, key):
    df = load_csv(path)
    if df.empty or key_col not in df.columns:
        return None
    hit = df[df[key_col].astype(str) == str(key)]
    return None if hit.empty else hit.iloc[0].to_dict()


def append_row(path, key_col, row):
    df = load_csv(path)
    if not df.empty:
        df = df[df[key_col].astype(str) != str(row[key_col])]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


# ------------------------------------------------------------------
# Test-split evaluation (used by phases 4-6 only)
# ------------------------------------------------------------------
def evaluate_on_test(weights, run_name, eval_dir, family_hint=None, data=DATA_TEST_YAML):
    """Single evaluation on the held-out test split (conf=0.001, IoU=0.7)."""
    model = load_model(weights, family_hint)
    m = model.val(data=data, split="val", conf=0.001, iou=0.7,
                  project=eval_dir, name=run_name, exist_ok=True, plots=True)
    names = m.names
    idx = list(m.box.ap_class_index)
    per_class = pd.DataFrame({
        "run": run_name,
        "class": [names[i] for i in idx],
        "AP50": m.box.ap50, "AP50_95": m.box.ap,
        "Precision": m.box.p, "Recall": m.box.r, "F1": m.box.f1,
    })
    labels = [names[i] for i in sorted(names)] + ["Background"]
    cm = pd.DataFrame(m.confusion_matrix.matrix.T, index=labels, columns=labels)   # rows = true, cols = predicted
    out = os.path.join(eval_dir, run_name)
    os.makedirs(out, exist_ok=True)
    cm.to_csv(os.path.join(out, "confusion_matrix_rows_true_cols_pred.csv"))
    p, r = float(m.box.mp), float(m.box.mr)
    overall = {"test_mAP50_95": float(m.box.map), "test_mAP50": float(m.box.map50),
               "test_Precision": p, "test_Recall": r, "test_F1_of_means": float(calc_f1(p, r))}
    return overall, per_class


# ------------------------------------------------------------------
# Size and speed (parameters, GFLOPs, file size, latency at batch size 1)
# ------------------------------------------------------------------
def _latency(weights, family_hint, device, n_warmup, n_runs, conf, imgs):
    model = load_model(weights, family_hint)
    for i in range(n_warmup):
        model.predict(imgs[i % len(imgs)], imgsz=IMGSZ, conf=conf, device=device, verbose=False)
    infer, total, wall = [], [], []
    for i in range(n_runs):
        t0 = time.perf_counter()
        r = model.predict(imgs[i % len(imgs)], imgsz=IMGSZ, conf=conf, device=device, verbose=False)[0]
        wall.append((time.perf_counter() - t0) * 1000)
        sp = r.speed
        infer.append(sp["inference"])
        total.append(sp["preprocess"] + sp["inference"] + sp["postprocess"])
    return {"infer_ms": float(np.mean(infer)), "pre_inf_post_ms": float(np.mean(total)),
            "wall_ms_mean": float(np.mean(wall)), "wall_ms_p95": float(np.percentile(wall, 95))}


def benchmark_model(weights, family_hint=None, n_warmup=10, n_runs=100, cpu_runs=30, conf=0.25):
    """Parameters, GFLOPs (at IMGSZ), weight-file size, and batch-1 latency on real test images.
    This is a measurement on THIS machine only; it does not establish edge-device or concurrent-user performance."""
    out = {"weights_MB": round(os.path.getsize(weights) / 1e6, 2)}
    # Fixed CPU thread count so that all models are timed identically (the default can differ between runs on a shared server).
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", 8)))
    model = load_model(weights, family_hint)
    out["params_M"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3)
    out["GFLOPs"] = None
    try:
        info = model.info(verbose=False, imgsz=IMGSZ)         # (layers, params, gradients, GFLOPs) in recent versions
        if info is not None and len(info) >= 4:
            out["GFLOPs"] = float(info[3])
    except Exception:
        pass
    if out["GFLOPs"] is None or out["GFLOPs"] != out["GFLOPs"]:   # None / NaN: ask Ultralytics' FLOPs counter directly
        try:
            from ultralytics.utils.torch_utils import get_flops
            out["GFLOPs"] = float(get_flops(model.model, IMGSZ))
        except Exception:
            out["GFLOPs"] = None
    del model

    imgs = sorted(glob.glob(os.path.join(SPLIT_DIR, "test", "images", "*")))[:20]
    out["latency_conf"] = conf
    out["host"] = platform.platform()
    out["torch_threads"] = torch.get_num_threads()
    if not imgs:
        return out
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out.update({f"gpu_{k}": v for k, v in _latency(weights, family_hint, 0, n_warmup, n_runs, conf, imgs).items()})
    out.update({f"cpu_{k}": v for k, v in _latency(weights, family_hint, "cpu", min(n_warmup, 5), cpu_runs, conf, imgs).items()})
    return out


# ------------------------------------------------------------------
# Evidence for the reviewers: which augmentations the installed Ultralytics really builds
# ------------------------------------------------------------------
def dump_transform_source(family, path):
    """Writes the source of the transform builders (and the Albumentations wrapper) of the installed version."""
    parts = []
    try:
        import ultralytics.data.augment as aug
        import ultralytics.data.dataset as ds
        parts += [("ultralytics.data.augment.v8_transforms", aug.v8_transforms),
                  ("ultralytics.data.augment.Albumentations", aug.Albumentations),
                  ("ultralytics.data.dataset.YOLODataset.build_transforms", ds.YOLODataset.build_transforms)]
        if family == "rtdetr":
            from ultralytics.models.rtdetr.val import RTDETRDataset
            parts.append(("ultralytics.models.rtdetr.val.RTDETRDataset.build_transforms", RTDETRDataset.build_transforms))
    except Exception as e:                                    # keep going; this is evidence-gathering only
        parts.append((f"could not import ({type(e).__name__})", None))
    chunks = []
    for title, obj in parts:
        try:
            body = inspect.getsource(obj) if obj is not None else str(title)
        except Exception as e:
            body = f"<source unavailable: {e}>"
        chunks.append(f"# ===== {title} =====\n{body}\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(chunks))
    return path


# ------------------------------------------------------------------
# Paired comparison across seeds (a, b: pandas Series indexed by seed)
# ------------------------------------------------------------------
def paired_summary(a, b, confidence=0.95):
    """Seed-paired difference a - b. `lb_one_sided` is the one-sided lower confidence bound at `confidence`
    (accept a over b when it is > 0); ci95_low/high is the usual two-sided 95% interval for reporting."""
    d = (pd.Series(a) - pd.Series(b)).dropna()
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 paired seeds")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else (np.inf if mean > 0 else (-np.inf if mean < 0 else 0.0))
    t1 = stats.t.ppf(confidence, n - 1)
    t2 = stats.t.ppf(0.975, n - 1)
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "t": float(t),
            "p_one_sided": float(stats.t.sf(t, n - 1)),
            "lb_one_sided": mean - t1 * se,
            "ci95_low": mean - t2 * se, "ci95_high": mean + t2 * se,
            "passes": bool(mean - t1 * se > 0)}
