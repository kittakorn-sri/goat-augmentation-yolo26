# Goat CAE / Orf lesion detection with YOLO26n: augmentation and optimizer selection pipeline

Code for the experiments of Manuscript 16540 (detection of caprine arthritis encephalitis (CAE) and Orf lesions in goat images).
The pipeline selects an augmentation set and an optimizer on a validation split, then evaluates the frozen configuration on a
held-out test split with new seeds.

The dataset is **not** included. See "Data" below.

## Pipeline

| Script | Phase | Splits used | What it does |
|---|---|---|---|
| `0a_prepare_dataset.py` | 0a | all | Builds a group-aware train/valid/test split (perceptual hash + `.rf.` augmentation siblings; `SPLIT_SEED = 42`). Writes `split_manifest.csv`. |
| `0b_verify_split.py` | 0b | all | Checks that no near-duplicate group crosses a partition (needs `similarity.py`). |
| `1_train_augmentation.py` | 1 | train, valid | Each of 15 augmentation techniques vs Baseline, 5 selection seeds. |
| `2_test_combinations_forward.py` | 2 | train, valid | Greedy forward selection of an augmentation set (mean seed-paired gain on valid mAP50-95). |
| `3_train_optimizer.py` | 3 | train, valid | Six optimizers + `auto`, learning-rate grid; an optimizer replaces `auto` only if the one-sided 95 % lower bound of its paired gain is > 0. |
| `4_stepwise_evaluation.py` | 4 | train, valid, **test** | Stepwise ladder (Baseline, + best single augmentation, + selected set, + optimizer) and a `cls_pw 0.0` sensitivity run, 5 evaluation seeds. |
| `5_compare_models.py` | 5 | train, valid, **test** | YOLO26s and YOLO11n vs YOLO26n with the frozen configuration and with default settings; size and speed benchmark. |
| `6_bootstrap_test.py` | 6 | **test** | Cluster bootstrap over test images (1000 resamples, same resamples for all models). |

Shared code: `config.py` (all settings), `common.py` (training, evaluation, benchmark helpers), `similarity.py`.
Helpers: `show_phase3_progress.py` (progress table while Phase 3 runs), `gflops_check.py` (parameters and GFLOPs).

### Seeds

- split: 42
- selection (phases 1-3): 7, 11, 13, 17, 19
- evaluation (phases 4-6): 101, 202, 303, 404, 505 (disjoint from the selection seeds on purpose)
- bootstrap resampling: 777

The test split is touched only by phases 4-6.

### Fixed training settings (see `config.py`)

Model `yolo26n.pt`, image size 512, batch 16, up to 600 epochs with patience 100, `cls_pw = 1.0`
(**not** the Ultralytics default, which is 0.0; the `cls_pw 0.0` run in phase 4 is a sensitivity check).

## Usage

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=...        # only if the dataset is downloaded through the API; never put the key in a file

python 0a_prepare_dataset.py --source /path/to/downloaded_dataset    # or without --source to use the API
python 0b_verify_split.py
python 1_train_augmentation.py
python 2_test_combinations_forward.py
python 3_train_optimizer.py
python 4_stepwise_evaluation.py
BENCH_THREADS=8 python 5_compare_models.py
python 6_bootstrap_test.py
```

Notes
- **Do not rerun `0a_prepare_dataset.py` after the split has been used**: it regenerates the split. Keep `single_split_data_v2/split_manifest.csv`
  (600 rows) under version control so the split used in the paper is documented.
- Each phase is resumable: finished runs are skipped. `EPOCHS=2 python <script>` is a quick smoke test.
- Phases 4 and 5 report Ultralytics `val()` metrics on the test split (the primary numbers).
- Phase 6 records the per-image true-positive flags from the same Ultralytics validator and resamples clusters of test images.
  It stops if the full-test-set mAP50-95 it computes differs from the value reported in phase 4/5 by more than `--tol` (default 0.02).
- Latency in phase 5 is measured at batch size 1 with fixed CPU threads (`BENCH_THREADS`, default 8) on one machine;
  it does not establish performance on edge devices.

### Output files

`results_*.csv` (per-run and summary tables), `phase3_selected.json`, `final_plan.json`, `results_final_test*.csv`,
`results_compare_*.csv`, `bootstrap_run_level.csv`, `bootstrap_ci.csv`, `bootstrap_paired_differences.csv`, `bootstrap_meta.json`.

## Data

The images come from the Roboflow Universe project `goat-qxgzi/goat-twi81` (version 4). The images were collected from several
sources and have different licences; the project page states CC BY 4.0 for the dataset compilation, which does not by itself
license the underlying third-party images. An image-level licence and provenance table is provided with the paper
(`Image_Licensing_Metadata_Updated4.xlsx`). Check redistribution terms before republishing the images.

## Reproducibility and limitations

- One split (`SPLIT_SEED = 42`) and a small test set (86 images, 56 clusters): confidence intervals are wide.
- Intervals over seeds (training variability) and over test clusters (phase 6) answer different questions; the paper reports both.
- Record your own versions (`pip freeze`) and hardware; results can differ slightly across Ultralytics and PyTorch versions.

## Licence and citation

Add a licence file of your choice and the citation of the paper before publishing.
