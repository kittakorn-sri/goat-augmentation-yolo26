# Goat CAE / Orf lesion detection with YOLO26n

Augmentation and optimizer selection on a validation split, then evaluation of the frozen configuration on a held-out test split. Images are **not** included.

## Pipeline

| Script | Phase | Purpose |
|---|---|---|
| `0a_prepare_dataset.py` | 0 | Group-aware train/valid/test split |
| `0b_verify_split.py` | 0 | Checks that no near-duplicate group crosses a split |
| `1_train_augmentation.py` | 1 | 15 augmentations vs baseline (validation) |
| `2_test_combinations_forward.py` | 2 | Greedy forward selection of an augmentation set (validation) |
| `3_train_optimizer.py` | 3 | Six optimizers × learning-rate grid vs `auto` (validation) |
| `4_stepwise_evaluation.py` | 4 | Stepwise ladder on the **test** set, 5 evaluation seeds |
| `5_compare_models.py` | 5 | YOLO26s and YOLO11n vs YOLO26n; size and speed (**test**) |
| `6_bootstrap_test.py` | 6 | Cluster bootstrap over test images (**test**) |

Settings are in `config.py`. `dataset_stats.py` gives image and instance counts per split.
Seeds: split 42; selection 7, 11, 13, 17, 19; evaluation 101, 202, 303, 404, 505; bootstrap 777.

## Usage

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=...   # only if the dataset is downloaded through the API

python 0a_prepare_dataset.py --source /path/to/dataset
python 0b_verify_split.py
python 1_train_augmentation.py
python 2_test_combinations_forward.py
python 3_train_optimizer.py
python 4_stepwise_evaluation.py
python 5_compare_models.py
python 6_bootstrap_test.py
```

Do not rerun `0a_prepare_dataset.py` after the split has been used; it regenerates the split (the one used in the paper is in `split_manifest.csv`). Phases are resumable.
Tested with Python 3.13, Ultralytics 8.4.180, PyTorch 2.11.0, Albumentations 2.0.8.

## Data and licence

Images come from the Roboflow Universe project `goat-qxgzi/goat-twi81` and have different source licences; check the terms of each image before republishing it.
Code: MIT License (`LICENSE`), which does not cover the images.
