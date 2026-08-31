"""
Build (once) and reuse (always after) a fixed, stratified train/val/test
split over the extracted EuroSAT-MS GeoTIFF files.

This is saved to disk (as JSON: file paths + integer labels per split) the
first time it's built, and loaded verbatim on every subsequent call. That
matters beyond convenience: docs/BLUEPRINT.md Section 10's ablations and
Section 11's paired statistical tests (paired t-test, Wilcoxon, McNemar)
all assume every model variant is compared on the *same* held-out samples.
Rebuilding a fresh random split per experiment would silently invalidate
every one of those comparisons.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# Fixed, alphabetical class ordering -> integer label mapping. This must
# never change once a checkpoint/split has been created against it.
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def list_eurosat_files(tif_root: Path) -> tuple[list[str], list[int]]:
    file_paths: list[str] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = tif_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Check that tif_root points at the folder containing all 10 class subfolders."
            )
        files = sorted(class_dir.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No .tif files found in {class_dir} -- extraction may be incomplete.")
        file_paths.extend(str(f) for f in files)
        labels.extend([label] * len(files))
    return file_paths, labels


def build_or_load_split(tif_root: Path, split_path: Path, val_frac: float = 0.15,
                         test_frac: float = 0.15, seed: int = 42) -> dict:
    if split_path.exists():
        with open(split_path) as f:
            return json.load(f)

    file_paths, labels_list = list_eurosat_files(tif_root)
    labels = np.array(labels_list)
    all_idx = np.arange(len(labels))

    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=test_frac, stratify=labels, random_state=seed,
    )
    val_size_within_train_val = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_within_train_val,
        stratify=labels[train_val_idx], random_state=seed,
    )

    def _subset(idx):
        return {"files": [file_paths[i] for i in idx], "labels": [int(labels[i]) for i in idx]}

    split = {
        "class_names": CLASS_NAMES,
        "seed": seed,
        "train": _subset(train_idx),
        "val": _subset(val_idx),
        "test": _subset(test_idx),
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f)
    return split


def build_or_load_kfolds(tif_root: Path, folds_path: Path, k: int = 5,
                          val_frac_of_trainval: float = 0.1765, seed: int = 42) -> dict:
    """
    Build (once) and reuse (always after) k stratified folds over the same file list.

    Stored as index lists into one shared file/label array rather than k copies of
    the paths -- five duplicated path lists would be ~27 MB of mostly repeated text.

    Every model variant in the ablation study MUST consume these same folds:
    docs/BLUEPRINT.md Section 11's paired tests (paired t, Wilcoxon, McNemar)
    compare variants sample-by-sample, which is only valid if the samples are
    identical across variants. Rebuilding folds per variant silently invalidates
    every one of those tests.

    `val_frac_of_trainval` defaults to 0.1765 so that val is ~14.1% of the whole
    dataset when k=5 (each test fold is 20%), keeping the train/val/test balance
    close to the single-split 70/15/15 used for the first run.
    """
    if folds_path.exists():
        with open(folds_path) as f:
            return json.load(f)

    file_paths, labels_list = list_eurosat_files(tif_root)
    labels = np.array(labels_list)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac_of_trainval,
            stratify=labels[trainval_idx], random_state=seed + fold_idx,
        )
        folds.append({
            "train": [int(i) for i in train_idx],
            "val": [int(i) for i in val_idx],
            "test": [int(i) for i in test_idx],
        })

    payload = {
        "class_names": CLASS_NAMES,
        "seed": seed,
        "k": k,
        "files": file_paths,
        "labels": [int(x) for x in labels_list],
        "folds": folds,
    }
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_path, "w") as f:
        json.dump(payload, f)
    return payload


def materialize_fold(folds: dict, fold_idx: int) -> dict:
    """Turn one fold's index lists into the {'train','val','test'} dict the loaders expect."""
    files, labels = folds["files"], folds["labels"]

    def _subset(name):
        idx = folds["folds"][fold_idx][name]
        return {"files": [files[i] for i in idx], "labels": [labels[i] for i in idx]}

    return {"train": _subset("train"), "val": _subset("val"), "test": _subset("test")}


def subsample_split(split_data: dict, frac: float, seed: int = 42) -> dict:
    """
    Stratified subsample of one split (intended for the TRAINING split only).

    Used by the low-data ablation: C1's confidence-weighted fusion showed no accuracy
    gain over plain averaging on full EuroSAT, with mean fusion weight 0.4995 and the
    branches agreeing on ~98% of samples -- i.e. no headroom for any adaptive rule.
    Starving the model of training data is the direct way to create disagreement and
    test whether C1's failure is saturation-specific.

    Val/test are deliberately left at full size by the caller: full test keeps results
    comparable with the existing sweeps, and full val keeps model selection and
    temperature fitting stable when training data is scarce.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if frac == 1.0:
        return split_data

    labels = np.array(split_data["labels"])
    keep, _ = train_test_split(
        np.arange(len(labels)), train_size=frac, stratify=labels, random_state=seed,
    )
    keep = np.sort(keep)
    return {"files": [split_data["files"][i] for i in keep],
            "labels": [int(labels[i]) for i in keep]}


def split_in_two(split_data: dict, frac: float, seed: int = 42) -> tuple[dict, dict]:
    """
    Stratified two-way split of one split. Returns (first, second) where `second`
    holds `frac` of the samples.

    Used to carve a dedicated CALIBRATION set out of validation. Without it the
    validation split does triple duty -- early stopping, checkpoint selection and
    temperature fitting -- so the temperature is fitted on data already used to
    choose the model, biasing reported ECE/Brier optimistically. That matters here
    because C4 is a contribution and C1's only surviving claim is calibration.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    labels = np.array(split_data["labels"])
    first_idx, second_idx = train_test_split(
        np.arange(len(labels)), test_size=frac, stratify=labels, random_state=seed,
    )

    def _take(idx):
        idx = np.sort(idx)
        return {"files": [split_data["files"][i] for i in idx],
                "labels": [int(labels[i]) for i in idx]}

    return _take(first_idx), _take(second_idx)
