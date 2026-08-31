import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from src.data.splits import list_eurosat_files, build_or_load_split, CLASS_NAMES


def _make_fake_dataset(tmp_path: Path, files_per_class: int = 20):
    tif_root = tmp_path / "tif"
    for class_name in CLASS_NAMES:
        class_dir = tif_root / class_name
        class_dir.mkdir(parents=True)
        for i in range(files_per_class):
            (class_dir / f"{class_name}_{i}.tif").touch()
    return tif_root


def test_list_eurosat_files_finds_all_classes_and_files():
    with tempfile.TemporaryDirectory() as tmp:
        tif_root = _make_fake_dataset(Path(tmp), files_per_class=10)
        files, labels = list_eurosat_files(tif_root)
        assert len(files) == 10 * len(CLASS_NAMES)
        assert len(labels) == len(files)
        assert set(labels) == set(range(len(CLASS_NAMES)))


def test_list_eurosat_files_raises_on_missing_class_folder():
    with tempfile.TemporaryDirectory() as tmp:
        tif_root = Path(tmp) / "tif"
        tif_root.mkdir()
        (tif_root / CLASS_NAMES[0]).mkdir()
        (tif_root / CLASS_NAMES[0] / "x.tif").touch()
        # deliberately don't create the other 9 class folders
        with pytest.raises(FileNotFoundError):
            list_eurosat_files(tif_root)


def test_list_eurosat_files_raises_on_empty_class_folder():
    with tempfile.TemporaryDirectory() as tmp:
        tif_root = _make_fake_dataset(Path(tmp), files_per_class=5)
        # empty out one class folder
        empty_class_dir = tif_root / CLASS_NAMES[0]
        for f in empty_class_dir.glob("*.tif"):
            f.unlink()
        with pytest.raises(FileNotFoundError):
            list_eurosat_files(tif_root)


def test_split_is_stratified_and_covers_all_samples():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tif_root = _make_fake_dataset(tmp_path, files_per_class=20)
        split_path = tmp_path / "split.json"

        split = build_or_load_split(tif_root, split_path, val_frac=0.2, test_frac=0.2, seed=0)

        n_total = 20 * len(CLASS_NAMES)
        n_train = len(split["train"]["files"])
        n_val = len(split["val"]["files"])
        n_test = len(split["test"]["files"])
        assert n_train + n_val + n_test == n_total

        # No file should appear in more than one split
        all_files = split["train"]["files"] + split["val"]["files"] + split["test"]["files"]
        assert len(all_files) == len(set(all_files))

        # Stratification: every class should appear in every split
        for subset_name in ("train", "val", "test"):
            labels_in_subset = set(split[subset_name]["labels"])
            assert labels_in_subset == set(range(len(CLASS_NAMES)))


def test_split_is_saved_and_reused_identically():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tif_root = _make_fake_dataset(tmp_path, files_per_class=15)
        split_path = tmp_path / "split.json"

        split_a = build_or_load_split(tif_root, split_path, seed=0)
        assert split_path.exists()

        # Second call must load the saved file, not rebuild with a new
        # random split -- this is what makes fold reuse across ablations valid.
        split_b = build_or_load_split(tif_root, split_path, seed=999)  # different seed, should be ignored
        assert split_a == split_b


# --- k-fold cross-validation splits (BLUEPRINT Section 11) ---

def _fake_tif_root(tmp_path):
    """Minimal on-disk layout list_eurosat_files() accepts: 10 class dirs with .tif files."""
    from src.data.splits import CLASS_NAMES
    root = tmp_path / "tif"
    for c in CLASS_NAMES:
        d = root / c
        d.mkdir(parents=True)
        for i in range(10):
            (d / f"{c}_{i}.tif").write_bytes(b"")
    return root


def test_kfolds_are_disjoint_and_cover_everything(tmp_path):
    from src.data.splits import build_or_load_kfolds
    folds = build_or_load_kfolds(_fake_tif_root(tmp_path), tmp_path / "folds.json", k=5)
    n = len(folds["files"])
    seen_tests = []
    for f in folds["folds"]:
        parts = [set(f["train"]), set(f["val"]), set(f["test"])]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert len(parts[0] | parts[1] | parts[2]) == n
        seen_tests.append(set(f["test"]))
    # every sample is tested exactly once across the k folds
    assert sum(len(t) for t in seen_tests) == n
    assert set().union(*seen_tests) == set(range(n))


def test_kfolds_are_stable_across_calls(tmp_path):
    """Paired significance tests are only valid if every variant sees identical folds."""
    from src.data.splits import build_or_load_kfolds
    root, path = _fake_tif_root(tmp_path), tmp_path / "folds.json"
    first = build_or_load_kfolds(root, path, k=5)
    second = build_or_load_kfolds(root, path, k=5)   # must load from disk, not rebuild
    assert first["folds"] == second["folds"]


def test_kfolds_are_stratified(tmp_path):
    from src.data.splits import build_or_load_kfolds
    folds = build_or_load_kfolds(_fake_tif_root(tmp_path), tmp_path / "folds.json", k=5)
    labels = folds["labels"]
    for f in folds["folds"]:
        counts = {}
        for i in f["test"]:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        assert len(counts) == 10                    # every class present in every test fold
        assert max(counts.values()) - min(counts.values()) <= 1


def test_materialize_fold_matches_indices(tmp_path):
    from src.data.splits import build_or_load_kfolds, materialize_fold
    folds = build_or_load_kfolds(_fake_tif_root(tmp_path), tmp_path / "folds.json", k=5)
    fold = materialize_fold(folds, 2)
    idx = folds["folds"][2]["test"]
    assert fold["test"]["files"] == [folds["files"][i] for i in idx]
    assert fold["test"]["labels"] == [folds["labels"][i] for i in idx]


# --- low-data ablation subsampling ---

def test_subsample_keeps_requested_fraction_and_stratification():
    from src.data.splits import subsample_split
    split = {"files": [f"f{i}.tif" for i in range(100)],
             "labels": [i % 5 for i in range(100)]}
    out = subsample_split(split, 0.2, seed=0)
    assert len(out["files"]) == 20
    counts = {}
    for l in out["labels"]:
        counts[l] = counts.get(l, 0) + 1
    assert set(counts) == {0, 1, 2, 3, 4}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_subsample_files_and_labels_stay_aligned():
    from src.data.splits import subsample_split
    split = {"files": [f"f{i}.tif" for i in range(100)],
             "labels": [i % 5 for i in range(100)]}
    out = subsample_split(split, 0.3, seed=1)
    for f, l in zip(out["files"], out["labels"]):
        assert int(f[1:-4]) % 5 == l


def test_subsample_full_fraction_is_identity():
    from src.data.splits import subsample_split
    split = {"files": ["a.tif", "b.tif"], "labels": [0, 1]}
    assert subsample_split(split, 1.0) is split


def test_subsample_rejects_invalid_fraction():
    import pytest
    from src.data.splits import subsample_split
    split = {"files": ["a.tif"], "labels": [0]}
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            subsample_split(split, bad)


def test_subsample_is_deterministic_for_a_seed():
    from src.data.splits import subsample_split
    split = {"files": [f"f{i}.tif" for i in range(100)],
             "labels": [i % 5 for i in range(100)]}
    assert subsample_split(split, 0.2, seed=7) == subsample_split(split, 0.2, seed=7)
    assert subsample_split(split, 0.2, seed=7) != subsample_split(split, 0.2, seed=8)
