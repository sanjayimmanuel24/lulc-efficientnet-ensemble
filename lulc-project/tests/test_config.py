"""
Tests for configs/config.yaml loading.

Context: this file was inert for the whole project -- nothing imported yaml, and
its values had drifted from what ran (batch_size 64 vs the actual 32), while
README.md advertised it as the reproducibility unit. `model.attention` likewise
offered an "eca | none" switch the model ignored. These tests pin the contract.
"""
import json
from pathlib import Path

import pytest

from src.config import (
    DEFAULT_CONFIG_PATH, config_defaults, dump_resolved_config, load_config, unmapped_keys,
)


def test_shipped_config_parses():
    cfg = load_config()
    assert cfg, "configs/config.yaml should exist and be non-empty"
    assert "train" in cfg and "model" in cfg


def test_missing_config_returns_empty_not_crash(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == {}


def test_defaults_map_to_argparse_dest_names():
    d = config_defaults(load_config())
    assert d["batch_size"] == 32          # corrected from the drifted 64
    assert d["epochs_per_stage"] == [5, 5, 10]
    assert d["image_sizes"] == [128, 160, 224]
    assert d["attention"] == "eca"
    assert d["folds"] == 5


def test_shipped_config_batch_size_matches_hardware_note():
    """224px peaks at 3.07 GB at batch 32 on a 4 GB card; 64 would not fit."""
    assert load_config()["train"]["batch_size"] <= 32


def test_partial_config_yields_only_present_keys():
    d = config_defaults({"train": {"lr": 0.001}})
    assert d == {"lr": 0.001}


def test_non_dict_sections_are_ignored():
    assert config_defaults({"seed": 42}) == {}
    assert unmapped_keys({"seed": 42}) == []


def test_unmapped_keys_are_reported_not_silently_dropped():
    """A setting that looks live but is ignored is how `model.attention` went stale."""
    reported = unmapped_keys(load_config())
    assert "model.backbone" in reported          # descriptive, no flag consumes it
    assert "model.attention" not in reported     # now genuinely wired


def test_dump_resolved_config_records_effective_values(tmp_path):
    class Args:
        pass
    a = Args()
    a.batch_size = 16
    a.tag = "demo"
    path = dump_resolved_config(a, tmp_path, {"device": "cuda"})
    saved = json.loads(Path(path).read_text())
    assert saved["batch_size"] == 16 and saved["tag"] == "demo"
    assert saved["device"] == "cuda"
