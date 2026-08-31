"""
Loader for configs/config.yaml.

Until now this file was decorative: nothing imported yaml, and its values had
drifted from what the scripts actually ran (it said batch_size 64; every real
run used 32). That matters more than it sounds -- README.md tells readers that
"the exact configs/config.yaml used for every run" is the reproducibility unit,
so a config that nothing reads is a reproducibility claim that cannot be honoured.

Contract:
  * config.yaml supplies DEFAULTS,
  * an explicit command-line flag always overrides them,
  * and the fully-resolved values are dumped next to each run's results
    (`resolved_config.json`) so what actually ran is recoverable from the
    artefacts alone, not inferred from a file that may have changed since.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# config.yaml key path -> argparse dest name. Only keys that a runner actually
# accepts appear here; anything else in the YAML is documentation, not settings,
# and is reported by unmapped_keys() rather than silently ignored.
_KEY_MAP = {
    ("data", "image_sizes"): "image_sizes",
    ("data", "spectral_indices"): "spectral_indices",
    ("data", "spectral_branch_mode"): "spectral_branch_mode",
    ("model", "attention"): "attention",
    ("train", "epochs_per_stage"): "epochs_per_stage",
    ("train", "batch_size"): "batch_size",
    ("train", "lr"): "lr",
    ("train", "num_workers"): "num_workers",
    ("evaluation", "k_folds"): "folds",
}


def load_config(path: Path | str | None = None) -> dict:
    """Read config.yaml. Returns {} if the file is absent, so scripts still run."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def config_defaults(cfg: dict) -> dict:
    """Flatten config.yaml into {argparse_dest: value} for parser.set_defaults()."""
    out = {}
    for (section, key), dest in _KEY_MAP.items():
        if isinstance(cfg.get(section), dict) and key in cfg[section]:
            out[dest] = cfg[section][key]
    return out


def unmapped_keys(cfg: dict) -> list[str]:
    """
    Keys present in config.yaml that no runner consumes.

    Surfaced deliberately: a setting that looks live but is ignored is exactly
    how `model.attention` sat in this file for months while the model hardcoded
    ECA. Callers print these rather than failing, since some keys (dataset names,
    bootstrap_samples) are genuinely descriptive.
    """
    mapped = {(s, k) for s, k in _KEY_MAP}
    out = []
    for section, body in cfg.items():
        if not isinstance(body, dict):
            continue
        for key in body:
            if (section, key) not in mapped:
                out.append(f"{section}.{key}")
    return sorted(out)


def dump_resolved_config(args, out_dir: Path, extra: dict | None = None) -> Path:
    """
    Write the fully-resolved settings for a run next to its results.

    This, not config.yaml, is the artefact to cite in the paper: config.yaml is
    a default source that can drift, while this records what the run actually used
    after command-line overrides.
    """
    payload = {k: v for k, v in vars(args).items()}
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
