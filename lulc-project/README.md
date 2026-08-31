# EfficientNet-B0 Dual-Branch Ensemble for LULC Classification

Companion code for `docs/BLUEPRINT.md`, the full research design document
(problem statement, related-work gap, architecture, math, experiments,
ablations, statistics, and an IEEE-reviewer-style critique).

## Current status

- Research blueprint (`docs/BLUEPRINT.md`) complete, including an honest engagement
  with prior work (Helber et al. 2019's EuroSAT paper) and its implications for
  contribution C2 — see Sections 3, 4, 9, 15, 18.
- Full codebase implemented: data pipeline, model, training loop, calibration,
  evaluation, statistical tests, explainability.
- 48 tests passing (pure NumPy/SciPy/sklearn tests, verified repeatedly across
  environments) + 8 additional torch-dependent tests (model shapes, single-batch
  overfit sanity check, calibration wiring) that pass once `requirements.txt` is
  installed on a real machine.
- EuroSAT-MS (all 10 classes, ~27,000 GeoTIFFs) downloaded and verified.
- A transform-alignment bug (RGB and spectral branch inputs could have received
  independent, potentially-inconsistent transforms) was caught and fixed before
  any real training.
- `scripts/train_eurosat.py` (the real-data training runner) had a Windows
  `ModuleNotFoundError: No module named 'src'` fixed — the script now inserts
  the project root onto `sys.path` itself, so it works regardless of how it's
  invoked (double-click, `python scripts\train_eurosat.py`, an IDE run button, etc.).
- **Not yet done:** the first real training run on EuroSAT-MS (smoke test queued
  next), the full k=5 ablation/statistical-validation protocol, and the paper draft.

## Moving this project to a new machine

If you switch devices mid-project, copy the entire `lulc-project/` folder over
(including `data/eurosat/` if you don't want to re-download 2GB), then on the
new machine:

```bash
pip install -r requirements.txt
pytest tests/ -v          # confirm all 48 (+8 torch) tests still pass here too
```

Nothing in this repo assumes a specific machine — no absolute paths, no
machine-specific config — except that `data/eurosat/` and any
`checkpoints/`/`results/` you've generated only exist wherever you put them.
Those aren't tracked by git/this zip and need to be copied or regenerated
manually.

## What this is, honestly

The original prompt asked for ~30 possible novel components (attention
variants, distillation, contrastive pretraining, ArcFace, MixUp/CutMix/
Mosaic, domain adaptation, meta-learners, ...). Section 15 of
`docs/BLUEPRINT.md` ("Rejected Alternatives") goes through each one and
explains, individually, why it was kept or cut. The short version: a paper
with 15 loosely-related add-ons reads as engineering, not science, and is
an easy reviewer target ("which component actually did the work?"). This
repo implements **4 contributions that reinforce each other**:

1. Confidence-aware adaptive fusion (`src/models/fusion.py`, `fusion_torch.py`)
2. True multispectral fusion via NDVI (`src/data/spectral_indices.py`) -
   computed from real Sentinel-2 NIR/Red bands, not approximated from RGB
3. ECA attention, chosen over SE/CBAM specifically for near-zero added cost
   (`src/models/attention.py`)
4. Calibration-aware evaluation (`src/calibration/temperature_scaling.py`),
   which matters *because* contribution #1 uses confidence as a fusion
   signal — if confidence is miscalibrated, the fusion weights inherit that
   bias. This is what makes it four coherent pieces instead of four
   unrelated bullet points.

## What has actually been verified in this environment, and what hasn't

This repo was built in a CPU-only sandbox with no GPU and a tight disk
quota. Installing the standard PyPI `torch` wheel here would pull 2-3 GB of
bundled CUDA libraries you can't use anyway (no GPU) just to run a shape
check — a bad trade, not a shortcut worth taking. So the verification was
split honestly along the line that actually matters for a research paper:

- **The novel formulas** (confidence-weighted fusion, NDVI/NDWI/NDBI,
  temperature scaling + ECE/MCE/Brier, label smoothing, McNemar/Wilcoxon/
  paired-t/bootstrap statistics, all core classification metrics) are
  implemented in pure NumPy/SciPy/scikit-learn and **43 unit tests pass in
  this environment right now** (`pytest tests/ -v`, excluding
  `test_models_torch.py`). This is where a wrong formula would silently
  invalidate a paper's central claims, so it's what got tested first.
- **The model/training code** (`src/models/ensemble.py`, `attention.py`,
  `src/training/train.py`, `src/explainability/gradcam.py`) is standard
  PyTorch + timm, written to the same APIs used throughout the ecosystem.
  It has **not** been executed in this sandbox. `tests/test_models_torch.py`
  contains the tests for it, including the classic "overfit a single batch"
  sanity check every model should pass before a real training run, and an
  equivalence check against the NumPy fusion/loss reference implementations.
  These are skipped here (`pytest.importorskip`) and will run as soon as you
  install `requirements.txt` in a real (ideally GPU) environment:

  ```bash
  pip install -r requirements.txt
  pytest tests/ -v
  ```

Don't trust the model code just because it reads correctly — run that
command once before starting real training, and confirm all tests pass
(especially `test_overfit_single_batch_sanity_check`) before spending GPU
hours on the full EuroSAT run.

## Project structure

```
configs/config.yaml          single default config (see "Configuration" below)
src/data/                    spectral index math + EuroSAT-MS / synthetic datasets
src/models/                  attention, fusion (numpy + torch), dual-branch ensemble
src/losses/                  label smoothing (numpy reference + torch wrapper)
src/calibration/             temperature scaling, ECE/MCE/Brier
src/evaluation/               metrics, statistical significance tests
src/training/                 training loop (progressive resizing, AdamW, cosine LR)
src/explainability/           Grad-CAM
tests/                        one test file per module above
docs/BLUEPRINT.md             full research design document
```

## Running a real training run on EuroSAT-MS

Once you've extracted EuroSAT-MS (see `scripts/download_eurosat.py` and the "Datasets" section below), run:

```bash
# Quick smoke test first (a few minutes) -- confirms the whole pipeline
# works end-to-end on real data before committing to a full run:
python scripts/train_eurosat.py --epochs-per-stage 1 1 1 --image-sizes 64 96 128

# Full run, matching docs/BLUEPRINT.md Section 8's training recipe:
python scripts/train_eurosat.py --epochs-per-stage 5 5 10 --image-sizes 128 160 224
```

This builds a single stratified 70/15/15 split (saved to `data/eurosat_split.json` and reused on every later run — see `src/data/splits.py`), trains the dual-branch model with progressive resizing, fits per-branch temperature scaling (C4) on the validation split, and reports final test-set accuracy/F1/kappa/ROC-AUC/ECE, saved to `results/first_run_metrics.json`. The trained checkpoint (including fitted calibration temperatures) is saved to `checkpoints/dual_branch_best.pt`.

This is a **first real run on a single split** — not yet the full k=5 cross-validation + ablation protocol from Section 10-11 of the blueprint. `src/data/splits.py`'s split-building logic is written to be reused by a future ablation-runner script so every model variant is compared on identical folds.

## Configuration

`configs/config.yaml` is a single plain YAML file loaded with `pyyaml`,
not a Hydra multirun setup. This was a deliberate simplicity-first choice
for a single-student, single-primary-dataset project — Hydra's value is in
composing many config groups and launching multirun sweeps, which this
project doesn't need yet. If your ablation study grows into dozens of
sweep combinations, wrapping `src/training/train.py`'s `fit()` with
`@hydra.main` is a small, mechanical change at that point (uncomment
`hydra-core` in `requirements.txt`) — don't add it before you need it.

## Reproducibility

- `TrainConfig.seed` (default 42) seeds `torch.manual_seed` /
  `torch.cuda.manual_seed_all`; combine with `torch.use_deterministic_algorithms(True)`
  in your own training script if you need bit-exact determinism (this adds
  a real runtime cost on some ops, hence not hard-coded on by default).
- Log the exact `configs/config.yaml` used for every run alongside its
  checkpoint and metrics — that's the reproducibility unit that actually
  matters for a reviewer, more than any specific tracking tool.
- Weights & Biases / MLflow are optional (commented in `requirements.txt`);
  add whichever your university's compute cluster already supports.

## Datasets

This repo does not bundle satellite imagery. Download:
- **EuroSAT (MS/multispectral, 13-band GeoTIFF)** — required for real
  NDVI/NDWI/NDBI (`src/data/dataset.py: EuroSATMSDataset`). The commonly
  circulated RGB-only EuroSAT release cannot produce genuine NDVI (no NIR
  band) — see `src/data/spectral_indices.py` docstring.
- **RESISC45** — cross-dataset generalization only (Section 11 of the
  blueprint), not used for primary training.
