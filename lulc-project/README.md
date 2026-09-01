# lulc-project

**Project overview, results, and quick start are in the [root README](../README.md).**

This file covers only what is specific to working *inside* this directory. All commands below
are run from here.

---

## Environment

```bash
pip install -r requirements.txt
```

PyPI's default Windows `torch` wheel is **CPU-only**. For GPU training install from the PyTorch
index instead:

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

## Tests

```bash
python -m pytest tests/ -v
```

111 tests. 60 are pure NumPy/SciPy/sklearn and run anywhere; the rest self-skip via
`pytest.importorskip` when torch is absent. `conftest.py` puts this directory on `sys.path`, so
pytest works from any working directory — do not add `__init__.py` shims to fix import errors.

Before spending GPU hours, confirm `test_overfit_single_batch_sanity_check` passes. It is the
canary for silently broken model/training wiring.

## Dataset

Not committed (2 GB). Download and extract:

```bash
python scripts/download_eurosat.py
```

It lands at `data/eurosat/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/`,
which is exactly the `TIF_ROOT` constant the scripts expect. Must be the **multispectral** release
— NDVI needs a real NIR band (B08) and cannot be approximated from RGB.

Split and fold definition files (`data/*.json`) are also untracked: they store absolute paths from
the machine that built them, so they are not portable. They regenerate identically from the seed —
`build_or_load_kfolds()` runs `StratifiedKFold` over a sorted file listing — so every variant is
still compared on the same folds.

## Configuration

`configs/config.yaml` supplies **defaults**; an explicit command-line flag always overrides them.
Every run also writes `resolved_config.json` beside its results with the effective values after
overrides — that file, not `config.yaml`, is the reproducibility record, since the config can
drift after a run.

Keys that no runner consumes are printed at startup rather than silently ignored.

## Running experiments

Single split, one model:

```bash
python scripts/train_eurosat.py --epochs-per-stage 5 5 10 --image-sizes 128 160 224 --num-workers 2
```

Cross-validation (the protocol all reported results use):

```bash
python scripts/run_cross_validation.py --tag augmented --augment -k 5 --num-workers 2
```

Resumable per fold — a fold whose `results/cv/<tag>/fold_<i>.json` exists is skipped, so an
interrupted sweep continues rather than restarting. Use `--only-fold N` to split work across
sessions.

Analysis and figures, none of which need training:

```bash
python scripts/predict.py --disagreements --random 8 --seed 3 --figure
```

```bash
python scripts/analyze_errors.py --tag default --compare baseline_resnet50
```

```bash
python scripts/compare_variants.py --a augmented:c1_confidence_fusion --b default:c1_confidence_fusion
```

`scripts/measure_efficiency.py`, `make_gradcam_figures.py` and `make_calibration_figures.py`
produce the Section 12/13 numbers and figures.

## Hardware notes (developed on a laptop RTX 3050, 4 GB VRAM / 7.3 GB RAM)

- **Use `--num-workers 2`.** With 4 workers, 224px exhausted the Windows commit limit
  (`Couldn't open shared file mapping ... error code 1455`). Next levers: `--batch-size 16`
  (1.60 GB VRAM at 224px vs 3.07 GB at batch 32), then `--num-workers 0`.
- **Run on mains, and disable sleep.** Windows modern standby powers the GPU down and invalidates
  the CUDA context, which surfaces as `CUDA error: unknown error`, `illegal memory access`, or
  `cuDNN error: EXECUTION_FAILED` at an arbitrary point. Every "random" GPU fault in this project
  was this. Fix: `powercfg /change standby-timeout-ac 0`, and set lid-close to do nothing on AC.
  On battery the Balanced plan also halves throughput (~2.5× slower epochs).
- The loader decodes 27,000 13-band GeoTIFFs per epoch and is I/O-bound; GPU utilisation sits
  around 65%. Precomputing the dataset into a memory-mapped array (~2.9 GB at float16) is the
  largest available speedup.

## Design decisions that are easy to break

- **One combined tensor, transformed once.** The RGB branch input is always a *slice* of the same
  transformed tensor the spectral branch sees. Building and transforming two tensors independently
  reintroduces a spatial-misalignment bug that was already found and fixed once.
- **Both branches are supervised directly** on their own logits. An unsupervised branch can hide
  behind the other, which would make the per-branch ablations meaningless.
- **Fusion has two implementations on purpose.** `src/models/fusion.py` is the NumPy reference;
  `fusion_torch.py` is the in-loop version. A test asserts they agree numerically — change one,
  change both.
- **The fusion rule reduces to plain averaging at equal confidence.** That identity is what makes
  "fixed 50/50" an *ablation* rather than a separate baseline.
- **`CLASS_NAMES` ordering defines the integer labels** and must never change once a split or
  checkpoint exists.
- **Don't store a module object on `EuroSATMSDataset`.** Windows spawns dataloader workers, which
  pickles the dataset; a module attribute breaks `--num-workers > 0`.
