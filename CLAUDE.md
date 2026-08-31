# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

The actual project root is `lulc-project/`, one level below this file. **All commands below are
run from inside `lulc-project/`.** Paths in the rest of this document are relative to that folder.

## Commands

```bash
pip install -r requirements.txt
```

```bash
pytest tests/ -v
```

- 60 tests are pure NumPy/SciPy/sklearn and run anywhere (68 total).
- 8 more live in `tests/test_models_torch.py` and self-skip via `pytest.importorskip` when torch
  is absent. Torch is **not** installed in the current environment, so those skip by default.
- `conftest.py` at the project root puts `lulc-project/` on `sys.path`, so `pytest` works from any
  working directory — do not add `__init__.py` shims or path hacks to fix import errors.

Single test / single file:

```bash
pytest tests/test_fusion_logic.py::test_equal_confidence_reduces_to_simple_average -v
```

Before spending GPU hours, confirm `test_overfit_single_batch_sanity_check` passes — it is the
canary for a silently broken model/training wiring.

Training (smoke test, then full run):

```bash
python scripts/train_eurosat.py --epochs-per-stage 1 1 1 --image-sizes 64 96 128
```

```bash
python scripts/train_eurosat.py --epochs-per-stage 5 5 10 --image-sizes 128 160 224
```

Cross-validation and variant comparison (docs/BLUEPRINT.md Sections 10-11):

```bash
python scripts/run_cross_validation.py --tag default --spectral-branch-mode rgb_plus_indices
```

```bash
python scripts/compare_variants.py --a default:c1_confidence_fusion --b default:fixed_average_50_50
```

`run_cross_validation.py` is resumable — a fold whose `results/cv/<tag>/fold_<i>.json` exists is
skipped, so an interrupted sweep continues rather than restarting. Delete that file to force a
re-run, and use `--only-fold N` to split work across sessions. Every variant must consume the same
`data/eurosat_folds_k5.json`; `compare_variants.py` refuses to run paired tests otherwise.

There is no linter, formatter, or build step configured.

### Dataset location

The EuroSAT-MS GeoTIFFs live at
`data/eurosat/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/<ClassName>/`
(27,000 files across the 10 classes), which is exactly the `TIF_ROOT` constant in
`scripts/train_eurosat.py`. `data/` is generated content, not tracked — if the project is moved to
another machine, either copy `data/eurosat/` across or re-run `scripts/download_eurosat.py`, which
downloads and extracts to that same path.

## Architecture

The design document `docs/BLUEPRINT.md` is the spec — Sections 4–11 define the four contributions,
the ablation table, and the statistical protocol. If code and blueprint disagree, flag it rather
than picking one. `docs/PROJECT_SUMMARY_FOR_REVIEW.md` is a plain-language version for faculty.

Data flow, end to end:

```
13-band Sentinel-2 GeoTIFF
  → EuroSATMSDataset          reflectance/10000, slices bands, computes NDVI etc.
  → combined (3+I, H, W)      RGB channels + I spectral-index channels, ONE tensor
  → EuroSATTransform          resize + ImageNet-normalize the RGB channels only
  → DualBranchEfficientNet    branch A sees combined[:3], branch B sees all 3+I
      each branch: timm EfficientNet-B0 → ECA attention → GAP → Linear → logits
  → confidence_weighted_fusion(softmax_A, softmax_B) → fused_probs
  → temperature scaling (fit per branch on val) → calibrated metrics
```

Load-bearing design decisions that are easy to break:

- **One combined tensor, transformed once.** The RGB branch input is always a *slice* of the same
  transformed tensor the spectral branch sees. Building and transforming two tensors independently
  reintroduces a spatial-misalignment bug that was already found and fixed once.
- **Both branches are supervised directly** on their own logits (`train.py::compute_loss`); the
  fused output is never optimized. An unsupervised branch can hide behind the other, which would
  make the per-branch ablations meaningless.
- **Fusion has two implementations on purpose.** `src/models/fusion.py` is the NumPy reference
  (the formula whose correctness the paper's claims rest on, unit-tested standalone);
  `src/models/fusion_torch.py` is the in-loop torch version. `test_models_torch.py` asserts they
  agree numerically. Change one → change both, and keep the equivalence test green.
- **The fusion rule reduces to plain averaging when both branches are equally confident.** That
  identity is what makes "fixed 50/50 ensemble" an *ablation* of this method rather than a separate
  baseline implementation. Do not "improve" the weighting in a way that breaks it.
- **The split is built once and reused.** `src/data/splits.py` writes `data/eurosat_split.json` on
  first run and loads it verbatim afterwards, because the paired significance tests (McNemar,
  Wilcoxon, paired-t) assume every variant is evaluated on identical held-out samples. Never
  regenerate a fresh random split per experiment. `CLASS_NAMES` ordering defines the integer labels
  and must never change once a split or checkpoint exists.
- **The spectral branch's input is configurable** (`--spectral-branch-mode`, see
  `src/data/dataset.py::spectral_in_chans`): `rgb_plus_indices` (default, blueprint),
  `nonrgb` (NIR+SWIR+indices, disjoint from branch A), `indices_only`. RGB always leads the
  stacked tensor so the single-transform/slice alignment invariant holds in every mode.
- **Multispectral release only.** NDVI needs a real NIR band (B08); it cannot be approximated from
  an RGB JPEG. `src/data/dataset.py::BAND_INDEX` maps band names into the 13-band stack.

## Project constraints

- Contributions are deliberately capped at four (adaptive fusion, genuine NDVI branch, ECA
  attention, calibration-coupled evaluation). `BLUEPRINT.md` Section 15 records, individually, why
  ~30 other candidate techniques (distillation, ArcFace, MixUp, contrastive pretraining, …) were
  rejected. Do not reintroduce them; a rejected item returning needs an explicit discussion first.
- Simplicity-first is a stated choice, not an oversight: plain YAML config over Hydra
  (`configs/config.yaml`), plain PyTorch loop over Lightning. Both files document the exact
  condition under which upgrading would be worth it. Don't preempt it.
- ECA was chosen over SE/CBAM specifically on parameter cost. Swapping in a heavier attention
  module undercuts the paper's efficiency framing.

## Current status

Full training run (`--epochs-per-stage 5 5 10 --image-sizes 128 160 224 --num-workers 2`) completed
2026-08-20 in 88.3 minutes on a laptop RTX 3050: **test accuracy 0.9916, macro-F1 0.9912, kappa
0.9907, ROC-AUC 0.9997, ECE 0.0117**; best val 0.9926 (stage 2, epoch 7). Metrics in
`results/first_run_metrics.json`, weights in `checkpoints/dual_branch_best.pt` (best eligible) and
`dual_branch_last.pt` (latest epoch). All 56 tests pass.

This is still **one stratified split, not the k=5 protocol**. Section 10-11's ablations,
cross-validation, paired significance tests, and the paper draft remain outstanding, as does any
per-branch (RGB-only / spectral-only) comparison needed to show C1 fusion actually earns its place.

## C3 (ECA attention) does not earn its place either

`results/cv/noattention/` (k=5, ECA replaced by nn.Identity, everything else identical to
`default`). Removing ECA is **better**, not neutral:

| variant | ECA removed | with ECA | delta | exact-binomial p | Holm-adjusted |
|---|---|---|---|---|---|
| RGB branch | 0.9895 | 0.9876 | +0.0019 | 0.0020 | **0.0205** |
| fixed 50/50 | 0.9928 | 0.9918 | +0.0010 | 0.034 | 0.272 |
| C1 fusion | 0.9927 | 0.9917 | +0.0010 | 0.028 | 0.254 |

- Only the RGB-branch comparison survives Holm correction; paired-t is non-significant throughout
  (p=0.069-0.211) because n=5 folds has little power against effects this small.
- `scripts/measure_efficiency.py`: ECA is 10 parameters total (1.2e-06 of the model), latency
  identical to 3 significant figures. So C3's *cost* claim is verified; its *benefit* claim is not.
- **Caveat**: +0.001-0.002 is exactly the magnitude where uncontrolled seed variance could explain
  the result. Adding an ECA layer also shifts RNG consumption at init, so the two sweeps are
  effectively different random draws. `default_seed43` / `default_seed44` exist to establish that
  noise floor — do not write up the ECA finding until they are in.

## Settled research finding: C1 fails, the ensemble succeeds

Full k=5 results and recommended paper framing: `docs/C1_FINDINGS.md`. Headlines:

- **C1 gives no accuracy gain over its own 50/50 ablation.** default: -0.0001, McNemar p=0.45;
  nonrgb: -0.0002, p=0.0625 with C1 right on *zero* samples the average missed. Ten folds, two
  architectures, never ahead. Mean fusion weight is 0.4995 — C1 sits at the equal-confidence
  point where it provably *is* plain averaging.
- **Decision-level fusion itself is a strong result**: +0.0031 over the best single branch,
  p~1e-10 in both architectures.
- **Disjoint branch inputs (`--spectral-branch-mode nonrgb`) beat the blueprint default** by
  +0.0010 (McNemar p=0.044); branch agreement drops 0.9857 -> 0.9786.
- **C2's ~3pp NDVI claim was a single-split artifact.** Across k=5 the spectral branch beats the
  RGB branch by +0.0010, not significant (p=0.19). Do not cite 3pp.
- **C1's surviving claim is calibration only, and only relative to the 50/50 average**: better ECE
  on 5/5 folds in both architectures (paired-t p=0.013, p=0.032). Both fused variants are *worse*
  calibrated than either single branch in absolute terms.
- With n=5 folds, two-sided Wilcoxon bottoms out at p=0.0625 and McNemar at 0-vs-5 likewise —
  neither can reach 0.05 at these counts. Don't read those as failures to replicate.

Report this honestly rather than tuning toward a positive result; the blueprint (Sections 9, 18)
already commits to doing so. RESISC45 (Section 11) is the remaining fair test of whether C1's
failure is dataset-limited.

## Training-loop behaviour worth knowing

- **Checkpoint selection is final-stage-only** (`TrainConfig.checkpoint_selection`, default
  `"final_stage"`). Progressive resizing validates each stage at a different resolution, so
  comparing val accuracy across stages is not apples-to-apples — a 96px-stage model could otherwise
  win and then be evaluated at 128px (this cost ~5pp in the first smoke test). Warmup stages are
  logged as ineligible. `"global"` restores the old behaviour for comparison.
- **Two checkpoints are written.** `checkpoint_path` holds the best *eligible* weights;
  `last_checkpoint_path` holds the latest epoch unconditionally, so a crash during the final
  stage's first validation cannot discard every warmup epoch. Both use write-then-rename.
- **There is still no resume.** Only weights are persisted, not optimizer/scheduler/epoch state,
  so an interrupted run restarts training from scratch.

## Environment constraints (this machine)

- **Run on mains, not battery.** On battery the Balanced plan throttles hard: identical 128px
  epochs took 431-525s on DC versus 174-254s on AC (~2.5x). This is separate from the standby
  issue above and does not cause faults, only slowdown — but it silently doubles sweep times.
- 7.3 GB RAM is the binding constraint, more than the 4 GB GPU. `--num-workers 4` at 224px
  exhausted the Windows commit limit (`Couldn't open shared file mapping ... error code 1455`).
  **Use `--num-workers 2`.** Next levers if it recurs: `--batch-size 16` (224px peaks at 1.60 GB
  VRAM there vs 3.07 GB at batch 32), then `--num-workers 0`.
- `pin_memory` is opt-in via `--pin-memory`, not always-on: pinned pages are non-pageable and add
  to the same commit pressure.
- PyPI's default Windows torch wheel is CPU-only; the CUDA build needs
  `--index-url https://download.pytorch.org/whl/cu126`.
- `--num-workers > 0` works only because `EuroSATMSDataset` no longer stores the rasterio module on
  the instance (spawn pickles the dataset). Don't reintroduce a module attribute there.
- **Windows modern standby kills running CUDA jobs — this caused every "random" GPU fault.**
  Seven faults across the project (`CUDA error: unknown error`, `illegal memory access`,
  `cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`) were all the same thing: entering standby powers
  the GPU down and invalidates the CUDA context, so the next kernel launch fails with whatever
  error it happens to hit. Two faults matched Kernel-Power standby events to the same minute
  (28-08 12:55, 26-08 20:10). Wi-Fi dropping at the same moment is the same cause, not a separate
  problem. It happens on AC as well as battery, which is what made it look like flaky hardware.
  Fix applied: `powercfg /change standby-timeout-ac 0` (verified: AC index 0x0, DC left at 3600s).
  Do NOT diagnose these as GPU instability — check standby first.
- The loader decodes 27,000 13-band GeoTIFFs per epoch, and GPU utilisation sits well under 100%
  as a result. Precomputing the dataset into a memory-mapped array (~2.9 GB at float16) is the
  single biggest available speedup, and matters more before any k=5 ablation sweep.
