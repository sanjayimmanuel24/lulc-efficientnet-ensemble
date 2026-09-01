# Dual-Branch EfficientNet-B0 for Sentinel-2 Land Cover Classification

A compact two-branch CNN for EuroSAT-MS land use / land cover classification, evaluated with
5-fold cross-validation, a literature baseline trained on identical folds, multiple-comparison
correction, and a measured seed noise floor.

**Headline: 0.9930 ± 0.0017 accuracy — beating a ResNet-50 baseline on the same folds by
+0.70pp (Holm-adjusted p ≈ 1e-25) with 8.0M parameters against 25.6M, at 24.6 images/sec on CPU.**

---

## Results

k=5 stratified cross-validation over all 27,000 EuroSAT-MS tiles; every variant evaluated on
identical folds. Headline pooled over 3 seeds (15 folds).

| model | accuracy | params | notes |
|---|---|---|---|
| **Dual-branch + fusion (augmented)** | **0.9930 ± 0.0017** | 8.0M | this work |
| Dual-branch + fusion | 0.9917 ± 0.0013 | 8.0M | without augmentation |
| Spectral branch alone | 0.9886 ± 0.0025 | 4.0M | |
| RGB branch alone | 0.9876 ± 0.0029 | 4.0M | |
| ResNet-50 baseline | 0.9864 ± 0.0018 | 25.6M | trained on the same folds |

The baseline lands within 0.1pp of Helber et al. (2019)'s published 98.57%, so the pipeline
reproduces the reference result before improving on it.

### What holds up

| claim | effect | evidence |
|---|---|---|
| Fusion beats ResNet-50 | +0.0054 … +0.0070 | Holm-adj. p ≈ 1e-25 |
| Fusion beats the best single branch | +0.0031 | Holm-adj. p ≈ 8e-9 |
| Temperature scaling improves calibration | ECE 0.0951 → 0.0043 | 22× reduction |
| Augmentation (dihedral) | +0.0012 | positive on 3/3 seeds, p=0.032 |

**The gain comes from fusion, not from a bigger backbone.** A single EfficientNet-B0 branch is
statistically indistinguishable from ResNet-50 (+0.0013, p=0.11). On the 1.43% of tiles where the
two branches disagree, RGB alone is right 44.7% of the time and the spectral branch 51.4%, while
fusion reaches **73.1% — 76% of the oracle ceiling**. Fusion also roughly halves run-to-run
variance.

### What does not hold up (reported, not removed)

| hypothesis | outcome |
|---|---|
| Entropy-weighted adaptive fusion | No gain over fixed 50/50 averaging (p_adj 0.906). Mean fusion weight **0.4995** — the point where the rule provably *is* plain averaging. |
| NDVI spectral branch | +0.0010, not significant (p=0.195). Per-class it is *backwards*: helps water, hurts the vegetation classes it targeted. |
| ECA channel attention | No benefit, leans slightly negative. Cost claim verified exactly: **10 parameters**, 1.24e-06 of the model. |
| Disjoint branch inputs | +0.0010 raw, does not survive correction (p_adj 0.307). |

All four sit below the **measured seed noise floor of 0.0007**, obtained by re-running an identical
configuration at three seeds. That floor is why they are not claimed. Full evidence and the
recommended paper framing: [`docs/C1_FINDINGS.md`](lulc-project/docs/C1_FINDINGS.md).

---

## Quick start

```bash
cd lulc-project
pip install -r requirements.txt
```

The CUDA build of PyTorch on Windows needs the PyTorch index — PyPI's default wheel is CPU-only:

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

Run the test suite (111 tests, ~25s, no GPU or dataset needed):

```bash
python -m pytest tests/ -v
```

### Classify real tiles with a trained model

```bash
python scripts/predict.py --random 8 --figure
```

Prints the fused prediction plus what each branch predicted independently and the fusion weight.
To see the cases where the architecture actually matters — random tiles are ~98.6% branch
agreement, so they make the design look redundant:

```bash
python scripts/predict.py --disagreements --random 8 --seed 3 --figure
```

### Reproduce the analysis (no training required)

```bash
python scripts/analyze_errors.py --tag default --compare baseline_resnet50
```

```bash
python scripts/compare_variants.py --a augmented:c1_confidence_fusion --b default:c1_confidence_fusion
```

Per-fold metrics and per-sample predictions for every sweep are committed under
`lulc-project/results/cv/`, so the statistical results can be re-derived without re-running
~40 GPU-hours of training.

---

## Reproducing the training

The dataset is not committed (2 GB). Fetch it:

```bash
python scripts/download_eurosat.py
```

Then a full cross-validation sweep (~5 hours on a laptop RTX 3050):

```bash
python scripts/run_cross_validation.py --tag augmented --augment -k 5 --num-workers 2
```

Sweeps are **resumable per fold** — a fold whose metrics file exists is skipped, so an interrupted
run continues rather than restarting. Fold definitions regenerate deterministically from the seed,
so every variant is compared on identical data.

Useful flags: `--baseline-backbone resnet50` (single-backbone literature baseline),
`--attention none` (attention ablation), `--spectral-branch-mode nonrgb` (disjoint branch inputs),
`--train-subset-frac 0.1` (low-data ablation), `--seed N` (seed variance).

---

## How it works

```
13-band Sentinel-2 GeoTIFF
  → reflectance/10000, NDVI computed from real NIR (B08) and Red (B04)
  → one combined tensor (RGB + index channels), transformed ONCE
  → branch A sees the RGB slice; branch B sees RGB + NDVI
      each: EfficientNet-B0 → ECA → global pool → linear head
  → per-branch softmax → confidence-weighted fusion → prediction
  → temperature scaling fitted on validation → calibrated output
```

Both branches are supervised directly on their own logits, so neither can hide behind the other
and the per-branch ablations stay meaningful. The single-transform-then-slice design guarantees
the two branches always see identically-oriented views of the same tile.

## Repository layout

```
lulc-project/
  src/           data pipeline, model, fusion, calibration, metrics, statistics
  scripts/       training, cross-validation, comparison, analysis, figures, demo
  tests/         111 tests (60 pure NumPy/SciPy, rest torch-gated)
  docs/
    BLUEPRINT.md      design document; C1-C4 hypotheses with measured verdicts
    C1_FINDINGS.md    full experimental findings and paper framing
  results/       per-fold metrics, per-sample predictions, figures
  configs/       config.yaml (loaded as CLI defaults; runs dump resolved_config.json)
```

## Figures

- `results/figures/gradcam_per_class_fold0.png` — what each branch attends to
- `results/figures/gradcam_disagreements_fold0.png` — where the branches diverge
- `results/figures/reliability_default.png` — calibration before and after temperature scaling
- `results/figures/predictions_demo.png` — predictions on held-out tiles

## Limitations

- **EuroSAT is saturated.** Per-branch accuracy is ~98.9%, and the branches agree on 98.6% of
  samples, leaving little room for any adaptive method. A 10%-training-data ablation was run to
  test whether this explains the negative results; it did not change them.
- **Single dataset.** RESISC45 cannot test this architecture — it is RGB-only, so NDVI cannot be
  computed and both branches would collapse to RGB. A genuine cross-dataset test needs a
  multispectral corpus such as BigEarthNet or So2Sat.
- **2× the cost of a single backbone.** Two EfficientNet-B0 backbones means 8.0M parameters and
  ~2× the inference latency of one branch, for +0.31pp. That trade-off is measured and stated
  rather than left implicit.
- **Upsampling dominates compute.** EuroSAT tiles are natively 64×64; running at 224×224 costs
  12× the MACs for no additional information. A resolution ablation is the largest untapped
  efficiency lever here.

## Dataset

EuroSAT-MS (multispectral, 13-band GeoTIFF), 27,000 tiles across 10 classes.
Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
Classification*, IEEE JSTARS, 2019. The RGB-only release cannot be used — NDVI requires a real
near-infrared band.
