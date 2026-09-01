# C1 (Confidence-Aware Adaptive Fusion): k=5 Findings

**Status: C1's accuracy claim does not survive cross-validation.** Two architectures, five folds
each, 27,000 samples per variant, identical folds throughout (`data/eurosat_folds_k5.json`).

Sources: `results/cv/default/summary.json`, `results/cv/nonrgb/summary.json`,
`results/cv/comparisons/`. Recipe: `--epochs-per-stage 5 5 10 --image-sizes 128 160 224`.

---

## 1. Results

`default` = blueprint architecture (spectral branch sees RGB + NDVI, 4 ch).
`nonrgb` = branch-input ablation (spectral branch sees NIR + SWIR + NDVI, 3 ch, no RGB overlap).

| variant | accuracy | macro F1 | ECE | Brier |
|---|---|---|---|---|
| **default** — RGB branch | 0.9876 ± 0.0029 | 0.9873 | 0.0040 | 0.0211 |
| **default** — spectral branch | 0.9886 ± 0.0025 | 0.9879 | 0.0036 | 0.0189 |
| **default** — fixed 50/50 | **0.9918** ± 0.0013 | 0.9914 | 0.0059 | 0.0147 |
| **default** — C1 fusion | 0.9917 ± 0.0011 | 0.9913 | 0.0049 | 0.0141 |
| **nonrgb** — RGB branch | 0.9821 ± 0.0074 | 0.9820 | 0.0062 | 0.0279 |
| **nonrgb** — spectral branch | 0.9896 ± 0.0023 | 0.9890 | 0.0038 | 0.0180 |
| **nonrgb** — fixed 50/50 | **0.9929** ± 0.0009 | 0.9925 | 0.0105 | 0.0152 |
| **nonrgb** — C1 fusion | 0.9927 ± 0.0008 | 0.9923 | 0.0078 | 0.0134 |

## 2. What holds up

**Decision-level fusion beats either branch — robustly.** C1 vs the best single branch:
+0.0031 in both architectures, McNemar p=5.9e-10 (default) and p=4.3e-10 (nonrgb),
bootstrap CIs [+0.0023,+0.0045] and [+0.0014,+0.0046]. This is the paper's strongest result.

**~~Disjoint branch inputs help.~~ RETRACTED** -- `nonrgb` beat `default` by +0.0010 raw
(McNemar p=0.044), but that does NOT survive Holm correction over the 20-comparison family
(**p_adj = 0.307**) and sits *below* the 0.0007 seed noise floor (Section 8). Branch agreement
does fall 0.9857 -> 0.9786, so the mechanism is real; the accuracy claim is not. Report the
agreement change as an observation, not the accuracy delta as a result.

**Augmentation helps -- real, consistent, but small.** Dihedral augmentation (flips +
90-degree rotations, `--augment`). Verified across three seeds (Section 10):

- headline **0.9930 +- 0.0017** pooled over 15 folds (3 seeds x k=5)
- seed-matched effect **+0.0012**, positive on **3/3** seeds, paired-t p=0.032
- **Do NOT quote 0.9934** -- that is seed 42 alone, the most favourable of three draws
  (42/43/44 give 0.9934 / 0.9924 / 0.9932)

**C1 improves calibration over plain averaging** — better ECE on 5/5 folds in *both*
architectures (paired-t p=0.0127 and p=0.0323) and better Brier on 5/5 folds
(p=0.0187 and p=0.0675).

## 3. What does not hold up

**C1 gives no accuracy gain over its own 50/50 ablation.**

| | default | nonrgb |
|---|---|---|
| C1 − average | −0.0001 | −0.0002 |
| McNemar | p=0.4531 (2 vs 5) | p=0.0625 (**0 vs 5**) |
| paired-t | p=0.3046 | p=0.1419 |

In `nonrgb`, across 27,000 samples, C1 was right where the plain average was wrong **zero times**.
Ten folds, two architectures, never ahead. Note 0-vs-5 is the exact-binomial floor for McNemar,
so p=0.0625 cannot reach 0.05 at this discordance count — the direction is nonetheless entirely
against C1.

**Why**: the fusion weights barely move. Mean `w_rgb` is 0.4995 (default) and 0.4967 (nonrgb) —
i.e. C1 sits essentially *at* the equal-confidence point where it provably reduces to averaging.
Entropy-based confidence does not separate a correct branch from an incorrect one when both are
near-saturated at 98-99% accuracy.

**C2's headline number was a single-split artifact.** The single split showed RGB 0.9610 vs
spectral 0.9914 (~3pp for NDVI). Across k=5 the gap is **+0.0010, not significant**
(McNemar p=0.1948, paired-t p=0.4957, CI [−0.0009,+0.0034]). Do not cite ~3pp anywhere.

**~~Fusion hurts ECE in absolute terms.~~ CORRECTED** -- that claim came from equal-WIDTH ECE,
which is unreliable at this accuracy (nearly every sample lands in the top bin). Under equal-MASS
binning (`adaptive_expected_calibration_error`), pooled over 27,000 samples, the ranking reverses:

| variant | ECE (equal-width) | ECE (equal-mass) | Brier |
|---|---|---|---|
| RGB branch | 0.0034 | 0.0071 | 0.0211 |
| spectral branch | 0.0032 | 0.0072 | 0.0189 |
| fixed 50/50 | 0.0054 | 0.0052 | 0.0147 |
| **C1 fusion** | 0.0043 | **0.0040** | **0.0141** |

Equal-width binning understated the single branches' miscalibration by ~2x. C1 fusion is the
best-calibrated variant on the trustworthy metric, in absolute terms. **Report equal-mass ECE**;
if reporting equal-width too, say which is which.

## 4. Recommended framing

State C1 as a **reported negative result**, which BLUEPRINT Sections 9 and 18 already commit to:

> Entropy-based adaptive fusion weights do not improve accuracy over fixed 50/50 averaging on
> EuroSAT (k=5, p=0.45 and p=0.06 across two branch configurations), though they yield a small
> but consistent calibration improvement (better ECE on 10/10 folds). The mechanism fails because
> near-ceiling branches produce near-identical entropies: mean fusion weight 0.4995, i.e. the rule
> operating at the point where it is mathematically identical to its own ablation.

Lead the paper with what is significant after correction: decision-level fusion (+0.31pp over
the best branch, p_adj≈8e-9), the ResNet-50 comparison (+0.54 to +0.70pp, p_adj≈1e-25), and
augmentation (+0.12pp seed-matched, 3/3 seeds, p=0.032). Do NOT lead with branch inputs -- see
the retraction above.

## 5. Low-data ablation: the failure is the mechanism, not the ceiling

The obvious defence of C1 was that EuroSAT is saturated, leaving nothing to arbitrate. Tested
directly with `--train-subset-frac 0.10` (training split 17,787 -> 1,778; val/test kept full so
results stay comparable), tag `lowdata10`:

| variant | 10% data | full data |
|---|---|---|
| RGB branch | 0.9731 ± 0.0016 | 0.9876 |
| Spectral branch | 0.9769 ± 0.0015 | 0.9886 |
| Fixed 50/50 | 0.9819 ± 0.0021 | 0.9918 |
| C1 fusion | 0.9819 ± 0.0020 | 0.9917 |

The headroom did appear — branch agreement fell 0.9857 -> 0.9712, doubling disagreements from
~1.4% to ~2.9%. C1 still gained nothing: **6 discordant pairs each way, p=1.0**, mean fold
difference +0.0000, bootstrap CI [-0.0002, +0.0001].

**Direct mechanistic evidence**: mean `w_rgb` is 0.4991 at 10% data vs 0.4995 at full data — the
weights do not move between regimes (sd 0.0154 -> 0.0231). When two similarly-trained branches
disagree, both are typically confident, so normalized entropy cannot separate them. C1 operates
at the equal-confidence point where it is mathematically identical to averaging, regardless of
data regime.

Meanwhile fusion's value *grows* as data shrinks: +0.0050 over the best branch (p=2.2e-14) vs
+0.0031 at full data. And C1's calibration edge weakens: Brier better 5/5 (p=0.0027) but ECE only
4/5 (p=0.13, vs p=0.013 at full data).

Limit worth stating: at 10% data per-branch accuracy is still ~97.5% and disagreement under 3%,
so this reduces the ceiling rather than removing it. A 1-2% subset would test harder. The
regime-independence of the fusion weights is the stronger evidence.

## 6. Caveat and next test

EuroSAT is at ceiling — branches agree on 97.9-98.6% of samples, leaving 380-580 disagreements
per fold for fusion to arbitrate. This is a weak setting for *any* adaptive method. The
cross-dataset RESISC45 evaluation (BLUEPRINT Section 11) **cannot test C1**: RESISC45 is RGB-only
(no NIR, no SWIR), so NDVI cannot be computed and both branches would collapse to RGB. A genuine
multispectral cross-dataset test needs BigEarthNet or So2Sat, not RESISC45. Given the low-data
result above, the negative result no longer depends on that test.


---

## 7. Analysis from saved artefacts (no retraining)

### 7.1 Where fusion actually helps (`scripts/analyze_errors.py`)

Pooled over 27,000 samples, tag `default`:

- branches agree on **26,613 (98.57%)**, accuracy there 0.9955
- branches disagree on **387 (1.43%)**: RGB right 44.7%, spectral right 51.4%,
  **fused right 73.1%**
- an ORACLE that always picked the correct branch would score 0.9612, so fusion
  **captures 76.1% of the available headroom**
- fusion follows RGB on 50.65% of disagreements -- no systematic branch preference

This is a far stronger statement of the fusion result than "+0.31pp accuracy", and it is the
mechanism figure to put in the paper.

### 7.2 C2 is not merely non-significant -- it is backwards

Per-class F1, spectral branch minus RGB branch:

- mean delta on **vegetation** classes (the ones NDVI targets): **-0.0010**
- mean delta on **non-vegetation** classes: **+0.0021**
- largest gain: SeaLake **+0.0101** (water); largest loss: Pasture **-0.0106** (vegetation)

NDVI was introduced to disambiguate vegetation classes and per-class it does the opposite. What
gain exists comes from water/built-up discrimination -- i.e. what NDWI/NDBI target, not NDVI.

The five most frequent confusions are all vegetation-vs-vegetation
(PermanentCrop->HerbaceousVegetation 26, Pasture->HerbaceousVegetation 21, PermanentCrop->AnnualCrop 18,
Pasture->AnnualCrop 18, AnnualCrop->PermanentCrop 17), which is the accuracy ceiling on EuroSAT.

Against the ResNet-50 baseline the dual-branch model wins **every class**, with the largest margins
on the hardest ones: PermanentCrop +0.0115, AnnualCrop +0.0095, HerbaceousVegetation +0.0076.

### 7.3 Grad-CAM corroborates the C1 negative (`scripts/make_gradcam_figures.py`)

`results/figures/gradcam_per_class_fold0.png`: on correctly-classified samples the RGB and spectral
CAMs are visually near-identical on 8 of 10 classes -- the qualitative counterpart of 98.57%
agreement and mean fusion weight 0.4995. `gradcam_disagreements_fold0.png` shows the minority where
the branches attend to genuinely different evidence (e.g. RGB fixating on a water strip -> "River",
spectral attending to field texture -> correct "AnnualCrop").

### 7.4 C4 is the best-supported contribution (`scripts/make_calibration_figures.py`)

Temperature scaling (fitted T = 0.580-0.605 across folds) reduces pooled ECE from **0.0951 to
0.0043 -- a 22x reduction**. The model is severely UNDER-confident before calibration (mean
confidence 0.8967 against ~100% top-bin accuracy), which label smoothing at 0.1 explains. This
effect is orders of magnitude above the seed noise floor, unlike C1/C2/C3.

Limitation to state: at ~99% accuracy so many samples sit at confidence ~1.0 that quantile bin
edges collapse, leaving only 2-3 distinct bins. Calibration analysis here is inherently coarse.

### 7.5 Efficiency, including the edge case (`scripts/measure_efficiency.py`)

| config | params | GMACs | batch-1 GPU | batch-1 CPU |
|---|---|---|---|---|
| single B0, 224px | 4.02M | 0.769 | 11.45 ms | 21.24 ms |
| dual-branch, 224px | 8.04M | 1.545 | 24.52 ms (40.8/s) | 40.70 ms (24.6/s) |
| dual-branch, 64px native | 8.04M | **0.129** | -- | 20.76 ms (48.2/s) |

- The model runs on **CPU with no GPU at all** at 24.6 img/s (224px) / 48.2 img/s (64px) --
  the concrete claim the edge-deployment framing needs.
- Costs **2.00x the parameters and ~2x the latency** of a single EfficientNet-B0. State this
  plainly; it is the price of the +0.31pp fusion gain.
- Upsampling 64px patches to 224px costs **12x the MACs** for no new information. A resolution
  ablation is the largest untapped efficiency lever in the project.
- ECA: **10 parameters total** (1.24e-06 of the model), latency unchanged to 3 s.f. Its *cost*
  claim is verified; its *benefit* claim is not (see CLAUDE.md).

## 8. Seed noise floor -- what is and is not claimable

Three seeds (42/43/44) of the identical `default` configuration:

- k=5 mean accuracy per seed: 0.9918 / 0.9914 / 0.9921 -> **range 0.0007, sd 0.0004**
- mean per-fold range across seeds 0.0017, max 0.0041 (fold 3)
- fold-to-fold sd is roughly **halved by fusion**: RGB branch +-0.0029/0.0043/0.0046 vs
  fused +-0.0013/0.0023/0.0010. Fusion improves *stability*, an independent claim worth making.

**Rule for the paper: differences below ~0.001 in k=5 mean accuracy are not distinguishable from
seed variance.** That disqualifies C1 (-0.0001), C2 (+0.0010), C3 (+0.0010 fused) and the
`nonrgb` variant (+0.0010). It leaves fusion-vs-best-branch (+0.0031) and fusion-vs-ResNet-50
(+0.0054) comfortably clear at 4x and 8x the floor.


## 9. Final configuration and the claimable set

`augmented` = `default` + dihedral augmentation. k=5, identical folds.

| variant | accuracy | macro F1 | ECE |
|---|---|---|---|
| RGB branch | 0.9864 ± 0.0033 | 0.9864 | 0.0041 |
| spectral branch | 0.9912 ± 0.0006 | 0.9907 | 0.0031 |
| fixed 50/50 | 0.9933 ± 0.0015 | 0.9930 | 0.0066 |
| **C1 fusion** | **0.9934 ± 0.0015** | 0.9930 | 0.0056 |

Mechanism note: augmentation did **not** increase branch complementarity (agreement 0.9854 vs
0.9857; mean w_rgb 0.4982 vs 0.4995). It improved the SPECTRAL branch's generalisation and
stability (0.9886 -> 0.9912, fold sd 0.0025 -> **0.0006**, a 4x variance reduction) while the RGB
branch slightly regressed. So it is a regularisation effect, not a diversity effect.

### What is claimable (Holm-adjusted, family of 20, vs a 0.0007 seed floor)

| claim | effect | p_adj | vs floor |
|---|---|---|---|
| fusion vs ResNet-50 baseline (augmented) | +0.0070 | 2.4e-25 | 10x |
| fusion vs ResNet-50 baseline (default) | +0.0054 | 2.6e-14 | 8x |
| fusion vs best single branch | +0.0031 | 7.7e-09 | 4x |
| augmentation vs no augmentation | +0.0017 | 0.010 | 2.4x |
| temperature scaling (ECE 0.0951 -> 0.0043) | 22x | -- | far above |

### What is NOT claimable

| claim | effect | why not |
|---|---|---|
| C1 adaptive fusion | -0.0001 | p_adj 0.906; below floor; mechanism explained |
| C2 NDVI branch | +0.0010 | p 0.195; below floor; per-class it *hurts* vegetation |
| C3 ECA attention | +0.0010 (leans negative) | p_adj 0.254; below floor; 10 params |
| `nonrgb` branch inputs | +0.0010 | p_adj 0.307; below floor |

Single-branch caveat: `noattention_rgb_branch_only` vs `default_rgb_branch_only` survives at
p_adj = 0.0225 (+0.0019), but it is a single-seed comparison at 2.7x the floor. Do not claim ECA
hurts without seed-matched `noattention` runs.


## 10. Seed verification of the headline configuration

`augmented` re-run at seeds 43 and 44 (`augmented_seed43`, `augmented_seed44`), identical folds.

| config | k=5 mean per seed (42/43/44) | range | sd |
|---|---|---|---|
| augmented | 0.9934 / 0.9924 / 0.9932 | 0.0010 | 0.0005 |
| default | 0.9917 / 0.9914 / 0.9921 | 0.0007 | 0.0004 |

Seed-MATCHED augmentation effect (the correct test -- pairing by seed removes seed variance):

| seed | augmented | default | delta |
|---|---|---|---|
| 42 | 0.9934 | 0.9917 | +0.0017 |
| 43 | 0.9924 | 0.9914 | +0.0010 |
| 44 | 0.9932 | 0.9921 | +0.0010 |

Mean **+0.0012**, positive on 3/3 seeds, paired-t p=0.032, and 1.2x the augmented config's own
seed range. Real and consistent, but small -- report it as such.

**Reporting rules this establishes:**

1. The headline is **0.9930 +- 0.0017** (15 folds, 3 seeds). Quoting 0.9934 is quoting the best
   of three seeds.
2. Quote both error bars where space allows: fold-level (+-0.0015) and seed-level (+-0.0005).
3. The single-seed estimate of the augmentation effect (+0.0017) overstated it by ~40%. Any
   future sub-0.002 effect must be seed-matched before it is claimed.
