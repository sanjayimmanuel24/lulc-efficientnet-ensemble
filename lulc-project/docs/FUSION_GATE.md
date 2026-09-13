# A Learned Fusion Gate: fixing the failure C1 exposed

**Status: POSITIVE, large, and mechanistically explained — with one honest cost.**

Entropy-weighted fusion (C1) cannot detect a broken branch. This replaces it with a ~1,100-parameter
gate that can, recovering ~99% of the accuracy that band dropout destroys.

Source: `scripts/train_fusion_gate.py`, `results/cv/augmented/fusion_gate.json`,
`results/figures/fusion_gate_augmented.png`. Backbones frozen; the gate trains in seconds on
cached logits.

---

## 1. The diagnosis this follows from

`docs/BAND_DROPOUT.md`: with the NIR-derived channel destroyed, the spectral branch drops to
0.1445 (chance = 0.10) while the RGB branch is untouched at 0.9864 — yet fused accuracy is only
0.9208 and C1's weight moves from 0.4982 to just 0.5973. It needed to approach 1.0.

**Why entropy fails, precisely.** Normalised predictive entropy is computed from the *softmax*,
which divides out logit magnitude. A network fed noise still produces a peaked distribution — it is
**confidently wrong**, and entropy sees only peakedness, not correctness. Logit *magnitude* is not
scale-invariant: max-logit and logsumexp (the energy score, Liu et al. 2020) collapse when a branch
receives garbage.

So the fix is not "a bigger fusion module". It is giving the gate a signal entropy structurally
cannot access.

## 2. Design

- **~1,100 parameters.** A 3-layer MLP on logit-derived features -> one sigmoid weight for the RGB
  branch. Both backbones frozen; no retraining.
- **Class-agnostic features.** Per branch: confidence, entropy-confidence, top-1/top-2 margin,
  **max logit**, **logsumexp (energy)**, logit spread. Joint: total-variation distance, argmax
  agreement, confidence gap, energy gap. The gate never sees class identity, so it cannot memorise
  "NDVI helps for Forest" — only how reliable each branch currently looks.
- **Trained with band-dropout augmentation** on validation at severities {0.0, 0.5, 1.0}. A gate
  trained only on clean data has never seen a broken branch and inherits C1's blindness.
- **Per fold, validation only**, evaluated on the held-out test fold.

## 3. Results (tag `augmented`, k=5, 27,000 test tiles)

Gate trained on 70% of validation; the remaining 30% is held out **solely** to choose the stopping
point. Baselines are evaluated with their fitted temperatures (their best configuration); the gate
uses raw logits, because logit magnitude is exactly what temperature scaling would rescale away.

| method | clean | 25% | 50% | 75% | 100% corrupt | total drop |
|---|---|---|---|---|---|---|
| Fixed 50/50 | 0.9933 | 0.9859 | 0.9709 | 0.9374 | 0.9211 | -7.22pp |
| C1 entropy fusion | 0.9934 | 0.9861 | 0.9714 | 0.9376 | 0.9208 | -7.26pp |
| **Learned gate** | 0.9910 | 0.9893 | 0.9887 | 0.9874 | **0.9860** | **-0.50pp** |
| RGB branch alone (fallback ceiling) | 0.9864 | 0.9864 | 0.9864 | 0.9864 | 0.9864 | 0 |
| **mean gate weight on RGB** | **0.521** | 0.911 | 0.968 | 0.979 | **0.988** | |

Gate minus entropy fusion:

| severity | delta | folds better | paired-t |
|---|---|---|---|
| 0.00 | **-0.0023** | **0/5** | **0.045** |
| 0.25 | +0.0031 | 4/5 | 0.185 |
| 0.50 | +0.0174 | 4/5 | 0.115 |
| 0.75 | +0.0498 | 5/5 | 0.102 |
| 1.00 | **+0.0652** | **5/5** | 0.160 |

### The gate is not overfitting

This was the check that could have invalidated the whole result. Adding the held-out split moved
test performance by <=0.0003 at every severity (0.9912 -> 0.9910 clean; 0.9857 -> 0.9860 at full
corruption), so the earlier fixed-epoch fit was not exploiting its training data.

The stopping epochs chosen per fold were **70, 270, 100, 50, 300** — a fixed 300 epochs was
over-training three of five folds. It happened not to matter here, but that could not have been
known without measuring it.

## 4. The claim

> Replacing entropy-weighted fusion with a 1,089-parameter learned gate reduces the accuracy lost
> to complete failure of an input modality from **7.26pp to 0.50pp** — roughly **15x less
> degradation**. The gate's weight on the surviving branch rises from 0.521 to 0.988 as the band
> degrades, where entropy-based weighting moves only 0.498 to 0.597. At full corruption the gate
> reaches 0.9860 against a 0.9864 fallback ceiling — it recovers **~99% of the available
> headroom**.

## 5. Honest costs and limits

- **It costs 0.23pp on clean data** (0.9910 vs 0.9934), worse on **0/5 folds**, paired-t
  **p=0.045** — a statistically significant regression. This is the standard
  robustness/accuracy trade-off: the gate is trained on a mixture that includes corrupted inputs,
  so it is not optimal when nothing is broken. **Report it as a trade-off, never as a free win.**
  A deployment that will never lose a band should keep plain averaging.
- **The robustness gains are not formally significant at n=5.** +6.52pp at full corruption on 5/5
  folds, but paired-t p=0.160, because entropy fusion's fold variance under corruption is enormous
  (+-0.0868). A 5/5 sign test floors at p=0.0625 regardless. State it as "large and consistent on
  every fold; n=5 limits formal significance", not as p<0.05.
- **Gate training now uses a dedicated split** (`--gate-split-frac 0.3`), so the gate's stopping
  point is chosen on data it never fit. What remains is that the *checkpoint* was selected on the
  same validation set, which makes the model's behaviour there mildly optimistic. Fixing that
  needs retraining with `--calibration-split-frac`, roughly 5 GPU-hours per sweep.
- **Corruption is synthetic.** Gaussian blending of the index channel is a proxy for sensor
  failure, not a real one. A genuinely missing band, cloud occlusion, or a 3-band sensor would each
  behave differently.
- **Single dataset**, as everywhere else in this project.

## 6. Where this leaves the four original contributions

C1's failure is what made this possible: it isolated *why* entropy cannot gate a fusion, which is
what the magnitude features were chosen to fix. The negative result is load-bearing, not
discarded — the paper should present it that way.
