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

| method | clean | 25% | 50% | 75% | 100% corrupt | total drop |
|---|---|---|---|---|---|---|
| Fixed 50/50 | 0.9933 | 0.9876 | 0.9664 | 0.9299 | 0.9159 | -7.7pp |
| C1 entropy fusion | 0.9934 | 0.9878 | 0.9659 | 0.9293 | 0.9151 | -7.8pp |
| **Learned gate** | 0.9912 | 0.9896 | **0.9891** | **0.9874** | **0.9857** | **-0.55pp** |
| RGB branch alone (fallback ceiling) | 0.9864 | 0.9864 | 0.9864 | 0.9864 | 0.9864 | 0 |
| **mean gate weight on RGB** | **0.381** | 0.895 | 0.955 | 0.979 | **0.989** | |

Gate minus entropy fusion:

| severity | delta | folds better | paired-t |
|---|---|---|---|
| 0.00 | **-0.0021** | **0/5** | **0.028** |
| 0.25 | +0.0018 | 4/5 | 0.118 |
| 0.50 | +0.0232 | 5/5 | 0.196 |
| 0.75 | +0.0582 | 5/5 | 0.151 |
| 1.00 | **+0.0705** | **5/5** | 0.176 |

## 4. The claim

> Replacing entropy-weighted fusion with a 1,100-parameter learned gate reduces the accuracy lost
> to complete failure of an input modality from **7.8pp to 0.55pp**. The gate's weight on the
> surviving branch rises from 0.381 to 0.989 as the band degrades, where entropy-based weighting
> moves only 0.498 to 0.597. At full corruption the gate reaches 0.9857 against a 0.9864 fallback
> ceiling — it recovers **~99% of the available headroom**.

## 5. Honest costs and limits

- **It costs 0.21pp on clean data** (0.9912 vs 0.9934), worse on **0/5 folds**, paired-t
  **p=0.028** — a statistically significant regression. This is the standard
  robustness/accuracy trade-off: the gate is trained on a mixture that includes corrupted inputs,
  so it is not optimal when nothing is broken. **Report it as a trade-off, never as a free win.**
  A deployment that will never lose a band should keep plain averaging.
- **The robustness gains are not formally significant at n=5.** +7.05pp at full corruption on 5/5
  folds, but paired-t p=0.176, because entropy fusion's fold variance under corruption is enormous
  (+-0.0971). A 5/5 sign test floors at p=0.0625 regardless. State it as "large and consistent on
  every fold; n=5 limits formal significance", not as p<0.05.
- **Validation does quadruple duty** now — early stopping, checkpoint selection, temperature
  fitting and gate training. Use `--calibration-split-frac` and a dedicated gate split before
  making strong claims about the absolute numbers.
- **Corruption is synthetic.** Gaussian blending of the index channel is a proxy for sensor
  failure, not a real one. A genuinely missing band, cloud occlusion, or a 3-band sensor would each
  behave differently.
- **Single dataset**, as everywhere else in this project.

## 6. Where this leaves the four original contributions

C1's failure is what made this possible: it isolated *why* entropy cannot gate a fusion, which is
what the magnitude features were chosen to fix. The negative result is load-bearing, not
discarded — the paper should present it that way.
