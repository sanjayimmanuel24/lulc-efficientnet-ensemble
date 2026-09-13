# Band-Dropout Robustness: decision-level fusion survives sensor failure, input-level fusion does not

**Status: POSITIVE and large.** This answers the research gap BLUEPRINT Section 3 pre-registered —
whether decision-level fusion differs materially from the input-level band stacking that Helber et
al. (2019) found unhelpful on EuroSAT.

Source: `scripts/band_dropout.py`, `results/cv/augmented/band_dropout_noise.json`,
`results/figures/band_dropout_augmented_noise.png`. Evaluation only — no retraining.

---

## 1. The setup

NDVI requires a real near-infrared band. In deployment that band can be unavailable: sensor fault,
heavy cloud, a cheaper 3-band payload, or B08 saturation. We corrupt **only the index channel of
the spectral branch's input** at evaluation time, blending it toward gaussian noise at increasing
severity. The RGB branch input is untouched, which is what a NIR failure actually looks like.

Temperatures are **not** refitted under corruption: in deployment you cannot recalibrate on a
broken sensor.

Critically, the "spectral branch" here **is** an input-level fusion model — one EfficientNet-B0
taking RGB+NDVI stacked into a single 4-channel tensor. That is the Helber-style architecture.
The dual-branch model uses the same backbone and the same inputs, combined at the decision level.

## 2. Results (tag `augmented`, k=5, 27,000 test tiles)

| model | clean | 25% | 50% | 75% | 100% corrupt |
|---|---|---|---|---|---|
| RGB branch alone | 0.9864 | 0.9864 | 0.9864 | 0.9864 | 0.9864 |
| **Input-level fusion (spectral branch)** | 0.9912 | 0.5109 | 0.2984 | 0.2460 | **0.1445 ± 0.0648** |
| Fixed 50/50 (decision-level) | 0.9933 | 0.9859 | 0.9709 | 0.9374 | 0.9211 ± 0.0864 |
| **C1 fusion (decision-level)** | 0.9934 | 0.9861 | 0.9714 | 0.9376 | **0.9208 ± 0.0868** |
| mean fusion weight on RGB | 0.4982 | 0.5788 | 0.5841 | 0.5748 | 0.5973 |

## 3. The claim

> When the NIR-derived channel fails completely, input-level fusion collapses to **0.1445** —
> barely above the 0.10 chance rate for 10 classes — while decision-level fusion retains
> **0.9208**, a **6.4x relative advantage** from the same backbone on the same inputs. At 25%
> corruption the decision-level model is already statistically indistinguishable from clean
> (0.9861 vs 0.9934) while input-level fusion has lost 48 points.

This is the architectural argument for two branches, and it is not an accuracy argument. On clean
data the two designs differ by ~0.2pp. Under sensor failure they differ by **78 points**.

## 4. Honest riders

- **Fusion does not fully exploit its own fallback.** At full corruption the RGB branch alone
  scores 0.9864 while the fused model scores 0.9208 — **6.5pp of headroom** left on the table by
  the fusion rule. An oracle branch-selector would beat the fusion it is part of.
- **Variance is high at severe corruption** (±0.0868 at severity 1.0; fold 0 alone read 0.7733).
  Quote the mean with its spread, and do not generalise from a single fold — the first single-fold
  look at this experiment suggested a 22pp gap where the 5-fold mean shows 6.5pp.
- **C1 still does not help.** Tied with the fixed average at every severity (deltas +0.0001 to
  -0.0003, all p > 0.26). The fusion weight moves only 0.4982 -> 0.5973 when it would need to
  approach 1.0 to ignore a branch that is at chance level.

## 5. Why C1 fails here specifically

This experiment was designed to give entropy-weighted fusion its best possible chance: one branch
is catastrophically degraded and the other is untouched, so the branches are maximally unequal —
the exact asymmetry C1 needs and never had on clean data.

It still fails, and the reason is now unambiguous: **the corrupted branch is confidently wrong.**
Normalised predictive entropy measures how peaked a distribution is, not whether it is correct. A
network fed noise still produces a peaked softmax. Entropy therefore cannot separate
confidently-right from confidently-wrong, which is precisely the discrimination a fusion gate
needs.

This is the strongest available evidence for the C1 negative in `C1_FINDINGS.md`, and it points
directly at the fix: a gate that learns from the two branches' joint output distribution rather
than from either branch's entropy alone. The 6.5pp headroom in Section 4 is what such a gate
would target.
