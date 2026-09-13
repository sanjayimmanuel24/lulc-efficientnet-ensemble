# Selective Prediction: the dual-branch model's actual contribution

**Status: POSITIVE, significant, and it survives multiple-comparison correction.**

The accuracy story for this architecture is real but small (+0.70pp over ResNet-50). The
*uncertainty* story is large: the same model **more than halves selective-prediction risk**
relative to a ResNet-50 trained on identical folds, at a third of the parameters.

Source: `scripts/selective_prediction.py`, `results/cv/<tag>/selective_prediction.json`,
`results/figures/risk_coverage_<tag>.png`. Computed from saved per-sample probabilities — no
retraining.

---

## 1. Why this question

An operational land-cover product does not need every tile classified automatically. It needs to
know **which tiles it can trust**. A system that classifies 95% of tiles at 0.13% error and routes
the rest to an analyst is more useful than one that classifies 100% at 0.66% error.

This is standard selective prediction (El-Yaniv & Wiener; Geifman et al.): rank predictions by a
confidence score, accept the top *coverage* fraction, and measure *risk* (error rate) on what is
accepted. AURC is the area under that risk–coverage curve; lower is better.

A dual-branch model is interesting here for a structural reason: it emits **two** uncertainty
signals — the fused confidence, and whether the two branches agree. A single-backbone model has
only the first.

## 2. Results (tag `augmented`, k=5, 27,000 pooled test samples)

Each model uses its **own** predictions and its **own** confidence — the honest deployment
comparison, not a fixed-prediction ablation.

| model | accuracy | AURC | acc@99% | acc@95% | acc@90% | params |
|---|---|---|---|---|---|---|
| RGB branch alone | 0.9864 | 0.00252 | 0.9916 | 0.9971 | 0.9976 | 4.0M |
| Spectral branch alone | 0.9912 | 0.00182 | 0.9949 | 0.9982 | 0.9984 | 4.0M |
| ResNet-50 baseline | 0.9864 | 0.00215 | 0.9905 | 0.9968 | 0.9977 | 25.6M |
| Fixed 50/50 | 0.9933 | 0.00077 | 0.9956 | 0.9987 | 0.9991 | 8.0M |
| **Dual-branch fusion** | **0.9934** | **0.00077** | **0.9961** | **0.9987** | **0.9991** | 8.0M |

Paired per-fold tests (folds are the replication unit, as everywhere else in this project):

| comparison | AURC (ours vs theirs) | ratio | folds won | paired-t | Holm-adj | bootstrap CI |
|---|---|---|---|---|---|---|
| **vs ResNet-50** | 0.00091 vs 0.00213 | **2.35x** | 5/5 | **0.0008** | **0.0032** | [-0.00149, -0.00106] |
| vs RGB branch | 0.00091 vs 0.00282 | 3.10x | 5/5 | 0.0032 | 0.0096 | [-0.00243, -0.00139] |
| vs spectral branch | 0.00091 vs 0.00214 | 2.35x | 5/5 | 0.0066 | 0.0132 | [-0.00164, -0.00082] |
| vs fixed 50/50 | 0.00091 vs 0.00091 | 1.00x | 5/5 | 0.152 | — | [-0.00001, -0.00000] |

All three substantive comparisons win on **every fold**, have bootstrap CIs excluding zero, and
survive Holm-Bonferroni over this four-comparison family.

## 3. The claim

> A compact dual-branch model halves the selective-prediction risk of a ResNet-50 baseline trained
> on identical folds (AURC 0.00091 vs 0.00213, 2.35x, paired-t p=0.0008, Holm-adjusted p=0.0032)
> using 8.0M parameters against 25.6M — 95% automated coverage at 0.13% error versus 0.32%.

Operationally: deferring 5% of tiles cuts the error rate from 0.66% to 0.13% (5x); deferring 10%
cuts it to 0.09%.

This reframes the architecture's purpose. The accuracy gain is secondary; the contribution is that
**decision-level fusion produces a substantially better uncertainty signal than any single
backbone here**, including one three times its size.

## 4. What this does NOT rescue

**C1 adaptive fusion is still exactly tied with the fixed 50/50 average** — AURC 0.00091 both,
ratio 1.00x, paired-t p=0.152, CI [-0.00001, -0.00000]. Selective prediction does not revive
entropy-weighted fusion. The credit belongs to *fusion*, not to *adaptive* fusion. Report it that
way; the C1 negative in `C1_FINDINGS.md` stands unchanged.

## 5. Promising but not yet claimable: the second uncertainty signal

Branch (dis)agreement, measured continuously as `1 - TV(p_rgb, p_spectral)`, is a competitive
rejection signal on its own and the pooled numbers favour combining it with confidence:

| selection score (ranking the same predictions) | AURC (pooled) |
|---|---|
| C1 fusion confidence | 0.00102 |
| branch agreement alone | 0.00074 |
| confidence x agreement | 0.00085 |

But per-fold this does not hold up: agreement-alone wins 3/5 folds (paired-t p=0.285), the
combined score 4/5 (p=0.389). **Underpowered at n=5 — report as a direction, not a result.**
Establishing it properly needs either more folds or a dataset with more branch disagreement than
EuroSAT's 1.4%.

## 6. Caveats to state in the paper

- **`cov@0.2%` is brittle.** A single error near the top of a confidence ranking pins the whole
  low-coverage region (the RGB branch reads 0.004 on tag `default` for exactly this reason). Lead
  with AURC and accuracy-at-coverage; quote coverage-at-risk as illustrative only.
- **This is a new question, not a re-test.** These comparisons are a separate family from the
  accuracy comparisons in `C1_FINDINGS.md` and are Holm-corrected within themselves. Do not pool
  the two families to claim a larger correction was applied.
- **Single dataset.** Same limitation as every other result here; EuroSAT's saturation caps
  branch disagreement at 1.4%, which is precisely what limits Section 5.
