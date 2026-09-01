# Research Blueprint: Confidence-Aware Dual-Branch EfficientNet-B0 Ensemble for Satellite LULC Classification

Companion code: `../` (see `README.md` for what has been implemented and tested).

---

## 1. Final Project Title

**"Confidence-Aware Adaptive Fusion of Dual-Branch EfficientNet-B0 for Calibrated, Efficient Land Use/Land Cover Classification from Sentinel-2 Imagery"**

Longer than the original working title on purpose: an IEEE reviewer should be able to guess your three real contributions (adaptive fusion, calibration, efficiency) from the title alone, without opening the PDF. "EfficientNet-B0 Ensemble Framework for LULC Classification" describes a category of paper (there are dozens); this title describes *this* paper.

---

## 2. Motivation

Land Use/Land Cover (LULC) classification from Sentinel-2 imagery supports urban planning, agricultural monitoring, and climate policy, and increasingly needs to run on resource-constrained platforms (edge devices on drones, low-power ground stations, national mapping agencies without large GPU budgets). EfficientNet-B0 is an attractive backbone for this setting precisely because it is small (~5.3M parameters), but a single EfficientNet-B0 branch trained on RGB alone discards the spectral information (near-infrared, shortwave-infrared) that is most diagnostic for vegetation, water, and built-up surfaces — information a human photo-interpreter would never discard, and information Sentinel-2 imagery actually provides for free if the pipeline is built to use it.

At the same time, EuroSAT — the standard benchmark for this line of work — is close to saturated: baseline CNNs already reach 97-99% test accuracy, which means a paper whose only claim is "our model gets +0.4% accuracy" is not a strong submission regardless of the architecture behind it. The more defensible engineering and scientific problem left on the table is not squeezing another fraction of a percent out of a near-solved benchmark, but building a system whose predictions can be *trusted* (calibrated confidence, explainable per-modality reasoning) at *no meaningful efficiency cost* — which is a genuinely open problem in this literature, and the one this project targets.

---

## 3. Research Gap

Three specific, checkable gaps in the current EuroSAT/RESISC45 LULC literature motivate this project:

1. **Ensemble fusion is almost always static.** Multi-branch remote sensing papers overwhelmingly use fixed-weight averaging or concatenation, decided once at design time (or learned once, globally, as a single scalar) rather than adapting per-sample to which branch is actually more trustworthy for *that specific image*. A cloud-edge patch and a clear rural patch do not deserve the same RGB/spectral trust ratio.
2. **Confidence calibration is essentially absent from this literature.** Papers report accuracy/F1/kappa; almost none report Expected Calibration Error, reliability diagrams, or Brier score for a LULC model, despite these models increasingly feeding into downstream decision systems (e.g., automated flagging for deforestation alerts) where a wrong answer delivered with 99% confidence is a different failure mode from the same wrong answer delivered with 51% confidence.
3. **"Efficient" architectures are rarely evaluated on efficiency.** Papers using EfficientNet or MobileNet backbones typically report parameter count once in a table and otherwise evaluate exactly like a ResNet paper would — accuracy tables only. There is a gap for a paper that treats the accuracy-efficiency-calibration trade-off as the actual object of study, with FLOPs/latency/energy reported at the same level of rigor as accuracy.
4. **Prior band-fusion evidence on EuroSAT is at the input level only, and it is a negative result.** Helber et al. (2019), the paper that introduced EuroSAT, tested RGB vs. color-infrared vs. SWIR band combinations by stacking them as extra input channels into a single CNN — and found RGB alone (98.57%, ResNet-50) *beat* every spectral-band combination they tried, including color-infrared (98.30%) and SWIR (97.05%). No prior EuroSAT paper has re-tested whether spectral information helps when combined at the **decision level** (separate branches, fused after each produces its own prediction) rather than the **input level** (channels concatenated before the first convolution) — a materially different architectural question that this negative result does not settle.

None of these four gaps require a fundamentally new architecture to address — which is exactly what keeps this project's scope realistic for a 3-4 month timeline while still being genuinely novel in combination.

---

## 4. Proposed Novel Framework

> **STATUS AFTER EXPERIMENTS (updated post-hoc).** The C1-C4 statements below are the
> *pre-registered hypotheses*, kept verbatim as the record of what was predicted before any
> model was trained. Each now carries a VERDICT line with the measured outcome. Full evidence:
> `docs/C1_FINDINGS.md`. Protocol: k=5 stratified CV on identical folds, six sweeps, a ResNet-50
> baseline on the same folds, Holm-Bonferroni over a 20-comparison family, and a seed noise floor
> measured at 0.0007 from three seeds of an identical configuration.
>
> **Two of the four hypotheses failed, one is unsupported, one holds strongly.** What the work
> actually contributes is stated in Section 14, which has been rewritten accordingly. The failures
> are reported rather than removed: Sections 9 and 18 committed to that in advance.

Four contributions, deliberately kept to four, chosen so each one is individually defensible **and** so they reinforce rather than merely coexist with each other (a property reviewers notice, and one that's absent from a "kitchen sink" list of 15 unrelated tricks).

### C1 — Confidence-Aware Adaptive Fusion
Per-sample fusion weights derived from each branch's own predictive entropy, rather than a fixed or globally-learned scalar. Formal definition in Section 6. Critically, this rule *provably reduces to plain averaging* when both branches are equally confident (tested in `tests/test_fusion_logic.py::test_equal_confidence_reduces_to_simple_average`), which means the "fixed-weight ensemble" baseline is not a separate model to implement — it is this same mechanism with one term ablated, giving a clean, apples-to-apples ablation instead of a reimplementation.

> **VERDICT: NOT SUPPORTED.** No accuracy gain over the fixed 50/50 average it generalises:
> -0.0001 (default, McNemar p=0.45, Holm-adjusted p=0.906) and -0.0002 (nonrgb, 0 samples right
> where the average was wrong). Ten folds, two architectures, never ahead. **Mechanism**: mean
> fusion weight is 0.4995 -- the rule sits at the equal-confidence point where Section 6 proves it
> *is* plain averaging. Normalised entropy does not separate a correct branch from an incorrect one
> when both are near-saturated. Tested against the saturation defence with a 10%-data ablation:
> disagreements doubled, weights did not move (0.4991), effect still zero. C1's surviving claim is
> calibration only: best ECE (0.0040 equal-mass) and Brier (0.0141) of all variants.

### C2 — Genuine Multispectral Fusion (RGB + NDVI branch)
The spectral branch is trained on true Sentinel-2 reflectance bands (Red=B04, NIR=B08 at minimum) via the EuroSAT *multispectral* release, not an approximation of NDVI computed from an RGB JPEG (which is not physically meaningful — RGB has no near-infrared channel). This sounds like a minor implementation detail; it is in fact a common, reviewer-visible mistake in student remote-sensing projects, and getting it right is itself worth stating explicitly in the paper's data section.

This contribution carries a genuine, stated risk, not a guaranteed win: Helber et al. (2019) — the EuroSAT paper itself — found that stacking spectral bands as *extra input channels into one CNN* did not help on this dataset (RGB alone beat every band combination they tested). C2 deliberately tests a different architectural question: whether spectral information helps when each modality gets its **own** backbone, specializing separately, combined only at the decision level via C1's adaptive fusion. The hypothesis is that naive channel-stacking forces a single set of early filters to jointly process RGB texture and spectral signal (mutual interference), whereas two specialized branches plus adaptive fusion can fall back to RGB when the spectral branch has nothing useful to add for a given sample — which prior input-level-only evidence cannot rule out either way. If the ablation (Section 10, row 2 vs. row 1) shows spectral fusion does *not* help on EuroSAT even at the decision level, that is reported as a finding, not hidden — see Section 9's dataset discussion and Section 18, critique #7.

> **VERDICT: NOT SUPPORTED as stated.** Across k=5 the spectral branch beats the RGB branch by
> +0.0010, not significant (McNemar p=0.195, below the 0.0007 seed floor). The ~3pp advantage seen
> on the first single split was an artifact of one unusually weak RGB branch -- **do not cite 3pp**.
> Worse for the hypothesis, per-class analysis shows the effect is *backwards*: the spectral branch
> is weaker than RGB on the vegetation classes NDVI targets (mean dF1 -0.0010) and stronger on
> non-vegetation (+0.0021), with its largest gain on SeaLake (+0.0101, water) and largest loss on
> Pasture (-0.0106). What gain exists points at NDWI/NDBI territory, not NDVI.

### C3 — Efficient Channel Attention (ECA) at the Fusion Point
Chosen over SE and CBAM after an explicit cost comparison (Section 13) because it adds a per-module parameter count in the single digits (a k-length 1D convolution, k≈3-5) versus the hundreds-to-thousands added by an SE bottleneck MLP, let alone CBAM's additional spatial-attention branch. This is the one attention mechanism that survives a genuine "efficiency-first" filter rather than being included because attention modules are fashionable.

> **VERDICT: NOT SUPPORTED; leans negative.** Removing ECA (`--attention none`, k=5) was *better*:
> +0.0019 on the RGB branch (Holm-adjusted p=0.0205) and +0.0010 fused (p_adj 0.254). Direction was
> consistent but every effect is at or below the seed floor and the comparison was single-seed, so
> the honest statement is "no evidence ECA helps, and some evidence it mildly hurts". The *cost*
> claim is verified exactly: ECA is **10 parameters total** (1.24e-06 of the model) with latency
> identical to three significant figures.

### C4 — Calibration-Coupled Evaluation
Temperature scaling fit post-hoc on a held-out validation split, evaluated with ECE/MCE/Brier score alongside standard accuracy metrics. This is not a bolt-on: because C1's fusion weights are themselves derived from confidence, a miscalibrated branch would bias the fusion mechanism itself, not just the reported confidence number. Reporting calibration is therefore *validating a component of the proposed method*, not merely padding the results section — this connection is worth stating explicitly in the paper, since it's what turns "we also measured ECE" from a checkbox into an argument.

> **VERDICT: SUPPORTED -- the strongest of the four.** Temperature scaling (fitted T = 0.58-0.61)
> cuts pooled ECE from **0.0951 to 0.0043, a 22x reduction**, orders of magnitude above any noise
> floor. The direction is the opposite of the usual assumption: label smoothing at 0.1 leaves the
> model severely *under*-confident (mean confidence 0.8967 against ~99% accuracy), and calibration
> corrects that. The coupling argument is weakened by C1's failure -- confidence turned out not to
> drive fusion in practice -- but calibration stands on its own as a reporting contribution.
> Caveat: equal-width ECE is unreliable at this accuracy; report equal-mass (adaptive) ECE.

**What ties these together:** C1 needs well-calibrated per-branch confidence to produce sensible weights (motivating C4); C2 gives the two branches a genuine reason to disagree in informative, class-dependent ways (motivating why adaptive fusion, C1, should beat static fusion at all — if the branches always agreed, adaptive weighting would have nothing to do); C3 is the connective tissue that lets C1/C2 be added without breaking the paper's efficiency framing. Four contributions, one coherent story.

---

## 5. System Architecture

```
                         ┌─────────────────────────────┐
                         │        Input Sentinel-2      │
                         │      patch (multispectral)   │
                         └───────────────┬──────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                          │
             RGB channels (B04,B03,B02)          RGB + spectral index channel(s)
                     │                                (e.g. + NDVI from B08,B04)
                     ▼                                          ▼
        ┌────────────────────────┐              ┌────────────────────────────┐
        │  Branch A: RGB         │              │  Branch B: Spectral        │
        │  EfficientNet-B0       │              │  EfficientNet-B0           │
        │  (ImageNet-pretrained, │              │  (stem conv adapted for    │
        │   in_chans=3)          │              │   3+I input channels)      │
        │        │               │              │        │                   │
        │  feature map (C,H',W') │              │  feature map (C,H',W')     │
        │        │               │              │        │                   │
        │   ECA channel attn.    │              │   ECA channel attn.        │
        │        │               │              │        │                   │
        │  global average pool   │              │  global average pool       │
        │        │               │              │        │                   │
        │  Linear classifier     │              │  Linear classifier         │
        │        │               │              │        │                   │
        │  logits_rgb (K,)       │              │  logits_spectral (K,)      │
        └────────────┬───────────┘              └────────────┬───────────────┘
                     │                                        │
                softmax                                   softmax
                     │                                        │
                probs_rgb                                probs_spectral
                     │                                        │
                     └───────────────┬────────────────────────┘
                                      ▼
                     ┌──────────────────────────────────┐
                     │  Confidence-Aware Adaptive Fusion │
                     │  (entropy-based per-sample        │
                     │   weights w_rgb, w_spectral)      │
                     └───────────────┬────────────────────┘
                                      ▼
                          fused_probs (K,)  →  argmax → predicted class
                                      │
                     ┌────────────────┴─────────────────┐
                     ▼                                    ▼
        Temperature-scaled calibration         Grad-CAM / Grad-CAM++ per branch
        (fit post-hoc on validation split)     (explainability, Section 12)
```

Both branches are supervised directly on their own logits during training (not only through the fused output) — see `src/training/train.py::compute_loss` docstring for why: a branch that is never directly supervised can "hide" behind the other branch, which would make per-branch ablations (Section 10) meaningless.

---

## 6. Mathematical Formulation

**Predictive entropy** of a branch's softmax output `p ∈ R^K`:

```
H(p) = - Σ_k  p_k · log(p_k)                (nats)
```

**Entropy-normalized confidence** (bounded to [0, 1]):

```
c = 1 - H(p) / log(K)
```

**Confidence-aware fusion weights** for branches A (RGB) and B (spectral):

```
w_A = c_A / (c_A + c_B + ε)
w_B = c_B / (c_A + c_B + ε)
p_fused = w_A · p_A + w_B · p_B
```

Property (proved by construction, tested in `test_equal_confidence_reduces_to_simple_average`): if `c_A = c_B`, then `w_A = w_B = 0.5` and `p_fused` is exactly the naive-average baseline. This is what makes "fixed 50/50 ensemble" a special case of the proposed method rather than a separate baseline implementation.

**Label smoothing cross-entropy**, smoothing factor `ε_ls`, true class `k*`:

```
y_smooth[k] = 1 - ε_ls + ε_ls/K     if k = k*
y_smooth[k] = ε_ls / K              otherwise

L = - Σ_k  y_smooth[k] · log_softmax(logits)[k]
```

**Temperature scaling**: fit scalar `T > 0` on validation logits `z` and labels `y` by minimizing NLL:

```
T* = argmin_T   - (1/N) Σ_i  log softmax(z_i / T)[y_i]
```

At inference, calibrated probabilities are `softmax(z / T*)`; argmax predictions are unchanged (T only rescales confidence, not rank order).

**Expected Calibration Error** (M equal-width confidence bins):

```
ECE = Σ_{m=1}^{M}  (|B_m| / N) · | acc(B_m) - conf(B_m) |
```

**Spectral indices** (Sentinel-2 surface reflectance, ρ):

```
NDVI = (ρ_NIR - ρ_Red) / (ρ_NIR + ρ_Red)
NDWI = (ρ_Green - ρ_NIR) / (ρ_Green + ρ_NIR)
NDBI = (ρ_SWIR - ρ_NIR) / (ρ_SWIR + ρ_NIR)
```

All five formulas above are implemented and independently unit-tested against hand-computed values in `tests/` (43 passing tests in this sandbox; see README for what still needs a torch-equipped environment).

---

## 7. Algorithm

```
Algorithm: Confidence-Aware Dual-Branch Training and Inference

Input: Sentinel-2 patches with 13-band GeoTIFF, labels y, config cfg
Output: trained DualBranchEfficientNet, fitted temperature T*

1. Split data into train / val / test (stratified by class)
2. For each progressive-resizing stage s in {128, 160, 224}:
     a. Build train/val dataloaders at resolution s
     b. For each epoch:
          - forward pass: logits_rgb, logits_spectral = model(rgb, rgb+spectral)
          - loss = CE_smoothed(logits_rgb, y) + CE_smoothed(logits_spectral, y)
          - backward + AdamW step (cosine LR schedule, warmup)
          - evaluate on val split using fused_probs.argmax() for accuracy
          - checkpoint best-val-accuracy weights; early-stop on patience
3. Load best checkpoint. Fit T* on validation logits (per branch) by minimizing NLL
4. On test split:
     a. Compute fused_probs using confidence-aware fusion (Section 6)
     b. Compute accuracy, macro-F1, weighted-F1, Cohen's kappa, ROC-AUC
     c. Compute ECE, MCE, Brier score (pre- and post-calibration)
     d. Run Grad-CAM/Grad-CAM++ on representative correct/incorrect samples per class
5. Repeat steps 1-4 across k=5 stratified folds; report mean ± std and
   bootstrap CIs; run paired significance tests against each baseline (Section 11)
```

---

## 8. Training Strategy

| Component | Choice | Justification |
|---|---|---|
| Optimizer | AdamW | Standard for CNN fine-tuning; decoupled weight decay avoids the L2-through-Adam-momentum interaction that plain Adam+L2 has. |
| LR schedule | Linear warmup (2 epochs) → cosine annealing | Warmup avoids destabilizing ImageNet-pretrained weights with a large initial LR; cosine avoids needing to hand-tune decay milestones. |
| Batch size | 64 | Fits comfortably on a single mid-range GPU (e.g. 8-12GB) at 224×224 for EfficientNet-B0; large enough for stable batch statistics. |
| Mixed precision | Enabled on CUDA | ~1.5-2x speedup on modern GPUs at negligible accuracy cost for this model size; automatically disabled on CPU-only runs. |
| Progressive resizing | 128 → 160 → 224 | See `src/training/train.py` docstring: speeds up early epochs, acts as a mild regularizer, and is training-time-only (zero inference cost), unlike most of the rejected alternatives in Section 15. |
| Loss | Label-smoothed cross-entropy (ε=0.1), per branch | Reduces overconfidence on a near-saturated benchmark where the risk of the model memorizing training-set idiosyncrasies is real; directly supports the calibration story (C4). |
| Regularization | Weight decay 1e-4, gradient clipping (norm 5.0), early stopping (patience 5) | Standard, cheap, and each is individually ablatable. |
| Transfer learning | ImageNet-pretrained EfficientNet-B0 for both branches; spectral branch's stem conv reinitialized/adapted for extra input channels | Sentinel-2 patches share low-level texture statistics with natural images despite the domain gap; the extra spectral channel(s) need a modified (not pretrained) stem, which is standard practice (e.g. as done when adapting ImageNet backbones to 4-channel RGBA or multispectral input). |

---

## 9. Experimental Design

### Datasets
- **Primary: EuroSAT (multispectral, 13-band).** Required for genuine NDVI; RGB-only EuroSAT cannot support the spectral branch as specified.
- **Secondary: RESISC45.** Harder benchmark (45 classes, more visual diversity), used for cross-dataset generalization (train primarily on EuroSAT, evaluate zero-shot and fine-tuned on RESISC45) — this is where genuine accuracy headroom still exists, unlike EuroSAT.
- **Optional: UC Merced.** Smallest dataset (2,100 images); useful only for a small-data-regime robustness check, not as a primary evaluation target. Include only if time permits after the primary EuroSAT + RESISC45 experiments are complete — do not let it displace them.
- **Additional datasets are not necessary.** Three well-established, differently-sized benchmarks are enough to support every claim in this framework (accuracy, generalization, small-data robustness); a fourth dataset would add data-wrangling time without adding a new claim.

### Input Data
**Recommendation: RGB + NDVI (not RGB + NDWI/NDBI simultaneously, and not all four indices stacked).** NDVI is the most broadly diagnostic index for the EuroSAT/RESISC45 class taxonomies (vegetation vs. non-vegetation is the single largest source of visual variance across LULC classes — forests, pastures, and crops are separated primarily by vegetation vigor and phenology, which NDVI encodes directly). NDWI and NDBI are tested as an ablation (Section 10: "different spectral indices") rather than included by default, because stacking all four indices simultaneously without individually ablating each one is exactly the kind of change a reviewer will ask you to isolate — better to isolate it yourself, in the paper, than have a reviewer ask for it in revision.

**Important caveat, and the actual reason EuroSAT + RESISC45 are both in scope rather than EuroSAT alone:** Helber et al. (2019) found that RGB alone beats every spectral-band combination they tried when bands are simply stacked as input channels into a single CNN (Section 3, gap #4). C2's bet is that *decision-level* fusion (separate branches, per Section 4) succeeds where their *input-level* stacking didn't — but this is a hypothesis to test, not an assumption to build the whole paper on. Concretely: run Ablation #1 vs. #2 (Section 10) on EuroSAT first. If spectral fusion shows negligible or no improvement there, that is expected given prior evidence and should be reported honestly, with RESISC45 (harder, more classes, more genuine headroom) as the dataset where the same ablation is repeated to test whether the effect appears once there is more room for it to show up. A paper that says "decision-level spectral fusion doesn't move the needle on a near-saturated benchmark but does on a harder one" is a more specific, more credible contribution than a paper that only reports a marginal EuroSAT gain and hopes no reviewer checks Helber et al.'s Table IV.

### Baseline Models
| Model | Include? | Why |
|---|---|---|
| ResNet-50 | Yes | Standard, widely reported baseline; needed for direct comparability with prior EuroSAT/RESISC45 papers. |
| ResNet-101 | No | Redundant with ResNet-50 for this comparison's purpose; adds a row without adding a distinct comparison point. Mention only if a reviewer specifically asks in revision. |
| DenseNet-121 | Yes | Common efficient-ish CNN baseline in remote sensing literature; good mid-size comparison point. |
| MobileNetV3 | Yes | The most relevant *efficiency* baseline — directly tests whether the proposed framework's efficiency claims hold up against another lightweight architecture, not only against large models. |
| EfficientNet-B0 (single branch, RGB only) | Yes — this is also Ablation #1 | The single most important baseline: isolates exactly what the second branch + fusion + attention are buying you. |
| EfficientNet-B1 | No, as a baseline | Would confuse the story ("is B1 the competitor or B0 the target?"); more useful as a footnote showing the dual-branch B0 ensemble is competitive with a larger single-branch model at lower combined cost — mention in the discussion, not the main results table. |
| Vision Transformer (ViT-Tiny) | Yes | Necessary modern-architecture comparison; ViTs are known to need more data/augmentation to match CNNs at this image size and dataset size, which is itself worth showing rather than assuming. |
| ConvNeXt-Tiny | Yes | Strong modern CNN baseline; also serves double duty as the literature's typical "large distillation teacher" (Section 15 explains why distillation itself was cut from the core paper). |

---

## 10. Ablation Studies

| # | Ablation | What it isolates / proves |
|---|---|---|
| 1 | RGB branch only | Baseline single-branch EfficientNet-B0 performance; the reference every other row is measured against. |
| 2 | RGB + NDVI, fixed 50/50 fusion (no adaptive weighting) | Isolates the benefit of spectral fusion *alone*, before adding C1. |
| 3 | RGB + NDVI, confidence-aware adaptive fusion (proposed) | Isolates the marginal benefit of C1 specifically, holding everything else fixed vs. row 2. |
| 4 | With ECA vs. without attention vs. SE vs. CBAM (4 sub-rows) | Justifies the specific attention choice with a real accuracy/parameter/FLOPs trade-off table, not just an assertion. |
| 5 | With vs. without label smoothing | Isolates the loss-function change; also shows its effect on calibration (ECE), not just accuracy — ties into C4. |
| 6 | With vs. without progressive resizing | Confirms the training-time-only claim: final accuracy should be comparable, training wall-clock time should differ. |
| 7 | With vs. without temperature scaling | Shows ECE/MCE/Brier before and after calibration; accuracy must be identical (by construction) — a useful sanity check that calibration is doing what it claims. |
| 8 | Different spectral indices: NDVI vs. NDWI vs. NDBI vs. NDVI+NDWI stacked | Justifies the "NDVI alone" default choice in Section 9 with actual numbers rather than only domain reasoning. |
| 9 | Different fusion strategies: concatenation vs. fixed-weight average vs. confidence-aware (proposed) vs. a small learned-gating MLP | Positions the proposed fusion rule against the two most obvious alternatives a reviewer would ask about. |
| 10 | k-fold variance of every row above | Without this, single-run ablation numbers invite the reviewer question "is that difference even real?" — Section 11 covers exactly this. |

Ablations intentionally **not** run: augmentation-pipeline ablation (MixUp/CutMix/RandAugment) is scoped down to at most one comparison (default pipeline vs. + one technique), not a full factorial sweep, per the timeline discussion in Section 16 — an exhaustive augmentation sweep is a paper's worth of work on its own and would dilute focus from C1-C4.

---

## 11. Statistical Validation

| Test | When to use it here |
|---|---|
| **k-fold cross-validation (k=5, stratified)** | Primary source of variance estimates for every headline number; report mean ± std, not a single train/test split. |
| **Paired t-test** | Comparing two models' per-fold accuracy/F1 when the *fold-level* differences are approximately normal (check with a quick normality plot before relying on this one). |
| **Wilcoxon signed-rank test** | Preferred over the paired t-test with only k=5 folds, since it doesn't assume normality of the differences — use this as the primary test, the t-test as a secondary corroborating check. |
| **McNemar's test** | For comparing two models on the *same* test set at the individual-prediction level (not fold means) — e.g. "does the proposed model disagree with the ResNet-50 baseline asymmetrically (more often right where ResNet is wrong, than vice versa)?" Implemented with the exact binomial test rather than the chi-square approximation, since discordant-pair counts on a modest test set are often small enough that the approximation is unreliable. |
| **Bootstrap confidence intervals** | Distribution-free interval for any single metric (e.g. macro-F1) on the test set, useful alongside k-fold std since it doesn't require retraining. |
| **Effect size (Cohen's d, paired)** | Report alongside every p-value — with k=5 folds, statistical significance and practical significance can diverge, and a reviewer will want to see both. |

All five of the above are implemented in `src/evaluation/statistical_tests.py` and independently unit-tested against synthetic cases with known ground truth (e.g., a McNemar test that must detect a constructed asymmetric-disagreement pattern, a bootstrap CI that must narrow as input variance decreases).

---

## 12. Explainability

| Technique | Included? | Why |
|---|---|---|
| Grad-CAM | Yes (primary) | Cheap (single backward pass), well-understood, sufficient localization quality for scene-level (not object-level) classification. |
| Grad-CAM++ | Yes (secondary, for failure-case figures) | Better handles cases with multiple/small discriminative regions (e.g. small water bodies within an agricultural patch) at negligible extra cost since it reuses the same hooks as Grad-CAM. |
| Eigen-CAM | No | Doesn't use gradients or class information at all (it's the principal component of activations), so its maps are not class-discriminative — a reviewer-facing weakness for a classification paper specifically. Could be mentioned as a robustness cross-check in an appendix, not the main explainability section. |
| Score-CAM | No | Requires a forward pass per channel-masked image (hundreds of extra forward passes per single explanation) — directly contradicts the paper's efficiency framing if used as a core method; mention only as future work if compute budget allows. |
| Layer-CAM | No, for scope reasons | Legitimate technique, but adding a fourth CAM variant with no comparison purpose beyond "we tried it too" reads as padding; two well-chosen CAM variants (Grad-CAM, Grad-CAM++) that are individually justified are stronger than four with no individual justification. |
| Per-branch (not only fused) visualization | Yes — this is the actual novel piece of the explainability section | Running Grad-CAM separately on the RGB branch and the spectral branch, then showing them side by side for the same image, directly demonstrates *what the spectral branch is adding* — e.g. confirming it attends to vegetation-boundary pixels the RGB branch's heatmap misses. This is a much stronger explainability argument than a single fused heatmap, and costs nothing extra since both branches already have their own forward/backward pass. |
| Confidence/weight visualization | Yes | Plotting the per-sample fusion weights (w_rgb, w_spectral) against class or against a per-sample difficulty proxy directly visualizes C1 in action — this is the single most paper-specific figure to include, since it doesn't exist in any static-fusion baseline. |
| Reliability diagrams | Yes (ties to C4) | Standard companion to ECE reporting; shows calibration before/after temperature scaling. |
| Failure case gallery | Yes | A dedicated figure of 8-12 misclassified examples with their Grad-CAM overlays and predicted-vs-true class, directly feeding the Error Analysis discussion below. |

### Error Analysis (feeds into the paper's discussion section)
Expect and report on: confusion between visually similar classes (e.g. Pasture vs. Permanent Crop, or Highway vs. Industrial in RESISC45), seasonal ambiguity (a fallow field vs. bare soil class), cloud/haze occlusion effects, mixed land-cover boundary patches (a single 64×64 or 256×256 patch straddling two classes), and small-object limitations (a single building in an otherwise agricultural patch). For each category, report whether the spectral branch specifically helps or hurts relative to RGB-only (this is a direct, cheap way to make the error analysis section feel specific to this method rather than generic).

---

## 13. Computational Complexity Analysis

Per-branch EfficientNet-B0: ~5.3M parameters, ~0.39 GFLOPs at 224×224 (standard published figures for the backbone alone). Two branches roughly doubles this before accounting for the attention/classifier heads, which is why every added module's cost must be justified individually rather than assumed acceptable:

| Component | Added parameters | Added FLOPs | Verdict |
|---|---|---|---|
| Second EfficientNet-B0 branch | ~5.3M | ~0.39 GFLOPs | Unavoidable cost of the core idea (C2); this is the real budget line, everything else must stay small relative to it. |
| ECA attention (×2 branches) | single digits × 2 (a k≈3-5 length 1D conv per branch) | negligible (<0.1% of backbone FLOPs) | Kept — this is the entire point of choosing ECA. |
| SE attention (×2 branches, for comparison only) | hundreds-thousands × 2 (depends on reduction ratio r) | small but non-negligible vs. ECA | Ablation-only, not in the proposed model. |
| CBAM (×2 branches, for comparison only) | SE's channel cost + a 7×7 spatial-attention conv | roughly 1.5-2× SE's added FLOPs | Ablation-only, not in the proposed model. |
| Confidence-aware fusion | 0 (no learned parameters — entropy is computed directly from softmax outputs) | negligible (a handful of elementwise ops per sample) | Free relative to a learned gating MLP; this is a deliberate design choice, not an oversight. |
| Temperature scaling | 1 scalar | negligible | Fit once, post-hoc; effectively free at both train and inference time. |

Full experimental section reports, per model in the comparison table: parameter count, GFLOPs, model size (MB), GPU inference latency, CPU inference latency, throughput (images/sec), and peak GPU memory during training — not accuracy alone. This table is what turns "efficient" from an adjective in the title into a measured claim.

---

## 14. Contributions (rewritten after experiments)

The original "Expected Contributions" list is preserved at the end of this section as the
pre-registered prediction. What the work actually establishes, with the evidence, is:

**1. A compact dual-branch fusion model that outperforms a substantially larger standard baseline
on matched folds.** 0.9930 +- 0.0017 accuracy (k=5, three seeds, 15 folds) against a ResNet-50
trained on the *same* folds under the *same* protocol: **+0.0054 to +0.0070, Holm-adjusted
p ~ 1e-25**, at **8.0M parameters versus 25.6M**, running on **CPU at 24.6 images/sec** with no
GPU. The controlled comparison matters: our ResNet-50 reaches 0.9864 on our folds, within 0.1pp of
Helber et al.'s published 98.57%, so the pipeline reproduces the literature baseline before
beating it.

**2. Evidence that the gain comes from decision-level fusion, not from the backbone.** A single
EfficientNet-B0 branch is *statistically indistinguishable* from ResNet-50 (+0.0013, p=0.11).
Fusion beats the best single branch by +0.0031 (p_adj ~ 8e-9). On the 1.43% of samples where the
branches disagree, RGB alone is right 44.7% of the time and the spectral branch 51.4%, while
fusion reaches 73.1% -- **76% of the oracle ceiling**. Fusion also roughly **halves run-to-run
variance** (fold sd 0.0029-0.0046 single-branch vs 0.0010-0.0023 fused), a reliability claim
independent of accuracy.

**3. A reported negative result on entropy-weighted adaptive fusion, with its mechanism.** C1
gives no accuracy gain over the fixed 50/50 average it generalises, across ten folds and two
branch configurations, and the reason is measurable rather than speculative: mean fusion weight
0.4995, i.e. the rule operating at the point where it is mathematically identical to its own
ablation. This is a genuine contribution -- entropy-weighted fusion is a widely reused idea and
its failure mode on saturated branches is not documented elsewhere.

**4. A statistical protocol that makes small effects interpretable, including a measured noise
floor.** k=5 CV on shared folds, paired McNemar (exact binomial) plus paired-t, bootstrap CIs,
Holm-Bonferroni over the full 20-comparison family, and -- the piece most often missing -- a
**seed noise floor of 0.0007** obtained by re-running an identical configuration at three seeds.
That floor disqualifies four candidate effects (C1, C2, C3, disjoint branch inputs) that would
otherwise have looked publishable at raw p < 0.05, and it is why the one surviving small effect
(augmentation, +0.0012) was seed-matched before being claimed.

### What was predicted (pre-registration, kept for the record)

1. A confidence-aware adaptive fusion rule that is provably a strict generalization of
   fixed-weight averaging -- **the generalization holds mathematically; the empirical benefit did
   not materialise.**
2. An efficiency-first attention comparison (ECA vs SE vs CBAM) -- **only ECA vs none was run;
   ECA showed no benefit. SE/CBAM remain untested.**
3. A calibration-coupled evaluation protocol -- **delivered; the strongest surviving contribution
   (22x ECE reduction), though the coupling to fusion weakened when C1 failed.**
4. A reproducible, statistically rigorous benchmark on EuroSAT-MS + RESISC45 -- **delivered for
   EuroSAT-MS and exceeded (seed floor, Holm correction, matched baseline). RESISC45 was dropped:
   it is RGB-only, so NDVI cannot be computed and both branches would collapse to RGB. A genuine
   multispectral cross-dataset test needs BigEarthNet or So2Sat.**

---

## 15. Comparison with Existing Literature & Rejected Alternatives

The brainstormed component list in the original project prompt is evaluated here individually — every rejection is a reviewer question pre-empted, not a shortcut taken.

| Candidate component | Verdict | Reasoning |
|---|---|---|
| Knowledge distillation from EfficientNet-B3/ConvNeXt/Swin teacher | **Rejected from core paper; future work** | Adds a full extra training stage (train or acquire a strong teacher, tune distillation temperature/loss weight) for marginal gain on an already ~98%+ saturated EuroSAT benchmark; disproportionate to a 3-4 month timeline. Legitimate direction on RESISC45 specifically, where headroom exists — listed under Future Extensions (Section 17), not claimed here. |
| ArcFace loss | **Rejected** | An open-set/verification-margin loss designed for face recognition; no established motivation for closed-set multi-class scene classification. Including it without strong justification is a common reviewer red flag ("why this loss for this task?"). |
| MixUp / CutMix / Mosaic, all simultaneously | **Rejected as a combined default; one included in a single ablation row instead** | Stacking several augmentations without individually ablating each invites "which one actually helped?" — isolate at most one (CutMix) as an ablation, not a silent default. |
| Self-supervised / contrastive pretraining | **Rejected from core paper; future work** | Needs a large unlabeled pretraining corpus and a long pretraining schedule — realistically a separate project on its own timeline. |
| Domain adaptation / semi-supervised learning | **Rejected from core paper; future work** | Same scope concern as above; the cross-dataset generalization experiment (Section 9) already tests transferability without requiring a full domain-adaptation method. |
| Meta-learner / stacking on top of the ensemble | **Rejected** | A second trainable stage needs careful nested cross-validation to avoid leakage, for a gain the parameter-free entropy-weighted fusion (C1) already targets without any extra training. |
| Full Bayesian deep ensembles | **Rejected as primary method; MC Dropout included as a secondary, optional uncertainty analysis** | Requires training multiple independent models beyond the two branches — expensive relative to benefit; MC Dropout gives a cheap secondary uncertainty signal without that cost, but is not load-bearing for any headline claim. |
| CBAM / SE / Coordinate Attention | **Included only as ablation comparisons, not the proposed mechanism** | See Section 13's cost table — each adds meaningfully more parameters/FLOPs than ECA for this task. |
| ScoreCAM / EigenCAM / LayerCAM as primary explainability | **Rejected as primary; Grad-CAM/Grad-CAM++ used instead** | See Section 12. |
| Multi-resolution patches / progressive resizing | **Kept** | Training-time only, zero inference cost, well-established practice (used in the original EfficientNet training recipe itself). |
| Vision Transformer / ConvNeXt as ensemble branches | **Rejected as branches; kept as comparison baselines** | Adding a transformer branch would undermine the "efficient CNN ensemble" framing; comparing against them as baselines (Section 9) is the right role for them in this paper. |
| Assuming spectral bands help on EuroSAT without qualification | **Rejected — addressed directly, not assumed** | Helber, Bischke, Dengel & Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification" (IEEE JSTARS, 2019) — the paper introducing this exact dataset — reports RGB-only input (98.57%, ResNet-50) outperforming every color-infrared and SWIR band-stacking combination they tested. This is the single most important piece of related work to cite and engage with directly (Sections 3, 4, 9), since it is a specific, checkable prior result that appears to cut against C2 at first glance. The paper's own experiment was input-level channel-stacking on one backbone, not decision-level dual-branch fusion — a distinction this project's Ablation #1 vs. #2 (Section 10) is designed to actually test, on both EuroSAT and RESISC45, rather than argue around. |

**Relative to prior EuroSAT/RESISC45 papers** (which this project positions itself against without needing a literature-review table reproduced here — build that table from the actual papers you cite once you begin writing, using this blueprint's contributions as the comparison axis): most prior dual-branch or multi-index fusion papers on these datasets use fixed-weight or concatenation fusion and report accuracy/F1 only, without calibration metrics or paired significance testing. The combination of (adaptive, provably-baseline-generalizing fusion) + (calibration reported and methodologically connected to the fusion mechanism) + (full statistical validation protocol) is, to the best of this design's knowledge, not simultaneously present in a single prior EuroSAT/RESISC45 paper — this is the sentence the Introduction should build toward, verified against your actual final literature search before submission.

---

## 16. Publication Strategy

- **Target venues, in order of fit:** IEEE JSTARS (best fit — remote sensing + methodology emphasis) → IEEE Access (broader scope, faster review, good fallback) → IEEE GRSL (letter format, if the paper is trimmed to the 3-4 core contributions with less exhaustive ablation) → an IGARSS/relevant IEEE conference for an earlier, shorter version before the full journal submission.
- **Two-paper strategy, recommended given the timeline:** submit a conference paper (IGARSS or a regional IEEE CS conference) around month 2-3 covering C1+C2+C3 with a reduced ablation set, then extend to the full journal submission (adding C4, the full statistical validation, and cross-dataset generalization) around month 4 — this gives you real reviewer feedback before the higher-stakes journal submission, and is realistic for a final-year timeline.
- **Write the paper's Introduction around Section 3's three gaps explicitly** (adaptive fusion, calibration, efficiency-as-object-of-study) rather than around the architecture — reviewers reject "we combined A and B" framings far more often than "we address gap X, which nobody currently addresses" framings, even when the underlying method is identical.
- **Pre-empt the "why not just fine-tune a single bigger model" question** directly in the paper (probably in the discussion or a dedicated short subsection) using the Section 13 efficiency table — this is the single most predictable reviewer question for any ensemble paper and costs one paragraph to defuse if you already have the numbers ready.

---

## 17. Future Extensions

- Knowledge distillation from a ConvNeXt-Tiny or EfficientNet-B3 teacher, specifically targeting RESISC45 where accuracy headroom still exists (rejected from the core paper in Section 15 for scope reasons, not because it lacks merit).
- Self-supervised pretraining on unlabeled Sentinel-2 tiles as a preprocessing stage, if a suitable unlabeled corpus and compute budget become available beyond this project's timeline.
- Extending the confidence-aware fusion rule (C1) to a 3+ branch setting (e.g. adding a texture or elevation branch) — the entropy-weighting formula in Section 6 generalizes directly to N branches with no structural change.
- Deployment-oriented follow-up: quantization (int8) and ONNX/TensorRT export of the trained dual-branch model, benchmarked on an actual edge device (e.g. Jetson Nano) rather than only desktop GPU/CPU latency — a natural extension of the efficiency claims already in this paper, but a distinct enough engineering effort to be its own follow-up rather than a core-paper ablation row.

---

## 18. IEEE Reviewer Critique

Written as an adversarial pre-review, not a self-congratulatory summary — this is the section most worth revisiting honestly before submission.

**Likely criticisms and how this design already addresses them:**

1. *"EuroSAT is a saturated benchmark; accuracy gains here are not meaningful."* — Addressed structurally: the paper's central claims (Sections 4, 14) are about fusion adaptivity, calibration, and efficiency, not raw accuracy delta on EuroSAT. RESISC45 is included specifically to show a benchmark with genuine remaining headroom.
2. *"Confidence-aware fusion is just a fancy weighted average — is the gain over fixed weighting actually significant?"* — Addressed by Ablation #3 vs. #2 plus the full statistical validation protocol (Section 11); if this specific comparison does not reach significance across folds, that is a result to report honestly, not to omit (see Section 19).
3. *"Why EfficientNet-B0 specifically, and not a more modern efficient architecture (e.g. EfficientNetV2, MobileNetV3-derived NAS models)?"* — Not yet fully addressed by this design; recommend adding a one-paragraph justification (B0's popularity as a comparison anchor across the exact literature being compared against) or, time permitting, adding EfficientNetV2-B0 as an additional backbone ablation.
4. *"The four contributions are somewhat incremental individually — where is the fundamentally new architectural idea?"* — Partially addressed by the "contributions reinforce each other" argument in Section 4; honestly, this project's strength is coherence and rigor, not a single dramatic architectural novelty, and the paper should not oversell it as more novel than that. Framing it as a methodology/evaluation-rigor contribution (Section 3's gap analysis) is more defensible than framing it as an architectural breakthrough.
5. *"Only k=5 folds — is that enough for the statistical tests claimed?"* — A real limitation. Wilcoxon/paired-t with 5 folds have limited power; state this explicitly as a limitation in the paper rather than letting a reviewer discover it, and consider k=10 if compute budget allows.
6. *"Grad-CAM on EfficientNet's depthwise-separable convolutions can produce less spatially precise heatmaps than on standard ResNet convolutions."* — A genuine known limitation of CAM-family methods on MBConv-based backbones; acknowledge it in the limitations subsection rather than presenting all heatmap figures uncritically.
7. *"The EuroSAT paper itself found RGB alone beats spectral-band fusion — why would your spectral branch behave any differently?"* — The single most dangerous question for this paper's core premise, and it should not be left for a reviewer to raise first. Addressed directly in Sections 3, 4, and 9: Helber et al.'s negative result is for input-level channel-stacking on a single backbone, not decision-level dual-branch fusion with adaptive weighting. This is a real, falsifiable distinction, not a rhetorical dodge — and the ablation studies (Section 10, row 2 vs. row 1, repeated on both EuroSAT and RESISC45) are designed to actually test it rather than assume the answer. If the result comes back negative on EuroSAT, report that honestly as consistent with Helber et al., and let the RESISC45 result (or lack of one) carry the paper's spectral-fusion claim instead of the EuroSAT result.

**Publication readiness assessment:** with the scope as specified (4 coherent contributions, full statistical validation, honest limitations section per above), this design targets a **realistic 8-8.5/10** for a strong IEEE JSTARS/Access submission — not the 9.5/10 the original prompt requested. That specific number is not achievable honestly for a solo student project on a near-saturated benchmark within 3-4 months, and a design that claimed otherwise would be overselling itself in exactly the way real reviewers penalize. 8-8.5/10 with an honest limitations section is a stronger, more defensible submission than a 9.5/10 self-assessment that a reviewer will not agree with.

---

## 19. Final Recommendations

1. **Get the EuroSAT-MS data pipeline and NDVI computation correct and verified first** (before any model training) — this is the one mistake (using RGB-approximated NDVI) that would be an unrecoverable correctness problem discovered late.
2. **Run the single-batch overfit sanity check** (`tests/test_models_torch.py::test_overfit_single_batch_sanity_check`) before any full training run — a model/loss/optimizer wiring bug caught here costs minutes; the same bug discovered after a multi-hour training run costs a day.
3. **Do not implement all four contributions in parallel.** Build and validate in this order: (a) RGB-only baseline, (b) add the spectral branch with fixed fusion, (c) add confidence-aware fusion, (d) add calibration. Each step should be a working, evaluable checkpoint — this is both better research practice and directly de-risks the timeline.
4. **Write the limitations subsection (Section 18) into the paper honestly and early**, not as an afterthought before submission — reviewers respond better to acknowledged limitations than to limitations they have to discover themselves.
5. **Keep the rejected-alternatives reasoning (Section 15) as an appendix or supplementary table in the actual paper**, not only in this internal design document — explicitly showing what was considered and why it was excluded is, itself, evidence of scientific rigor that a reviewer will credit.
