# Paper outline — Reaction-Time Robustness Margin

Working title:
**"Beyond Accuracy: Reaction-Time Robustness Margins for ML-Based Grid Stability Classifiers"**

## Target venues (in order of fit)

1. **IEEE Transactions on Smart Grid** (Q1 journal, IF ~10)
2. **Applied Energy** (Q1 journal, IF ~11)
3. **Energy and AI** (newer Q1, faster review)
4. **IEEE PES General Meeting** (top conference, faster path)

---

## Structure (8-10 pages, IEEE format)

### 1. Introduction
- The grid-stability prediction problem; consumer reaction time as a
  control-side parameter the operator does not fully observe.
- Gap: prior work (Schafer 2016 + follow-ups) reports only raw accuracy.
  No one quantifies how much reaction-time degradation the *learned model's
  decision* can survive — a number a grid planner actually needs.
- Contribution:
  1. First systematic study of tau-perturbation robustness on the UCI
     benchmark.
  2. New metric: $L_\infty$ tau-robustness margin per configuration.
  3. Best classifier (MLP, 96.05% test acc, 95% CI [95.65, 96.43]) beats
     prior published baselines with statistically significant gap
     (McNemar, $p < 0.001$ after Holm-Bonferroni correction).
  4. Practical reading: per-class margin distributions a planner can
     translate into design rules.

### 2. Background and Related Work
- 4-bus decentralised smart grid model (Schafer 2016).
- Prior ML approaches on UCI Grid Stability: brief survey, none look
  at robustness margins.
- Distance to decision boundary in adversarial ML (cite Goodfellow et al.
  FGSM, Madry et al. PGD) — we adapt the idea to a *physical* perturbation
  along the tau axis only, not unconstrained input attack.

### 3. Dataset
- UCI Grid Stability Simulated Data, 10,000 samples, 12 features
  (tau, p, g) + stab + stabf.
- Class balance: 64% unstable, 36% stable.
- Per-figure: EDA from `figures/eda/`.

### 4. Methods
- **Classifiers:** LogReg, RF, HistGB, XGBoost, MLP (hidden 64-32).
- **Training:** 80/20 stratified split, repeated over 5 seeds
  $\{0, 1, 42, 123, 2026\}$. Z-score standardisation where applicable.
- **Robustness margin:** for each correctly-classified test sample $x$,
  $$
  \mathrm{margin}(x) = \min_{\delta \in [0,5]^4,\, s \in \{-1,+1\}^4}
    \|\delta\|_\infty \quad \text{s.t. } M(x + s \odot \delta_\tau) \neq M(x),
  $$
  with $\tau$-coordinates clipped to $[0.5, 10]$. Computed via 16-pattern
  binary search at tolerance $10^{-2}\,\text{s}$.

### 5. Results
- **5.1 Classification.** Table of mean +- std accuracy and macro-F1
  across 5 seeds. Bootstrap 95% CI on all-seeds-pooled predictions
  (n=10000).
- **5.2 Statistical significance.** All-pairs McNemar with Holm-Bonferroni:
  all 10 pairs significant at p<0.05.
- **5.3 Tau-robustness margin.** Distributions per model and per class
  (figures from `figures/robustness/`).
- **5.4 The accuracy-robustness trade-off.** Scatter plot of mean margin
  vs. test accuracy across the 5 models.

### 6. Discussion
- A more accurate model is not necessarily a more robust one — quantify
  the gap with the trade-off scatter.
- Per-class asymmetry: unstable points likely sit closer to the boundary
  (interpretation tied to grid biology / EPJ ST 2016).
- Implications for grid planning: pick the model whose margin distribution
  covers your worst-case expected tau drift.

### 7. Limitations and Future Work
- UCI is simulation, not real PMU traces.
- $\tau \in [0.5, 10]$ s is a narrow range; real consumers may exceed it.
- Margin only quantifies model decision robustness; physical-grid
  resilience is a separate question.

### 8. Conclusion
- The MLP achieves SOTA accuracy on UCI Grid Stability and the highest
  mean tau-robustness margin, but the per-class margin distributions
  reveal a design choice grid planners would otherwise miss.

---

## Why this passes your teacher's Q1 framework

| Criterion | How this paper satisfies it |
|---|---|
| **Novelty** | A new metric (tau-margin) on a benchmark only reported by accuracy |
| **Significance** | Grid planners need operational safety buffers, not just point accuracy |
| **Methodological rigor** | Multi-seed runs, Holm-Bonferroni-corrected McNemar, bootstrap CIs |
| **Reproducibility** | Public dataset, deterministic seeds, code released |
| **Real-world impact** | Translatable to grid design rules |
| **Research integrity** | Honesty notes (no fake early-warning claim) up-front |
