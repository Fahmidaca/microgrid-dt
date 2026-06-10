# Microgrid Stability — Reaction-Time Robustness Margin

Research paper pipeline for **predicting grid stability and quantifying the
operational safety buffer** against degraded consumer reaction times, on the
UCI Electrical Grid Stability Simulated Data benchmark
(Schafer et al., *Eur. Phys. J. Special Topics* 225, 569 (2016)).

## 🎯 Research question

> *"Beyond raw classification accuracy, how much consumer reaction-time
> degradation can a learned grid-stability classifier absorb before its
> prediction flips?"*

This goes beyond what prior work on the UCI benchmark reports
(Schafer 2016 and follow-ups quote only accuracy / regression error).
We introduce a **tau-robustness margin** that measures, per
configuration, the smallest reaction-time perturbation
$\|\delta\tau\|_\infty$ that flips the model's prediction. Larger margin =
safer grid operating point.

## 📂 Layout

```
microgrid-dt/
├── data/raw/uci_grid/Data_for_UCI_named.csv   # 10k rows, 14 cols
├── src/
│   ├── loaders/load_uci_grid.py               # stream + cache download
│   ├── eda.py                                 # 5 EDA plots
│   ├── baselines.py                           # 5 models x 5 seeds
│   ├── significance.py                        # McNemar + bootstrap + Holm
│   └── robustness.py                          # NOVEL: tau margin per sample
├── figures/eda/                               # 5 EDA pngs
├── figures/robustness/                        # 2 robustness pngs
├── results/                                   # CSV outputs (gitable)
├── models/                                    # *.pkl trained classifiers
├── requirements.txt
└── README.md
```

## 🚀 Reproduce end-to-end

```bash
pip install -r requirements.txt
python src/eda.py          # plots only, no claims yet
python src/baselines.py    # ~30s  -> trains 5 models x 5 seeds, saves preds
python src/significance.py # <5s   -> McNemar + bootstrap CI, Holm-Bonferroni
python src/robustness.py   # ~3min -> tau-robustness margin per model
```

## 📊 Headline results (held-out test set, n_test=2000 per seed × 5 seeds)

| Model     | Acc          | 95% CI         | macro-F1     |
|-----------|-------------:|---------------:|-------------:|
| MLP       | **96.05%**   | [95.65, 96.43] | 95.72%       |
| HistGB    | 94.72%       | [94.28, 95.13] | 94.24%       |
| XGBoost   | 94.39%       | [93.92, 94.83] | 93.87%       |
| RF        | 92.16%       | [91.64, 92.70] | 91.38%       |
| LogReg    | 81.50%       | [80.72, 82.29] | 79.58%       |

All pairwise differences are significant at $p < 0.05$ after Holm-Bonferroni
correction (10 paired McNemar tests).

## 🆕 Novel contribution — Reaction-Time Robustness Margin

For each correctly-classified test sample, we compute
$\|\delta\tau\|_\infty$ — the smallest $L_\infty$ perturbation in
reaction time (across the four agents) that flips the model's prediction.
This is computed by exhaustive binary search across the 16 sign patterns
on $\tau_{1..4}$, with clipping to the simulation range $[0.5, 10]\,\text{s}$.

Larger margin = the prediction (and thus the grid's certified stability
status) survives more degradation in agent responsiveness. See
`figures/robustness/` for distributions per model and per class.

## 🔬 Honesty notes (what we do NOT claim)

- We do **not** claim "early warning" prediction. The UCI dataset is a
  set of 10,000 static grid configurations with analytical stability
  labels, not a time series. There is no warning horizon to predict.
- We do **not** claim our results transfer to real PMU/SCADA data; the
  reaction-time range used by Schafer 2016 ($\tau \in [0.5, 10]$ s) is
  narrower than what some real consumers exhibit.
- Tau-robustness margin is a **model-side** robustness metric; it does
  not claim the underlying physical grid is robust — only that the
  classifier's decision is.
