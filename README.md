# Microgrid Digital Twin — Stability, Power Quality, and Forecasting

This project builds a digital twin for renewable microgrids. It combines
two kinds of work:

- A **CSE / machine-learning side** that predicts whether a grid
  configuration is stable, quantifies how much the model's decision can
  be trusted, and forecasts power quality a few minutes into the future.
- An **EEE / physics side** that simulates a Bangladesh-context PV
  microgrid with an Active Power Filter, computes THD, and tracks
  battery ageing over time.

The CSE side owns the machine-learning pipelines and the tau-robustness
metric. The EEE side owns the Simulink model and the IEEE 14-bus
parametric study. This README walks through everything that is in the
repo so far.

If you are on the team — please jump straight to the
**"How to run everything"** section. If you are the supervisor — the
**"What's novel"** section is the important one.

---

## The one-minute summary

The CSE side has built three separate ML pipelines so far:

1. **UCI Grid Stability** — 5 classifiers on a public benchmark
   dataset, plus a new robustness metric we call the
   **tau-robustness margin**. Best accuracy 96.05%.
2. **Synthetic Bangladesh microgrid dataset** — 50,000 rows,
   50 columns of physics-informed synthetic grid data. Used it to
   train a **3-class IEEE 519 compliance classifier**
   (PASS / MARGINAL / FAIL). Best accuracy 99.82%.
3. **Multi-horizon THD forecasting** — LSTM, GRU, and MLP predicting
   voltage THD 5, 15, and 30 minutes ahead. All three deep models
   beat the naive persistence baseline by ~30% on RMSE.

Alongside these, the EEE side's Simulink script is in `eee_sim/`, and
the CSE side wrote a runnable Python port of it that reproduces the
headline result: **THD 27% → 2%** when the Active Power Filter turns on.

Everything is on GitHub at
**https://github.com/Fahmidaca/microgrid-dt-1**.

---

## Two halves that meet in the middle

```
                     MICROGRID DIGITAL TWIN
                     ======================
                              |
              +---------------+---------------+
              |                               |
       CSE / ML SIDE                    EEE / PHYSICS SIDE
              |                               |
   src/         (Python)             eee_sim/    (MATLAB + Python port)
   results/    (metrics)             figures/eee_sim/  (sim plots)
   figures/    (7 folders)
   models/     (checkpoints)
              |                               |
              +---------------+---------------+
                              |
                    "AI-Optimized Control"
                (the point where my ML models
                  become the controller inside
                     the EEE simulation)
```

That last box is the paper's main contribution. The EEE teammate's
compliance table has a scenario called *"AI-Optimized Control"* that
gets the best THD — that AI is *my* trained model. That's what makes
this an interdisciplinary paper instead of two separate ones.

---

## Folder map (click to browse)

| Folder | What's in it |
|---|---|
| [`src/`](src/) | All the Python — 8 scripts covering data loading, EDA, baselines, statistics, robustness, synthetic-data generation, compliance classifier, forecasting |
| [`data/raw/uci_grid/`](data/raw/uci_grid/) | UCI Grid Stability CSV (10k rows, downloaded automatically by the loader) |
| [`data/synthetic/`](data/synthetic/) | My synthetic Bangladesh microgrid dataset (50k rows, .csv + .parquet) — see the strong "do not publish on this" warning in that folder's README |
| [`eee_sim/`](eee_sim/) | The EEE teammate's Simulink builder + my Python port that reproduces the same math |
| [`figures/`](figures/) | 7 sub-folders of plots — EDA, robustness, forecasting, compliance, EEE simulation, synthetic-data EDA |
| [`results/`](results/) | Every metric I've computed as a CSV — model accuracies, McNemar p-values, robustness margins, forecasting RMSE etc. |
| [`models/`](models/) | Trained model checkpoints (`.pkl` for sklearn, `.pt` for PyTorch). Gitignored — regenerate by rerunning the scripts. |
| [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) | Draft paper structure and target venues |
| [`SRS_MicrogridDigitalTwin.tex`](SRS_MicrogridDigitalTwin.tex) | System requirements spec document |

---

## How to run everything (from a clean clone)

```bash
git clone https://github.com/Fahmidaca/microgrid-dt-1.git
cd microgrid-dt-1
pip install -r requirements.txt
```

That takes about 2 minutes. After that you can run any of the pipelines
independently:

```bash
# UCI Grid Stability pipeline — total ~5 minutes
python src/eda.py                    # 5 EDA plots
python src/baselines.py              # 5 classifiers x 5 seeds
python src/significance.py           # McNemar tests + bootstrap CIs
python src/robustness.py 100         # tau-robustness margin (~3 min)

# Synthetic dataset + downstream pipelines
python src/generate_synthetic_dataset.py 50000   # regenerate the CSV/parquet
python src/compliance_classifier.py              # 3-class IEEE 519 compliance
python src/forecasting.py                        # LSTM / GRU / MLP forecasting

# EEE physics simulation (my Python port of teammate's MATLAB)
python eee_sim/microgrid_pq_twin.py              # THD 27% -> 2% demo
```

Each script prints its results to the terminal and drops CSVs + PNGs
into `results/` and `figures/`.

---

## Part 1 — UCI Grid Stability pipeline

### What it is

The UCI Electrical Grid Stability Simulated Data benchmark from
Schafer et al. 2016 (*Eur. Phys. J. Special Topics* 225, 569). 10,000
snapshots of a small 4-bus decentralised smart grid, each described by
12 features (four reaction times `tau1..tau4`, four powers `p1..p4`,
four price-elasticity coefficients `g1..g4`) and labelled stable or
unstable.

It is an older dataset but still cited in 2024 and 2025 papers because
"benchmarks don't expire, methods do." The CSE side uses this as the
foundation because it is peer-reviewed, well understood, and lets us
compare against published baselines honestly.

### What is implemented

**File:** [`src/loaders/load_uci_grid.py`](src/loaders/load_uci_grid.py)
Auto-downloads and caches the CSV from the UCI archive on first use.

**File:** [`src/eda.py`](src/eda.py) → plots into
[`figures/eda/`](figures/eda/)
Five exploratory plots (class balance, tau distributions, correlation
matrix, tau-vs-stability boundary, PCA projection).

**File:** [`src/baselines.py`](src/baselines.py) → results into
[`results/baseline_summary.csv`](results/baseline_summary.csv)
Trains five classifiers — Logistic Regression, Random Forest, HistGB,
XGBoost, and MLP — with five different random seeds each. This is 25
independent training runs total, which lets us report mean ± std
instead of a single lucky number.

Best result: **MLP at 96.05% ± 0.62% accuracy** (macro-F1 95.72%). All
seeds land in a tight window because the split is deterministic.

**File:** [`src/significance.py`](src/significance.py) → results into
[`results/mcnemar_pairs.csv`](results/mcnemar_pairs.csv) and
[`results/bootstrap_ci.csv`](results/bootstrap_ci.csv)
For every pair of models, runs a paired McNemar's test with
Holm–Bonferroni correction to see if the differences are statistically
significant. Also computes 2000-iteration bootstrap 95% confidence
intervals. All 10 pairwise differences are significant at p < 0.05
after correction, so nothing here is "we got lucky."

### What's novel — the tau-robustness margin

This is the CSE side's main contribution.

**File:** [`src/robustness.py`](src/robustness.py) → plots into
[`figures/robustness/`](figures/robustness/)

Existing papers on this benchmark only report accuracy. But a grid
operator cares about *safety buffers*, not just correctness. So the
CSE side defined a new metric.

For every correctly-classified test point, the pipeline does a binary
search over the smallest reaction-time perturbation
`||delta_tau||_inf` (in seconds) that flips the model's prediction.
It searches all 16 sign patterns on `tau1..tau4` and clips to the
simulation range `[0.5, 10]`. The result is a **per-sample
tau-robustness margin** — how much consumer response times can
degrade before the model's stability call changes.

The interesting finding: **the most accurate model (MLP) is NOT the
most robust.** Logistic Regression has the lowest accuracy (81.5%) but
the largest safety buffer (mean margin 1.33 s stable / 2.12 s unstable).
So "more accurate" ≠ "more deployable" — an insight that pure-accuracy
reports hide completely.

That is the publishable insight of the CSE-side work.

### Honesty notes for Part 1

- The dataset is static configurations, not time series. We do not
  claim "early warning" prediction on this data because there is no
  time axis to predict along.
- We do not claim the results transfer to real PMU / SCADA data.
- Tau-margin is a robustness of the *classifier's decision*, not a
  guarantee about the underlying physical grid.

These are stated up front in the paper draft so no reviewer can accuse
the paper of overclaiming.

---

## Part 2 — Synthetic Bangladesh microgrid dataset

### Why synthetic in the first place

For the second stage of the project the CSE side needed a dataset with
more variables than UCI Grid — voltage, current, THD, frequency,
irradiance, battery state, load, and so on — plus a time axis so
forecasting could be tested. Real datasets with all that in one place
are hard to get, so a synthetic one was generated instead.

**Very important — this data is not for publication claims.** Every
row has a `source = "SYNTHETIC_GENERATOR_v1"` watermark column so
nothing gets mistakenly used as real evidence. There is a very
prominent warning in [`data/synthetic/README.md`](data/synthetic/README.md)
that reminds everyone this is for pipeline development only. When the
EEE side's Simulink IEEE 14-bus model starts producing real output,
that will replace the synthetic data and the same scripts will be
rerun.

This is flagged to the supervisor because Faisal Sir was very clear
in his announcement that fabricated data is one of the worst
research-integrity problems. Building a synthetic dataset to test the
ML code is fine; presenting synthetic results as real is not.

### What is implemented

**File:** [`src/generate_synthetic_dataset.py`](src/generate_synthetic_dataset.py)
Produces 50,000 rows across 10 different operating scenarios
(base case with no renewables, PV-only at 20% and 40% penetration,
wind-only, hybrid PV+wind, hybrid with heavy nonlinear loads, hybrid
with energy-storage mitigation, and finally "AI-optimized control").

The 50 columns cover almost everything a grid operator would see:

- **Time:** timestamp, hour of day, day of year
- **Environment:** solar irradiance, ambient temperature, humidity, wind speed
- **Voltage / current:** three-phase RMS, unbalance, neutral current
- **Power:** active, reactive, apparent, power factor
- **Frequency:** value, deviation from 50 Hz, rate of change (RoCoF)
- **Harmonics:** V-THD, I-THD, individual 5th / 7th / 11th / 13th
- **Renewables:** PV kW, wind kW, RE penetration %
- **Storage:** battery SOC, SOH, voltage, current, temperature, power
- **Load:** total kW, nonlinear-load fraction
- **Mode flags:** grid-connected, APF on, ESS on, AI-control on, fault flag
- **Labels:** operating scenario, IEEE 519 compliance, stability
- **Economics:** per-step cost in Bangladesh Taka, cumulative cost

Relationships between features are physics-informed, not independent
noise. PV depends on irradiance and NOCT-derated cell temperature,
wind follows a cube law of wind speed, THD grows with renewable
penetration and nonlinear-load fraction and shrinks with ESS + AI
control, battery temperature rises with I-THD-squared (ohmic heating),
and so on.

The generator outputs two files:

- [`data/synthetic/microgrid_synthetic_v1.csv`](data/synthetic/microgrid_synthetic_v1.csv)
  (~18 MB, human-readable)
- [`data/synthetic/microgrid_synthetic_v1.parquet`](data/synthetic/microgrid_synthetic_v1.parquet)
  (~6 MB, 3× smaller, loads faster)

Plus four EDA plots into [`figures/synthetic/`](figures/synthetic/).

### How to view the data as a table

The CSV opens fine in Excel (50k rows is well below Excel's limit).
For the parquet file (which is faster to work with) the free
**Tad viewer** works well (https://www.tadviewer.com — MIT-licensed,
no account, no fees). The VS Code "parquet-viewer" extension by
dvirtz also works.

---

## Part 3 — IEEE 519 compliance classifier

### What it does

**File:** [`src/compliance_classifier.py`](src/compliance_classifier.py)
Uses the synthetic dataset to train a 3-class classifier that predicts
whether a snapshot is IEEE-519 **PASS**, **MARGINAL**, or **FAIL**.

Class balance is heavy — about 81% PASS, 17% MARGINAL, 2% FAIL — so
the pipeline uses macro-F1 alongside plain accuracy, and reports
per-class F1 so the FAIL class (which is the rarest and most
operationally important) does not get lost.

Crucially, **V-THD, I-THD, and all six harmonic percentages are dropped
from the feature set.** Those are what define the label, so keeping
them would make the task trivial. The classifier only sees indirect
operational signals — voltage, current, load, storage state, weather.
The idea is to detect impending violations *before* a THD analyzer
confirms them.

### Results

Same setup as UCI: 5 classifiers × 5 seeds.

| Rank | Model | Accuracy | Macro-F1 | FAIL-F1 |
|---|---|---:|---:|---:|
| 1 | **XGBoost** | **99.82% ± 0.05%** | 99.55% | 99.28% |
| 2 | RF | 99.81% | 99.41% | 98.86% |
| 3 | HistGB | 99.79% | 99.36% | 98.79% |
| 4 | LogReg | 99.67% | 98.90% | 97.81% |
| 5 | MLP | 99.18% | 97.20% | 94.41% |

Only 24 errors out of 10,000 test samples on the best model.

### Honesty note about these numbers

The results are **too good to be believable**, which is a real
finding worth reporting. The synthetic generator's physics is very
clean — features like `RE_penetration_pct`, `nonlinear_load_frac`,
`ESS_on`, and `AI_control_on` essentially determine V-THD and
therefore the compliance verdict. Even after the direct THD columns
were dropped, the model can reconstruct them almost perfectly from the
other inputs.

Real grid data has much more chaos. Once the EEE side's real Simulink
output replaces the synthetic data, accuracies are expected to land
somewhere in the 85–92% range, which will actually be a much better
story for a paper.

Results are in
[`results/compliance_summary.csv`](results/compliance_summary.csv) and
[`figures/compliance/`](figures/compliance/).

---

## Part 4 — Multi-horizon THD forecasting

### What it does

**File:** [`src/forecasting.py`](src/forecasting.py)

The synthetic dataset has a time axis (one sample per minute for 34
days), so this pipeline predicts voltage THD five, fifteen, and thirty
minutes ahead using the last thirty minutes of operational signals as
input.

Four models:

- **Persistence baseline** — the naive prediction that says
  "future value = current value." No training required. Any deep
  model must beat this to be worth publishing.
- **MLP** — flattens the 30-minute window and runs it through three
  dense layers.
- **LSTM** — two-layer LSTM with 64 hidden units.
- **GRU** — same as LSTM but with GRU cells.

Uses a **chronological split** (first 70% for training, next 10% for
validation, last 20% for test). Random shuffling would leak future
information into the training set — a common time-series mistake that
the pipeline avoids on purpose.

### Results

| Model | Horizon | RMSE | MAE | R² | Early-warning F1 |
|---|---:|---:|---:|---:|---:|
| Persistence | 5 min | 1.50 | 1.17 | -0.67 | **0.088** |
| MLP | 5 min | 1.07 | 0.84 | +0.15 | 0.000 |
| **LSTM** | 5 min | **1.06** | **0.83** | **+0.16** | 0.000 |
| GRU | 5 min | 1.07 | 0.83 | +0.16 | 0.000 |

**Good news:** all three deep models beat persistence by 30% on RMSE.
The pipeline is extracting real signal.

**Honest weakness:** the early-warning F1 (whether the model
correctly predicts a future breach of the 5% IEEE 519 threshold)
collapses to zero for all deep models. Since breaches only happen 2%
of the time, MSE loss teaches the model to hug the mean and never
predict a breach.

This is not a bug — it is a fundamental limitation of point-forecast
MSE training on rare-event problems. Fixing it needs either
class-weighted loss, quantile forecasting, or reframing as a binary
"will there be a breach?" classification. That is the next step.

All six plots — truth-vs-prediction overlays at each horizon, RMSE
bars, early-warning F1 bars, and training curves — are in
[`figures/forecasting/`](figures/forecasting/).

---

## Part 5 — EEE-side physics simulation

This part is owned by the EEE side. The CSE side helped clean up the
MATLAB script and wrote a runnable Python port so the whole team can
see results without a MATLAB license.

**Files:**

- [`eee_sim/build_microgrid_pq_digital_twin.m`](eee_sim/build_microgrid_pq_digital_twin.m)
  — the MATLAB Simulink builder (needs MATLAB R2021b+)
- [`eee_sim/microgrid_params.m`](eee_sim/microgrid_params.m)
  — Bangladesh microgrid parameters + weather profiles
- [`eee_sim/microgrid_pq_twin.py`](eee_sim/microgrid_pq_twin.py)
  — the Python port, runs anywhere with Python 3.10+

Pipeline: PV + weather → 6-pulse rectifier harmonic signature →
SRF (synchronous reference frame) Active Power Filter → FFT-based
THD analyzer → battery-degradation twin with Bangladesh Taka cost.

**Headline result** (validated by both MATLAB and Python):

- **THD_i drops from 27.13% to 2.07%** the instant the APF turns on
  (well below IEEE 519's 5% limit).
- Projected battery replacement cost: about 27,000 BDT/year.

Four plots in [`figures/eee_sim/`](figures/eee_sim/) show the
turn-on transient, the THD time series, battery degradation, and
the harmonic spectrum before vs after the APF activates.

---

## What's still needed

Honest note on the gaps:

- **Real data from the EEE side.** The IEEE 14-bus Simulink model
  producing actual `.slx` outputs. All synthetic-data results need to
  be re-run on that before the paper submission.
- **Class-weighted forecasting.** The forecasting pipeline needs
  either a class-weighted MSE loss or a reformulation as a binary
  breach classifier to make the early-warning F1 meaningful.
- **Statistical significance for the forecasting numbers.** Right now
  only point estimates are reported. Multi-seed + bootstrap CI is the
  fix — same treatment as the UCI baselines.
- **Reproducibility audit.** UCI side has one; the synthetic-data
  side does not yet.
- **Physical validation from the EEE side.** The Simulink builder
  reproduces the expected THD suppression, but a formal reproduction
  of Schafer 2016 is not done.
- **Paper draft.** Structure is in
  [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md); the LaTeX manuscript itself
  is not written yet.

---

## How everything connects to the paper

The paper's story is this:

1. **The problem** — how do we know if a renewable microgrid is stable
   and IEEE-519 compliant, and how much can we trust that prediction?
2. **CSE contribution** — a tau-robustness margin metric that reveals
   the accuracy-vs-robustness trade-off prior work misses. Validated
   on the UCI Grid Stability benchmark.
3. **EEE contribution** — a Bangladesh-context PV microgrid
   digital-twin simulation with an Active Power Filter, showing the
   physical mechanism behind the metric.
4. **The bridge** — the CSE side's trained ML model *is* the
   "AI-Optimized Control" scenario in the EEE compliance table, which
   is what makes this an interdisciplinary paper rather than two
   separate ones.
5. **Extensions** — multi-horizon THD forecasting (started, needs work)
   and IEEE-519 compliance classification (pipeline done, waiting on
   real data).

Target venues (in
[`PAPER_OUTLINE.md`](PAPER_OUTLINE.md)):
IEEE Transactions on Smart Grid (Q1, IF~10) or Applied Energy
(Q1, IF~11), with a fallback to PES General Meeting workshops.
