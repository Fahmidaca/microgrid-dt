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

The CSE side has built four separate ML pipelines so far:

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
4. **Power-quality disturbance detection, prediction & explainability**
   — a second, smaller dataset (5,000 rows, field-measured at the
   Jamalpur site per team confirmation — see Part 6) with sensor-fault
   flags and four disturbance classes. Built an unsupervised fault
   detector, a 4-class disturbance predictor (99.7% accuracy), a SHAP
   explainability layer, and an interactive dashboard tying all three
   together (`streamlit run src/dashboard.py`). Two of the dataset's
   columns are team-calculated rather than measured — see Part 6
   before using `economic_cost_BDT` as an independent target.

Alongside these, the EEE side's Simulink builder is in `eee_sim/` — as
of the 2026-08-21 team meeting it has not been successfully run
(license/file-size issues), so it has not independently validated
anything yet. The CSE side wrote a runnable, numpy-only Python port of
the same physics that does run end-to-end and produces the headline
result: **THD 27% → 2%** when the Active Power Filter turns on. Treat
this as one working implementation, not a cross-validated one, until
the Simulink model actually runs.

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
                (the point where the CSE side's
                  ML models become the controller
                  inside the EEE simulation)
```

That last box is the paper's main contribution. The EEE side's
compliance table has a scenario called *"AI-Optimized Control"* that
gets the best THD — that AI is the CSE side's trained model. That is
what makes this an interdisciplinary paper instead of two separate
ones.

---

## Folder map (click to browse)

| Folder | What's in it |
|---|---|
| [`src/`](src/) | All the Python — 13 scripts covering data loading, EDA, baselines, statistics, robustness, synthetic-data generation, compliance classifier, forecasting, anomaly detection, disturbance prediction, explainability, and the interactive dashboard |
| [`data/raw/uci_grid/`](data/raw/uci_grid/) | UCI Grid Stability CSV (10k rows, downloaded automatically by the loader) |
| [`data/synthetic/`](data/synthetic/) | The synthetic Bangladesh microgrid dataset (50k rows, .csv + .parquet) — see the strong "do not publish on this" warning in that folder's README |
| [`data/external/`](data/external/) | The Part 6 power-quality disturbance dataset (5k rows, field-measured at Jamalpur per team confirmation) plus two real HOMER Pro exports used in Part 5: `homer_hourly_simulation.csv` (8,760-row annual hourly simulation) and `homer_npc_coe_optimization.csv` (HOMER's own 3-architecture NPC/COE optimization table). Two columns of the disturbance dataset are team-calculated, not measured — see Part 6 and [`docs/DATA_PROVENANCE_AND_QUALITY.md`](docs/DATA_PROVENANCE_AND_QUALITY.md) before citing `economic_cost_BDT`. `mesa_del_sol/` (gitignored — download it yourself from `doi:10.5061/dryad.fqz612jzb`) holds the independent real-microgrid dataset used for external V/f validation. |
| [`docs/`](docs/) | [`CSE_WORK_SUMMARY.md`](docs/CSE_WORK_SUMMARY.md) — a CSE-only walkthrough of everything in this section, plus the `data/external/` provenance investigation write-up and the original teammate scripts it was based on |
| [`eee_sim/`](eee_sim/) | The EEE side's Simulink builder (not yet running, see Part 5), the CSE side's Python port that reproduces the same math, a HOMER-hour-driven scenario runner, the HOMER economic-optimization summary script, the EEE-to-CSE bridge test, the cost-uncertainty Monte Carlo, and the Mesa Del Sol external-validation check |
| [`figures/`](figures/) | 10 sub-folders of plots — EDA, robustness, forecasting, compliance, EEE simulation, synthetic-data EDA, plus Part 6's anomaly / disturbance / xai |
| [`results/`](results/) | Every metric computed as a CSV — model accuracies, McNemar p-values, robustness margins, forecasting RMSE, anomaly-detection and disturbance-classifier scores, etc. |
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

# EEE physics simulation (Python port of the MATLAB Simulink model)
python eee_sim/microgrid_pq_twin.py              # THD 27% -> 2% demo
python eee_sim/microgrid_pq_twin_scenarios.py    # same twin driven by real HOMER hours
python eee_sim/homer_economic_optimization.py    # HOMER's own NPC/COE optimization table
python eee_sim/twin_to_cse_bridge.py             # feeds twin waveforms through the real CSE classifier (run disturbance_classifier.py first)
python eee_sim/sensitivity_analysis.py 200       # cost uncertainty (90% CI, not just a point estimate)
python eee_sim/mesa_del_sol_validation.py        # V/f idealization check vs. a real independent microgrid (needs the Dryad CSV locally, see below)

# Power-quality disturbance + cyber-resilience pipeline (Part 6)
python src/anomaly_detection.py         # Isolation Forest fault detection + threshold tuning
python src/disturbance_classifier.py    # 4-class disturbance prediction (run this before explainability.py / dashboard.py)
python src/explainability.py            # SHAP feature attribution on the disturbance model
python src/supervised_fault_check.py    # supervised vs. unsupervised fault detection, compared

# Interactive dashboard (needs disturbance_classifier.py run at least once first)
streamlit run src/dashboard.py          # opens in your browser at localhost:8501
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

**Headline result** (from the Python port; the Simulink model has not
yet been run successfully, so this is not cross-validated — see the
note in Part 5 below):

- **THD_i drops from 27.13% to 2.07%** the instant the APF turns on
  (well below IEEE 519's 5% limit).
- Projected battery replacement cost: about 27,000 BDT/year.

Four plots in [`figures/eee_sim/`](figures/eee_sim/) show the
turn-on transient, the THD time series, battery degradation, and
the harmonic spectrum before vs after the APF activates.

### Honesty note — Simulink status (2026-08-21)

At the 2026-08-21 team meeting the EEE side reported the Simulink
model (`build_microgrid_pq_digital_twin.m`) has not run successfully
— the `.slx` file is large and has not opened cleanly. The supervisor
redirected the team to a Python/Anaconda-based simulation instead.
That already exists: `microgrid_pq_twin.py` is a from-scratch numpy
reimplementation of the same physics (not a wrapper around Simulink),
runs on plain Python 3.10+, and is what actually produces every
number in this section. Until the Simulink model runs, don't describe
these results as "validated by both tools" — there is one working
implementation right now, not two independent ones.

### HOMER-driven scenario analysis

**File:** [`eee_sim/microgrid_pq_twin_scenarios.py`](eee_sim/microgrid_pq_twin_scenarios.py)
Drives the same electrical-level twin with real operating points
pulled from a HOMER Pro annual hourly simulation
([`data/external/homer_hourly_simulation.csv`](data/external/),
8,760 rows) instead of one arbitrary demo timeline. It picks three
hours matching the proposal's three test scenarios — peak-load hour
(harmonic distortion), the hour with the sharpest renewable-output
drop (voltage sag), and a low-solar/high-wind/high-load hour
(combined weather-electrical) — and re-runs the APF/THD pipeline at
each.

**Honest finding:** THD suppression comes back identical (27.13% →
2.07%) at all three operating points. This is not a bug — in this
model, harmonic content is a fixed *percentage* of the fundamental
current, so THD is mathematically scale-invariant to how heavily
loaded the system is. Only the projected cost differs meaningfully
across the three hours (driven by how much sunlight each hour had,
not by THD or load). Read this as "the APF's suppression is robust
across the annual operating envelope" plus "cost varies with solar
throughput" — not as the three scenarios differing in power-quality
severity, since in this model they don't. The idealized 6-pulse
rectifier model doesn't capture load-dependent harmonic variation;
that would need a source-impedance/commutation-reactance term with a
citable physical basis, not an invented curve.

### HOMER system-level cost optimization

**File:** [`eee_sim/homer_economic_optimization.py`](eee_sim/homer_economic_optimization.py)
Reads HOMER Pro's own optimization results
([`data/external/homer_npc_coe_optimization.csv`](data/external/)) —
a real techno-economic comparison of three candidate system
architectures, not an estimated or derived column. This is the
strongest cost evidence in the project:

| Architecture | PV | Wind turbines | Fuel cell | Renewable fraction | Cost of energy | Net present cost |
|---|---|---|---|---|---|---|
| Config 1 | 465 kW | 23 | 80 kW | 81.4% | **$0.0278/kWh** | $501,600 |
| Config 2 | — | 15 | 80 kW | 53.3% | $0.0671/kWh | $736,617 |
| Config 3 | — | — | 80 kW | 0% | $0.1032/kWh | $1,120,089 |

Going from 0% to 81.4% renewable penetration cuts the cost of energy
by roughly 4×. Unlike `economic_cost_BDT` in the Part 6 dataset (a
disputed per-row formula) or even this twin's own assumed-parameter
cost model, these numbers come directly from HOMER's optimizer.

### Closing the loop: EEE twin → CSE classifier

**File:** [`eee_sim/twin_to_cse_bridge.py`](eee_sim/twin_to_cse_bridge.py)
Until now the EEE physics simulation and the CSE ML pipeline ran on
completely unrelated data with no interaction, despite the proposal's
title promising an integrated "digital twin." This closes that gap:
it injects four kinds of event into the electrical-level twin — None,
Harmonic_Distortion, Voltage_Sag, Combined_Weather_Electrical, the
same four classes Part 6's disturbance dataset uses — extracts the
same 10-feature vector, and asks the RF classifier from
[`src/disturbance_classifier.py`](src/disturbance_classifier.py)
(trained entirely on real Jamalpur field data, never on anything
simulated) to classify the physics-simulated waveform.

**First attempt: 25% accuracy — the classifier just predicted "None"
every time.** Diagnosis: `plant_step` assumes an ideal, zero-impedance
voltage source, so `THD_voltage_pct` came out exactly 0.0 for every
simulated case, no matter the scenario — but real Jamalpur data shows
`THD_voltage_pct` is the single biggest discriminator between
Harmonic_Distortion (mean 11.3%) and Voltage_Sag (mean 2.9%). A
zero-impedance source is structurally incapable of producing that
signal. Also found and fixed a real bug: the sag-scenario analysis
window was sampling *after* the sag had already ended.

**Fix:** added a standard textbook inductive source reactance
(`X_h = h · X1`, i.e. harmonic impedance scales linearly with harmonic
order) on top of `plant_step`, applied only inside this bridge script
so every other script's already-committed results are untouched. Also
recalibrated the harmonic-injection amplitude to real Jamalpur THD
magnitudes per scenario (baseline ~3%, disturbance ~7-9%) instead of
the uncalibrated default's 27% — same 6-pulse harmonic-order shape
(physically justified), rescaled severity to match observed field
values instead of an invented curve.

**Result after the fix: 93% accuracy** (100 simulated cases, 25 per
class) — Harmonic_Distortion and None both 100% F1, Voltage_Sag 88%,
Combined_Weather_Electrical 84%. The Isolation Forest anomaly detector
(also trained only on real data) flags 96-100% of disturbed simulated
cases vs. 28% of normal ones. `X1` (the assumed source reactance) is
itself an unmeasured parameter — same epistemic status as `cost_kWh`
and `base_fade` below, worth stating as an assumption in the paper,
not a measured grid property. Full per-feature domain-shift numbers
in [`results/twin_to_cse_bridge_domain_shift.csv`](results/).

### Cost uncertainty (not just a point estimate)

**File:** [`eee_sim/sensitivity_analysis.py`](eee_sim/sensitivity_analysis.py)
The twin's cost projection depends on two assumed, uncited parameters
(`cost_kWh` = 24,000 BDT/kWh, `base_fade` = 8×10⁻⁷ SoH/kWh
throughput). Reporting a single deterministic number from those
implies false precision — the CSE side already reports every accuracy
number as mean ± std over 5 seeds with bootstrap CIs, and the EEE side
had no equivalent treatment until now. A 200-run Monte Carlo with both
parameters perturbed ±20% gives:

- Point estimate: 26,955 BDT/yr
- Monte Carlo mean: 26,871 BDT/yr, std: 4,279 BDT/yr
- **90% interval: [19,623 – 34,171] BDT/yr**

Report the interval in the paper, not just the point estimate.

### External validation: is "constant V, constant f between events" realistic?

**File:** [`eee_sim/mesa_del_sol_validation.py`](eee_sim/mesa_del_sol_validation.py)
The twin's `plant_step()` holds voltage and frequency at an exact
constant except during the one scripted sag — a modeling convenience,
never checked against a real grid. Mesa Del Sol (University of New
Mexico microgrid, Dryad `doi:10.5061/dryad.fqz612jzb`) is an unrelated,
independently-collected, DOI-cited real system — 60 Hz/~484 V, so its
raw numbers don't compare to Jamalpur/the twin's 50 Hz/230 V, but its
*normalized* (per-unit) variability does. One representative month
(April 2023, 259,200 valid 10-second samples, after dropping an
8,424-row sentinel/placeholder block the source logger itself inserts):

- Frequency: continuous std ≈ 0.03% pu, full range ≈ ±0.4% pu
- Voltage: continuous std ≈ 0.5–0.6% pu, full range ≈ 4–5% pu
- …with **no fault or sag event scripted** for nearly all of that month.

That's real, independent evidence that "V and f sit at an exact
constant except during one scripted event" is an idealization worth
naming explicitly as a limitation in the paper, rather than an
assumption nobody checked. No THD/harmonic columns in this dataset, so
it validates V/f behavior only — it cannot and does not validate
Jamalpur's THD numbers or the disturbance classifier. Raw CSVs (467 MB
across 15 months) aren't committed — cite the DOI, don't redistribute
someone else's dataset; download it yourself and drop a monthly file at
`data/external/mesa_del_sol/<Month>_<Year>.csv` to reproduce. See
[`results/mesa_del_sol_validation.csv`](results/mesa_del_sol_validation.csv)
and
[`figures/eee_sim/09_mesa_del_sol_validation.png`](figures/eee_sim/09_mesa_del_sol_validation.png).

---

## Part 6 — Cyber-resilience, disturbance prediction & explainability

### Why this part exists

A follow-on proposal, *"A Cyber-Resilient, Explainable Digital Twin
Framework for Predictive and Cost-Aware Power Quality Management in
Renewable Microgrids,"* asks for three things Parts 1–5 didn't cover
yet:

1. An **anomaly-detection layer** that can flag a sensor fault or bad
   reading without being told in advance what a fault looks like
   (the "cyber-resilience" piece — a grid shouldn't trust a reading
   just because it arrived).
2. A **dedicated prediction model** for *which kind* of power-quality
   disturbance is happening, not just whether the grid is stable or
   IEEE-519 compliant.
3. An **explainability layer**, so a prediction comes with a reason a
   human can check, instead of a black-box number.

This part adds all three, using a new, smaller dataset that a team
member provided separately from the Part 2 synthetic dataset.

### The dataset

**File:** [`data/external/microgrid_power_quality_dataset.csv`](data/external/)
— 5,000 rows, 16 columns: voltage/current/frequency, temperature +
irradiance (weather), three harmonic percentages, voltage/current THD,
a `sensor_fault_flag`, a `disturbance_type` (Voltage_Sag /
Harmonic_Distortion / Combined_Weather_Electrical / none), and three
battery/cost columns.

**Please read the honesty note below before using this dataset in
any claim about real-world performance.**

### What is implemented

**File:** [`src/anomaly_detection.py`](src/anomaly_detection.py) →
results in
[`results/anomaly_detection_summary.csv`](results/anomaly_detection_summary.csv),
[`results/anomaly_threshold_tuning.csv`](results/anomaly_threshold_tuning.csv),
plots in [`figures/anomaly/`](figures/anomaly/)
Fits an Isolation Forest (unsupervised — it never sees the fault
labels during training) on this part's disturbance dataset **and** on
the Part 2 synthetic dataset, side by side, rather than merging the
two (their column schemas aren't compatible — one is single-phase with
16 columns, the other three-phase with 50). Also adds 5-fold
cross-validated threshold tuning on top of the raw anomaly score, to
separate "is the ranking any good" (ROC-AUC) from "did we pick a good
cutoff" (precision/recall/F1).

**File:** [`src/disturbance_classifier.py`](src/disturbance_classifier.py)
→ results in
[`results/disturbance_summary.csv`](results/disturbance_summary.csv),
plots in [`figures/disturbance/`](figures/disturbance/)
Trains 5 classifiers × 5 seeds (same recipe as Parts 1 and 3) to
predict `disturbance_type` from the electrical + weather columns.

**File:** [`src/explainability.py`](src/explainability.py) → results
in
[`results/shap_feature_importance.csv`](results/shap_feature_importance.csv),
plots in [`figures/xai/`](figures/xai/)
Runs SHAP on the trained disturbance classifier to show which features
drive each prediction. (Run `disturbance_classifier.py` first — this
script loads its saved model instead of retraining.)

**File:** [`src/supervised_fault_check.py`](src/supervised_fault_check.py)
→ results in
[`results/supervised_vs_unsupervised_fault_detection.csv`](results/supervised_vs_unsupervised_fault_detection.csv)
A supervised classifier trained directly on `sensor_fault_flag` (i.e.
*with* label access), to measure how much of the anomaly-detection gap
above is "the fault signal isn't really there" vs. "Isolation Forest
just can't find it unsupervised."

**File:** [`src/dashboard.py`](src/dashboard.py) —
**live at https://fahmidaca-microgrid-dt-srcdashboard-7yxtjd.streamlit.app**,
or run locally with `streamlit run src/dashboard.py`
The proposal's Module 7 (visualization dashboard) and the piece that
ties Parts 6's three models together into one view instead of three
separate scripts. A slider picks a row of the disturbance dataset
("the current reading"); the page reacts live:

- **Current reading** — voltage/current/frequency/THD/weather as
  metrics, with deviation from nominal shown for voltage/current/frequency.
- **Alerts** — runs the saved disturbance classifier and the Isolation
  Forest anomaly detector on that row and shows a plain-language
  alert banner for each (disturbance predicted / not, anomalous
  reading / not), plus the ground-truth fault flag if that row has one.
- **Why this prediction** — a live SHAP bar chart for the specific row
  selected, not just an aggregate plot.
- **Cost impact** — this row's cost/degradation numbers plus a
  cumulative-cost chart up to the selected row. Explicitly labelled as
  team-calculated, not measured, per the honesty note above — the
  dashboard itself carries the same caveat instead of presenting these
  numbers as ground truth.
- **Historical trend** — THD/voltage/current over the whole dataset
  with disturbance rows color-coded and the selected row marked.

### Results

**Disturbance prediction** (5 models × 5 seeds; RF was the specific
model saved and used for the confusion matrix / SHAP plots below):

| Model | Mean accuracy | Mean macro-F1 |
|---|---:|---:|
| HistGB | 99.38% | 98.56% |
| XGBoost | 99.34% | 98.49% |
| **RF** (saved model) | 99.26% (99.70% on its saved split) | 98.29% |
| MLP | 98.80% | 97.27% |
| LogReg | 98.02% | 95.36% |

**Anomaly / fault detection**, Isolation Forest, this part's dataset
vs. the Part 2 synthetic dataset:

| Dataset | True fault rate | ROC-AUC | Precision / Recall / F1 (default threshold) | Precision / Recall / F1 (5-fold CV-tuned threshold) |
|---|---:|---:|---|---|
| This part's disturbance dataset | 2.40% | **0.70** | 0.038 / 0.233 / 0.066 | 0.057 / **0.64** / 0.105 |
| Part 2 synthetic dataset | 0.42% | 0.50 (chance) | 0.005 / 0.005 / 0.005 | ~0 / ~0.02 / ~0.01 |

The Part 2 synthetic dataset's `fault_flag` is injected as pure random
noise in its generator (`src/generate_synthetic_dataset.py`),
uncorrelated with any feature on purpose — so a chance-level 0.50
ROC-AUC there is the *correct* result, confirming the detector isn't
finding phantom patterns, not a failure of the method.

**Supervised vs. unsupervised**, on this part's dataset only:

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Supervised RF (uses fault labels) | 1.00 | 0.46 | 0.63 | 0.82 |
| Unsupervised IsolationForest (CV-tuned) | 0.057 | 0.64 | 0.105 | 0.70 |

Reading this: the fault signal *is* present in the electrical
features (supervised access closes most of the gap), but it isn't
shaped like a natural multivariate outlier, which is why the
unsupervised method struggles. That's a real methodological finding
worth stating in the paper as-is, not a sign anything is broken.

### Honesty note about the disturbance dataset — resolved, please still read this

**Update: the team member who supplied this dataset has confirmed the
electrical readings were collected at the Jamalpur powerplant site.**
That resolves the main provenance question — the voltage / current /
frequency / harmonic / THD / temperature / irradiance /
`sensor_fault_flag` columns are field-measured data.

One thing that confirmation does *not* explain, and is still worth
knowing before citing it: two of the derived columns are exact
formulas, not independent measurements. `economic_cost_BDT` is
`battery_degradation_rate × ~1500` for essentially every single row
(correlation 0.99999998), and `battery_capacity_loss_pct` correlates
0.999 with plain row order — it behaves like a counter, not a value
that responds to which disturbance happened in that row. No sensor
logs a cost column like that; the straightforward explanation, now
that the electrical signals are confirmed field data, is that these
two columns were **calculated by the team afterward** rather than
measured — which is completely normal for a derived field, it just
needs to be described that way rather than as an independent
measurement.

**How to describe each column in the manuscript:**

| Column(s) | Status |
|---|---|
| voltage/current/frequency/harmonics/THD/weather/`sensor_fault_flag` | Field-measured at Jamalpur, per team confirmation |
| `battery_degradation_rate` | Correlates sensibly with the electrical features — describe as derived from them, confirm the exact method with the teammate |
| `economic_cost_BDT`, `battery_capacity_loss_pct` | Team-calculated/estimated, not measured — re-derive `economic_cost_BDT` from an explicit tariff/replacement-cost formula before using it as a validation target, since as-is a model would just rediscover the ×1500 constant |

Full numbers and the original investigation are in
[`docs/DATA_PROVENANCE_AND_QUALITY.md`](docs/DATA_PROVENANCE_AND_QUALITY.md).

---

## What's still needed

Honest note on the gaps:

- **Real data from the EEE side.** As of the 2026-08-21 meeting, the
  Simulink model still hasn't run (file-size/license issues) and the
  supervisor has redirected the team to Python-based simulation
  instead of chasing `.slx` output — see Part 5's honesty note. The
  Python port already covers this; the open item is now the HOMER
  hourly/optimization data (see Part 5), not Simulink specifically.
- **Class-weighted forecasting.** The forecasting pipeline needs
  either a class-weighted MSE loss or a reformulation as a binary
  breach classifier to make the early-warning F1 meaningful. (Parts
  1-4 thread, not a blocker for the ICCIT submission.)
- **Statistical significance for the forecasting numbers.** Right now
  only point estimates are reported. Multi-seed + bootstrap CI is the
  fix — same treatment as the UCI baselines. (Parts 1-4 thread, not a
  blocker for the ICCIT submission.)
- ~~Reproducibility audit.~~ — **done (2026-08-21).** Every pipeline
  in this README (UCI, synthetic-data, Part 6, both EEE scripts) was
  run end-to-end from a clean environment. Found and fixed two real
  bugs in the process: `ax.boxplot(labels=...)` was removed in
  matplotlib 3.11 (renamed to `tick_labels`, broke `robustness.py` and
  `generate_synthetic_dataset.py`), and `torch` was missing from
  `requirements.txt` even though `forecasting.py` requires it. Both
  fixed; a clean `pip install -r requirements.txt` now actually
  reproduces everything in this README.
- **Physical validation from the EEE side.** The Simulink builder
  reproduces the expected THD suppression, but a formal reproduction
  of Schafer 2016 is not done. (Note: this item belongs to the Parts
  1-4 / tau-robustness paper thread, not the ICCIT Part 5-6 submission
  — not a blocker for the paper in `paper/ICCIT2026_draft.tex`.)
- ~~Paper draft.~~ — **started (2026-08-21).**
  [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) is the outline for a
  *different* paper (the tau-robustness margin work, Parts 1-4,
  targeting a Q1 journal) — it does not cover Part 6's proposal
  ("Cyber-Resilient, Explainable Digital Twin...") at all. The actual
  ICCIT 2026 draft is
  [`paper/ICCIT2026_draft.tex`](paper/ICCIT2026_draft.tex): full
  section structure, every headline number pulled from a committed,
  re-runnable script, honest caveats stated inline rather than
  hidden. Not compile-tested (no LaTeX toolchain was available when it
  was written). Section II (Related Work) now cites 7 real papers
  [`paper/references.bib`](paper/references.bib) — each verified to
  resolve against Crossref/Dryad's public metadata API (title, authors,
  journal, DOI all confirmed), summarized from structured abstracts
  rather than a full read of each PDF, so skim the actual papers
  yourselves before submitting to confirm the framing is fair. The
  author block is still a stub — add real author names/affiliations
  before submitting.
- **`economic_cost_BDT` — resolved by decision, not by formula.** Its
  underlying column, `battery_degradation_rate`, has no documented
  time/unit basis (per-hour? per-reading? cumulative?), so any
  "properly re-derived" cost formula built on it would just be another
  arbitrary constant — the exact problem being fixed. Decision: do not
  use `economic_cost_BDT` as a modeling target at all. Part 5's twin
  cost model (with a 90% CI, see "Cost uncertainty") and HOMER's own
  optimization table are both provenance-clean and used instead. Ask
  the teammate who supplied the dataset what units/timebase
  `battery_degradation_rate` is in, alongside the Jamalpur handoff-doc
  request already in progress.
- ~~Resolve the Part 6 dataset's origin~~ — **done.** Team confirmed
  Jamalpur field origin for the electrical readings; see Part 6's
  honesty note and [`docs/DATA_PROVENANCE_AND_QUALITY.md`](docs/DATA_PROVENANCE_AND_QUALITY.md)
  for the per-column writeup. A handoff-document request is in
  progress (as of 2026-08-21) for further independent confirmation.
- ~~Visualization dashboard~~ — **done**, see Part 6's
  [`src/dashboard.py`](src/dashboard.py) (`streamlit run src/dashboard.py`,
  smoke-tested 2026-08-21).
- ~~EEE-side real annual operating data~~ — **partially done.** The
  HOMER hourly simulation (8,760 rows) and optimization results are
  real HOMER Pro output — see Part 5. Still missing: the Simulink
  electromagnetic-transient model itself.
- ~~EEE and CSE sides are disconnected pipelines~~ — **done.** See
  Part 5's "Closing the loop" section:
  [`eee_sim/twin_to_cse_bridge.py`](eee_sim/twin_to_cse_bridge.py)
  feeds EEE-simulated waveforms through the CSE classifier trained
  only on real data - 93% sim-to-real accuracy after fixing a real
  zero-source-impedance modeling gap the test itself surfaced.
- ~~External validation against an independent real microgrid~~ —
  **done.** [`eee_sim/mesa_del_sol_validation.py`](eee_sim/mesa_del_sol_validation.py)
  checks the twin's constant-V/constant-f assumption against one month
  (April 2023, 259,200 valid 10-second samples after dropping a
  placeholder/sentinel block) of real telemetry from the Mesa Del Sol
  microgrid (University of New Mexico via Dryad,
  `doi:10.5061/dryad.fqz612jzb` — a different continent, 60 Hz/~484 V
  system, so absolute numbers aren't comparable to Jamalpur; only the
  normalized, per-unit behavior is). Result: real frequency has
  continuous std of ~0.03% pu (range ±0.4% pu) and real voltage has
  continuous std of ~0.5-0.6% pu (range ~4-5% pu) even with **no**
  fault or sag event scripted for most of the record — concrete,
  independently-sourced evidence that `plant_step()`'s exact-constant
  V/f between events is an idealization, not a hand-waved caveat. No
  THD/harmonic columns in this dataset, so it validates V/f behavior
  only, not the disturbance classifier. See
  [`results/mesa_del_sol_validation.csv`](results/mesa_del_sol_validation.csv)
  and
  [`figures/eee_sim/09_mesa_del_sol_validation.png`](figures/eee_sim/09_mesa_del_sol_validation.png).
  Raw CSVs are not committed (467 MB across 15 months, belongs to the
  Dryad DOI — cite it, don't redistribute it); download it yourself and
  drop the monthly file at `data/external/mesa_del_sol/Apr_2023.csv` to
  re-run.
- ~~No uncertainty quantification on the EEE side~~ — **done.** See
  Part 5's "Cost uncertainty" section:
  [`eee_sim/sensitivity_analysis.py`](eee_sim/sensitivity_analysis.py)
  reports a 90% CI on the cost projection instead of one deterministic
  number.

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
