# ⚠️ SYNTHETIC DATASET — DO NOT USE FOR PUBLICATION CLAIMS ⚠️

This folder contains **synthetically generated** microgrid time-series data.
Every row of every file has `source = "SYNTHETIC_GENERATOR_v1"` so any
downstream code or paper table can verify the origin at a glance.

## What this dataset IS for

✅ Building and testing the ML/analysis pipeline (proof-of-concept)
✅ Data augmentation on top of a real base dataset
✅ Explicit "simulation study" experiments where the paper clearly says the
    data are synthetic
✅ Course project / thesis engineering demos

## What this dataset IS NOT for

❌ Presenting model accuracy on synthetic data as "real-world" results in
    a Q1 journal or IEEE conference paper
❌ Any claim that begins with "we evaluated on real microgrid data..."
❌ Skipping the real Simulink simulator run when a paper requires it

**This matters because the supervisor explicitly told us that
Fabricated Data is a top research-integrity violation:**

> *"গবেষণাগত অসততার একটি বড় কারণ হলো ... Fabricated Data ...
>  একবার বিশ্বাস হারালে ... সবকিছুই ক্ষতিগ্রস্ত হয়।"*

Before publishing, replace this synthetic data with either:

1. Output from the EEE side's Simulink IEEE 14-bus simulator, OR
2. A published real-world dataset (e.g. UCI Grid Stability, LBNL CERTS
   real microgrid traces, PJM operational data)

## Files

| File | Size | Purpose |
|---|---|---|
| `microgrid_synthetic_v1.csv`      | ~18 MB | Human-readable, git-friendly |
| `microgrid_synthetic_v1.parquet`  | ~6 MB  | Compressed, load 5-10× faster |

Both files contain the same 50,000 rows × 50 columns.

## Feature schema (50 columns)

**Time** (3)
- `timestamp`, `hour_of_day`, `day_of_year`

**Environment** (4)
- `irradiance_Wpm2`, `ambient_T_C`, `humidity_pct`, `wind_speed_mps`

**Voltage** (4)
- `V_rms_a_V`, `V_rms_b_V`, `V_rms_c_V`, `V_unbalance_pct`

**Current** (4)
- `I_rms_a_A`, `I_rms_b_A`, `I_rms_c_A`, `I_neutral_A`

**Power** (4)
- `P_active_kW`, `Q_reactive_kVAR`, `S_apparent_kVA`, `power_factor`

**Frequency** (3)
- `freq_Hz`, `freq_dev_Hz`, `RoCoF_Hz_per_s`

**Harmonics** (6)
- `V_THD_pct`, `I_THD_pct`, `harm_5th_pct`, `harm_7th_pct`,
  `harm_11th_pct`, `harm_13th_pct`

**Renewables** (3)
- `PV_kW`, `wind_kW`, `RE_penetration_pct`

**Storage** (6)
- `batt_SOC_pct`, `batt_SOH_pct`, `batt_V_V`, `batt_I_A`,
  `batt_T_C`, `batt_P_kW`

**Load** (2)
- `load_kW`, `nonlinear_load_frac`

**Mode flags** (5)
- `grid_connected`, `APF_on`, `ESS_on`, `AI_control_on`, `fault_flag`

**Labels** (3)
- `operating_scenario` (10 categories),
  `IEEE_519_compliance` (PASS/MARGINAL/FAIL),
  `stability_label` (0/1)

**Economics** (2)
- `energy_cost_BDT`, `cum_cost_BDT`

**Watermark** (1)
- `source = "SYNTHETIC_GENERATOR_v1"`

## Time span & sampling

- **Start:** 2026-06-01 00:00:00
- **End:**   2026-07-05 17:19:00 (~34 days)
- **Sample rate:** 1 sample per minute

## Class balance

| Label | Balance |
|---|---|
| Stability (unstable = 1) | ~15.9 % unstable, ~84.1 % stable |
| IEEE 519 compliance | ~81 % PASS, ~17 % MARGINAL, ~2 % FAIL |
| Operating scenario | 10 scenarios × 5,000 rows each |

## Physical relationships built in

The generator uses domain-informed correlations, not independent noise:

- **PV power** depends on irradiance + temperature (with NOCT derate)
- **Wind power** follows a cube law of wind speed, capped at rated
- **V-THD** grows with renewable penetration and nonlinear-load fraction,
  falls with ESS mitigation and AI control
- **I-THD** ≈ 1.6 × V-THD (typical distortion coupling)
- **Frequency deviation** grows with renewable variability, damped by
  storage and AI control
- **Battery temperature** rises with I-THD² (ohmic heating)
- **Battery SoH** decays slowly with time and faster with harmonic stress
- **IEEE 519 compliance** verdict derived from V-THD + I-THD thresholds
- **Stability label** derived from frequency, THD, unbalance, and faults

## How to regenerate

```bash
# from repo root
python src/generate_synthetic_dataset.py            # default 50,000 rows
python src/generate_synthetic_dataset.py 100000     # larger
python src/generate_synthetic_dataset.py 10000      # smaller for quick tests
```

Both `.csv` and `.parquet` files are overwritten.

## How to load

```python
import pandas as pd

# fastest
df = pd.read_parquet("data/synthetic/microgrid_synthetic_v1.parquet")

# or CSV (slower, larger)
df = pd.read_csv("data/synthetic/microgrid_synthetic_v1.csv",
                 parse_dates=["timestamp"])

# ALWAYS verify the watermark before using
assert (df["source"] == "SYNTHETIC_GENERATOR_v1").all(), \
    "This dataset is not the expected synthetic version"
```

## Distribution plots

See [`figures/synthetic/`](../../figures/synthetic/) for:

1. `01_vthd_by_scenario.png` — V-THD distribution across all 10 scenarios
2. `02_freq_vs_RE.png` — frequency deviation vs. renewable penetration
3. `03_daily_profile.png` — sample day showing load / PV / wind curves
4. `04_correlations.png` — correlation heatmap of key features

---

## What has been trained and tested on this dataset

Two downstream ML pipelines currently use this dataset. Their code lives
in [`src/`](../../src/) and their outputs live in
[`results/`](../../results/) and [`figures/`](../../figures/).

### 1. IEEE 519 compliance classifier (3-class)

**Script:** [`src/compliance_classifier.py`](../../src/compliance_classifier.py)
**Task:** Predict `IEEE_519_compliance` ∈ {PASS, MARGINAL, FAIL} from the
operational signals only (V-THD, I-THD, and individual harmonic
percentages are dropped from the feature set as they directly determine
the label). The classifier sees voltage, current, load, storage state,
weather, mode flags, and one-hot scenario.

**Setup:**
- 43 features after preprocessing (bools → int, categorical scenario
  one-hot-encoded)
- 80 / 20 stratified train-test split
- 5 classifiers trained: Logistic Regression, Random Forest, HistGB,
  XGBoost, MLP
- 5 random seeds per model (25 runs total)

**Metrics reported:** accuracy, macro-F1, per-class F1 for PASS /
MARGINAL / FAIL, and a confusion matrix on the best model.

**Headline results** (mean across 5 seeds, n_test = 10,000):

| Model | Accuracy | Macro-F1 | FAIL-F1 |
|---|---:|---:|---:|
| XGBoost | 99.82% ± 0.05% | 99.55% | 99.28% |
| Random Forest | 99.81% | 99.41% | 98.86% |
| HistGB | 99.79% | 99.36% | 98.79% |
| LogReg | 99.67% | 98.90% | 97.81% |
| MLP | 99.18% | 97.20% | 94.41% |

Full CSV: [`results/compliance_summary.csv`](../../results/compliance_summary.csv)
Plots: [`figures/compliance/`](../../figures/compliance/)

**Honest note:** these accuracies are unrealistically high because the
synthetic generator's physics is very clean. Once the EEE side's real
Simulink output replaces this data, expect the numbers to drop into
the 85–92 % range, which is a much better story for a paper.

### 2. Multi-horizon V-THD forecasting

**Script:** [`src/forecasting.py`](../../src/forecasting.py)
**Task:** Predict `V_THD_pct` at 5, 15, and 30 minutes in the future,
using the last 30 minutes of 26 operational signals (voltage, current,
power, frequency, RE, storage, load) plus the current V-THD value as
a lookback window.

**Setup:**
- Sliding window of length 30 → shape (49,940 × 30 × 27)
- **Chronological split** (no random shuffling — random splits would
  leak future information into training):
  - train = first 34,958 samples
  - val   = next 4,994
  - test  = last 9,988
- 4 models: Persistence baseline, MLP, LSTM (2-layer, 64 units),
  GRU (2-layer, 64 units)
- 20 epochs, Adam optimiser, MSE loss, early-stopping on val

**Metrics reported:** RMSE, MAE, R², and an early-warning F1 on the
binary event "V-THD will exceed the IEEE 519 5 % limit within the
next horizon".

**Headline results** (test set, 5-minute horizon):

| Model | RMSE | MAE | R² | Early-warning F1 |
|---|---:|---:|---:|---:|
| Persistence (naive) | 1.50 | 1.17 | -0.67 | 0.088 |
| MLP | 1.07 | 0.84 | +0.15 | 0.000 |
| LSTM | 1.06 | 0.83 | +0.16 | 0.000 |
| GRU | 1.07 | 0.83 | +0.16 | 0.000 |

Full CSV: [`results/forecasting_summary.csv`](../../results/forecasting_summary.csv)
Plots: [`figures/forecasting/`](../../figures/forecasting/) (truth-vs-forecast overlays at each horizon, RMSE bars, early-warning F1 bars, training curves)

**Honest note:** all three deep models beat the persistence baseline by
~30 % on RMSE — the pipeline extracts real signal. But early-warning
F1 collapses to 0 for the deep models because breach events are only
2 % of samples and MSE loss pushes predictions toward the mean. This
is a real methodological limitation of point-forecast + MSE on rare
events, not a bug. The next iteration reframes the task as binary
breach classification or class-weighted loss.

### Trained model checkpoints

Trained models are saved (but gitignored — regenerate by rerunning
the scripts):

- `models/compliance/*.pkl` — the 5 sklearn compliance classifiers,
  seed 0
- `models/forecasting/*.pt` — the 3 PyTorch forecasters (MLP, LSTM,
  GRU) with best-validation weights

## When to delete these files

Once the EEE side's Simulink IEEE 14-bus simulator is producing real
output CSVs, delete `microgrid_synthetic_v1.*` and update the ML pipeline
to point at the real files. The synthetic generator script
(`src/generate_synthetic_dataset.py`) can stay in the repo as a
utility for future pipeline testing.
