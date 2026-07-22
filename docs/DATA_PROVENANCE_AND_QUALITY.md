# Data Provenance & Quality Note

Original note by a previous analysis pass, preserved verbatim below,
plus an addendum documenting what was found while integrating Dataset A
into this repo (`src/anomaly_detection.py`, `src/disturbance_classifier.py`,
`src/explainability.py`, `src/supervised_fault_check.py`).

Source `.docx` and the two original reference scripts (`anomaly_detection.py`,
`signal_processing.py`) are kept alongside this file for the record.

---

## Addendum — findings from repo integration

**1. The supporting scripts confirm an AI-analysis-tool origin, not a
teammate's local pipeline.**
`docs/reference_anomaly_detection_original.py` and
`docs/reference_signal_processing_original.py` (found in `~/Downloads/`,
same place as this note) hardcode
`/mnt/user-data/uploads/microgrid_power_quality_dataset.csv` as their
input path — a sandbox-upload path convention characteristic of a
hosted AI code-execution/analysis tool, not a local development
machine. This doesn't prove the *dataset itself* was fabricated by AI,
but it does mean the claim "field data from Jamalpur" has not been
independently substantiated by anything found on this machine — the
earliest artifact we can find is already an AI-assisted analysis
pipeline, not a data-collection record.

**2. New finding not in the original note: the cost/battery columns are
formulaically derived, not independently measured.**

```
economic_cost_BDT / battery_degradation_rate  →  mean 1499.99, std 1.14
corr(economic_cost_BDT, battery_degradation_rate) = 0.99999998
corr(battery_capacity_loss_pct, row_index)          = 0.9991604
```

`economic_cost_BDT` is `battery_degradation_rate × ~1500` for essentially
every row, with no independent noise term — two real physical quantities
computed from different processes (energy pricing vs. electrochemical
aging) would not correlate at 0.99999998. `battery_capacity_loss_pct`
correlates 0.999 with plain row order, i.e. it behaves like
`row_number × 0.00028`, not a value that responds to which disturbance
occurred in that row. `battery_degradation_rate` itself *does* correlate
sensibly with `voltage_rms_V` (r=-0.51) and `THD_voltage_pct` (r=0.34),
so that one column has a defensible physical basis — it's specifically
the two *derived* columns downstream of it that show the formulaic
fingerprint.

**Practical implication:** don't build the proposal's "digital twin cost
mapping" module as if `economic_cost_BDT` were an independent ground
truth to validate against — it's a fixed linear transform of a column
already in the dataset, so any model "predicting" it will trivially hit
~1.00 R² by discovering that formula, not by learning a cost model. Worth
re-deriving cost from THD/degradation using an explicit, published
tariff-plus-replacement-cost formula instead, and treating it as a
transformation stage in the twin, not a target label to fit.

**3. Reproduced the note's Section 6 comparison independently
(`src/supervised_fault_check.py`)**, on a different train/test split:

| Method | Precision (fault) | Recall (fault) | F1 | ROC-AUC |
|---|---|---|---|---|
| Supervised RF (labels used) | 1.00 | 0.46 | 0.63 | 0.82 |
| Unsupervised IsolationForest (auto threshold) | 0.038 | 0.23 | 0.066 | 0.70 |
| Unsupervised IsolationForest (CV-tuned threshold) | 0.057 | 0.64 | 0.10 | 0.70 |

The note reported 1.00 precision / 0.50 recall for the supervised case
on its own split — this run got 1.00 / 0.46, a close independent
replication. This is a real, stable property of the dataset (the
fault signal is learnable given labels, much less so without them), not
an artifact of one lucky split.

**4. "Dataset B" naming collision.** This note's "Dataset B" is a HOMER
Pro/HOMER Grid hourly simulation export (`hourly_CSV.csv`,
`hourly_CSV-2.csv`, 8,760 rows × 44 cols) — **searched for and not found
anywhere on this machine.** It is a *different file* from
`data/synthetic/microgrid_synthetic_v1.csv` in this same repo (a
custom 50-column three-phase generator, documented in
`src/generate_synthetic_dataset.py`, already used elsewhere in this
project for the IEEE-519 compliance classifier). Keep these two straight
in the manuscript — recommend calling this repo's dataset
"Dataset C" or "the synthetic benchmark set" if the HOMER file is
recovered later, to avoid two different things both being called
"Dataset B."

**5. Recommendation, restated plainly:** of the note's two Section 7
options — (a) chase down a handoff artifact from the teammate proving
the Jamalpur origin, or (b) reclassify Dataset A as synthetic — go with
**(b)**. Nothing found here supports (a), the statistical fingerprints
in both the original note and this addendum all point the same
direction, and the note's own suggested manuscript language already
gives you a way to write this up that is defensible with a reviewer.
Reclassifying costs nothing structurally: the proposal's own Module 1
already describes "synthetic voltage/current signals with fault
scenarios," so no methodology section needs to change, only the
provenance claim.

---

## Original note (verbatim)

### 1. Purpose of This Note

This note documents the origin, structure, and quality characteristics
of the two datasets used to support the proposed framework: (1) a
power-quality disturbance dataset (voltage/current/harmonics/battery/cost
fields), and (2) an hourly renewable-microgrid simulation dataset
(HOMER-style output). It is intended to be transparent about what is
known, what is assumed, and what should be verified before these
datasets are described in a publication.

### 2. Dataset Summary

| Attribute | Dataset A: Power Quality | Dataset B: Hourly Simulation | Dataset B-subset: Selected Columns |
|---|---|---|---|
| File name | microgrid_power_quality_dataset.csv | hourly_CSV.csv | hourly_CSV-2.csv |
| Rows | 5,000 | 8,760 (1 year, hourly) | 8,760 |
| Columns | 16 | 44 | 10 (subset) |
| Stated origin | Reported by a team member as field data collected from an industrial site in Jamalpur. | Consistent with HOMER Pro / HOMER Grid microgrid simulation software output. | Derived from Dataset B. |
| Verification status | Not independently verified. No collection protocol, instrumentation record, or site documentation available at time of writing. | Structurally consistent with a known simulation tool; treated as simulated, not field-measured, data. | Same as Dataset B. |

### 3. Data Quality Observations — Dataset A

The following characteristics were observed during profiling and should
be disclosed regardless of the dataset's final documented origin, since
they affect how results should be interpreted:

**3.1 Unusually low variability and completeness**
- Zero missing values across all 5,000 rows for every electrical,
  environmental, and cost field.
- Zero duplicate rows.
- Voltage (mean 219.3 V) and frequency (mean 49.999 Hz, std 0.03 Hz) are
  tightly clustered around nominal values with smooth, low-noise
  variation.
- No gaps, timestamp irregularities, sensor dropouts, or communication
  losses of the kind typically present in industrial SCADA or
  data-logger exports.

Field-collected industrial power-quality logs generally contain some
combination of missing intervals, sensor drift, duplicate polling
artifacts, and irregular sampling. Their near-total absence here is
atypical for raw field telemetry and is worth stating explicitly rather
than implying the data was collected exactly as-is from live equipment.

**3.2 Category structure mirrors the proposal's own test-scenario design**

The `disturbance_type` field contains exactly three categories:
Voltage_Sag (564 rows), Combined_Weather_Electrical (528 rows), and
Harmonic_Distortion (467 rows), alongside 3,441 rows with no disturbance
(label 0). These three categories correspond closely to the three test
scenarios already defined in Section 7 of the proposal (harmonic
distortion from nonlinear loads; voltage sag from renewable fluctuation;
combined weather-electrical disturbances). This alignment is worth
noting: it may reflect that the dataset was purpose-built to match the
proposal's experimental design, which is a reasonable and common
practice for framework validation, but is a different claim than the
dataset being an unfiltered export from an operating industrial site.

**3.3 Clean separability between classes**

THD_voltage_pct differs sharply by disturbance_label: disturbance-free
rows average 2.47% (std 0.53), while disturbance rows average 6.95%
(std 4.25), with comparatively little overlap between the two
distributions. Clean class separability is desirable for model training
and is not itself evidence of any particular origin, but it is worth
noting as a factor that will make prediction accuracy look strong during
evaluation on this data — a caveat relevant to interpreting Section 7's
accuracy metrics.

**3.4 Formatting note (non-substantive)**

The `disturbance_type` field is blank (read as null by standard tools)
rather than containing the literal string "None" for the 3,441
non-disturbance rows. This is a labeling/export convention, not missing
telemetry — the corresponding `disturbance_label` field is populated (0)
for every one of these rows.

*(Repo-integration note: this convention is exactly what broke
`disturbance_classifier.py` on first run — pandas' default `na_values`
list treats the literal string `"None"` as missing too, so reading the
CSV naively drops the real category. Fixed with an explicit
`fillna("None")` after load; see `src/disturbance_classifier.py` and
`src/supervised_fault_check.py` / `src/anomaly_detection.py` for the
same handling.)*

### 4. Data Quality Observations — Dataset B (Hourly Simulation)

- Column structure (BESS state-of-charge, inverter/rectifier flows,
  fuel-cell and electrolyzer/hydrogen fields, renewable penetration)
  matches the standard output schema of HOMER-family microgrid
  simulation tools.
- The file begins with a literal `sep=,` line (an Excel locale artifact)
  that must be skipped during parsing.
- Data covers a full calendar year at hourly resolution (8,760 rows),
  consistent with a complete annual simulation run rather than a partial
  or field-logged extract.
- This dataset does not contain harmonics, THD, or any power-quality
  disturbance fields — HOMER-family tools model energy balance and
  economics, not electromagnetic transient or harmonic behavior. It
  cannot substitute for Dataset A's disturbance data and should be
  described only as system-level generation/load/battery context.

### 5. Recommendation for the Manuscript

Regardless of which working assumption is used going forward, the
following language pattern is suggested for the data section of the
paper, since it is accurate under either origin story:

> "The power quality dataset used in this study was obtained from a
> project team member and reported to originate from an industrial site
> in Jamalpur; the original collection protocol and instrumentation were
> not independently available to the authors. The dataset exhibits
> notably low noise and complete coverage relative to typical field
> telemetry, which is disclosed here as a limitation. Results should be
> interpreted with this caveat, and validation against an independently
> sourced or synthetically generated dataset with explicit ground truth
> is recommended as future work."

This framing is honest about uncertainty without asserting the data is
fabricated, and it protects the paper from a reviewer later challenging
an unqualified "real industrial field data" claim.

### 6. Pipeline Findings That Reinforce the Above (Signal Processing / Anomaly Detection / Prediction runs)

Initial runs of the Module 2–4 pipeline against Dataset A produced two
results that should be read together with the data quality notes above,
since they point the same direction:

- Isolation Forest (unsupervised anomaly detection, Module 3) achieved
  only ~0.06 precision/recall on the sensor_fault_flag ground truth,
  despite a ROC-AUC of 0.73. A supervised Random Forest trained on the
  same features and given access to labels reached 1.00 precision /
  0.50 recall on the same fault class. This gap indicates the fault
  signal is present in the electrical features but is not structured as
  a natural multivariate outlier — worth stating as a methodological
  finding rather than a shortfall, and worth reconsidering whether
  Isolation Forest is the right unsupervised choice for this fault type.
- The disturbance prediction model (Module 4, Random Forest baseline)
  achieved 1.00 precision/recall/F1 and ROC-AUC of 0.9999 on held-out
  test data, with voltage_rms_V alone carrying roughly 50% of feature
  importance. Perfect or near-perfect separability on a multi-class
  power disturbance problem is unusual for field-realistic data and is
  consistent with disturbance_label having been assigned via a rule
  applied directly to these same features during dataset construction,
  rather than reflecting the natural difficulty of real-world
  prediction. This should be disclosed as a limitation, and validation
  against a second, independently-labeled dataset is recommended before
  reporting these numbers as representative performance.

### 7. Open Items

- Confirm with the teammate whether any handoff artifact exists (email,
  file transfer log, shared drive link) naming the source plant or
  contact.
- If no such artifact exists, consider formally re-classifying Dataset A
  as "synthetic, generated to reflect field-realistic power quality
  disturbance scenarios" — which fits the proposal's Module 1
  description without requiring any changes to downstream code.
- Decide before submission which classification will appear in the
  manuscript, and keep it consistent across the abstract, methodology,
  and data availability statement.
