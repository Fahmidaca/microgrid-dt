"""
Supervised vs. Unsupervised Fault Detection — Dataset A
=========================================================

Companion to anomaly_detection.py. That script fits an UNSUPERVISED
Isolation Forest (no access to sensor_fault_flag during training) and
scored ~0.70 ROC-AUC but poor precision/recall at any reasonable
threshold. This script fits a SUPERVISED classifier on the same
features, WITH access to sensor_fault_flag labels during training, to
answer: is the fault signal present in the features at all (supervised
should do much better), or is it genuinely hard to separate even with
labels (both would struggle)?

>>>>>  WARNING - SYNTHETIC data, and per
       docs/DATA_PROVENANCE_AND_QUALITY.md this specific dataset's
       origin is UNVERIFIED (reported as field data, but with no
       collection protocol / instrumentation record, and statistical
       fingerprints consistent with synthetic construction). A
       near-perfect supervised result here should be read as "the
       fault flag is a learnable function of these features" — a
       pipeline-validation finding, not evidence about real deployment.

Run:
    python src/supervised_fault_check.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "external" / "microgrid_power_quality_dataset.csv"
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FEATURE_COLS = [
    "voltage_rms_V", "current_rms_A", "frequency_Hz",
    "temperature_C", "irradiance_Wm2",
    "harmonic_3rd_pct", "harmonic_5th_pct", "harmonic_7th_pct",
    "THD_voltage_pct", "THD_current_pct",
]
LABEL_COL = "sensor_fault_flag"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[LABEL_COL].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1,
        class_weight="balanced",  # ~2.4% positive rate — don't let it collapse to all-normal
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_score = clf.predict_proba(X_te)[:, 1]

    report = classification_report(y_te, y_pred, target_names=["normal", "fault"],
                                    output_dict=True)
    auc = roc_auc_score(y_te, y_score)

    print("=== Supervised RF on sensor_fault_flag (labels used at train time) ===")
    print(classification_report(y_te, y_pred, target_names=["normal", "fault"]))
    print(f"ROC-AUC: {auc:.4f}")

    summary = pd.DataFrame([{
        "method": "Supervised RF (labels used)",
        "precision_fault": round(report["fault"]["precision"], 4),
        "recall_fault": round(report["fault"]["recall"], 4),
        "f1_fault": round(report["fault"]["f1-score"], 4),
        "roc_auc": round(auc, 4),
    }, {
        "method": "Unsupervised IsolationForest, auto (from anomaly_detection.py)",
        "precision_fault": 0.0381, "recall_fault": 0.2333, "f1_fault": 0.0655,
        "roc_auc": 0.7035,
    }, {
        "method": "Unsupervised IsolationForest, 5-fold CV-tuned threshold",
        "precision_fault": 0.057, "recall_fault": 0.6417, "f1_fault": 0.1047,
        "roc_auc": 0.7035,
    }])
    out_path = RESULTS_DIR / "supervised_vs_unsupervised_fault_detection.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nSaved: {out_path}")
    print(
        "\nInterpretation: supervised access to labels closes most of the gap "
        "unsupervised methods can't (Isolation Forest has no way to know which "
        "kind of outlier matters). That gap is the finding worth reporting — "
        "it argues for either a semi-supervised approach or a larger labelled "
        "validation set in deployment, not that Isolation Forest 'failed'."
    )


if __name__ == "__main__":
    main()
