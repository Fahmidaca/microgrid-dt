"""
Cyber-Resilience Anomaly Detection — Isolation Forest
======================================================

Implements the "Anomaly Detection (CSE)" module from the
Cyber-Resilient, Explainable Digital Twin proposal: an unsupervised
sensor-fault / false-data-injection detector, evaluated against two
independently generated datasets rather than a forced merge of the two
(their schemas are not compatible — Dataset A is single-phase/16 cols,
Dataset B is three-phase/50 cols).

    Dataset A: data/external/microgrid_power_quality_dataset.csv
               5,000 rows. Ground truth: sensor_fault_flag (2.4% positive).
    Dataset B: data/synthetic/microgrid_synthetic_v1.csv
               50,000 rows. Ground truth: fault_flag (0.4% positive,
               injected as `rng.random(n) < 0.005` in the generator).

>>>>>  WARNING - both datasets are SYNTHETIC. This validates the
       anomaly-detection pipeline end-to-end; it is not evidence the
       method will catch real false-data-injection attacks. Dataset B's
       fault_flag in particular is close to pure random noise by
       construction, which caps how learnable it can be from features.

For each dataset:
    1. Select continuous power-quality features only (drop labels,
       derived/leakage columns, and identifiers).
    2. Fit IsolationForest with contamination matched to the dataset's
       true positive rate (best case for the algorithm) AND with
       contamination='auto' (the realistic case, where the true rate
       is unknown in deployment).
    3. Score against ground truth: precision, recall, F1, ROC-AUC.

Run:
    python src/anomaly_detection.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: we only save PNGs, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_recall_curve,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "anomaly"; FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

DATASET_A_PATH = ROOT / "data" / "external" / "microgrid_power_quality_dataset.csv"
DATASET_B_PATH = ROOT / "data" / "synthetic" / "microgrid_synthetic_v1.csv"

DATASET_A_FEATURES = [
    "voltage_rms_V", "current_rms_A", "frequency_Hz",
    "temperature_C", "irradiance_Wm2",
    "harmonic_3rd_pct", "harmonic_5th_pct", "harmonic_7th_pct",
    "THD_voltage_pct", "THD_current_pct",
]
DATASET_A_LABEL = "sensor_fault_flag"

DATASET_B_FEATURES = [
    "V_rms_a_V", "V_rms_b_V", "V_rms_c_V", "V_unbalance_pct",
    "I_rms_a_A", "I_rms_b_A", "I_rms_c_A", "I_neutral_A",
    "P_active_kW", "Q_reactive_kVAR", "S_apparent_kVA", "power_factor",
    "freq_Hz", "freq_dev_Hz", "RoCoF_Hz_per_s",
    "V_THD_pct", "I_THD_pct",
    "harm_5th_pct", "harm_7th_pct", "harm_11th_pct", "harm_13th_pct",
    "batt_SOC_pct", "batt_V_V", "batt_I_A", "batt_T_C",
]
DATASET_B_LABEL = "fault_flag"


def evaluate_isolation_forest(
    df: pd.DataFrame, feature_cols: list[str], label_col: str,
    dataset_name: str,
) -> list[dict]:
    X = df[feature_cols].to_numpy(dtype=float)
    y_true = df[label_col].astype(int).to_numpy()
    true_rate = y_true.mean()

    Xs = StandardScaler().fit_transform(X)

    rows = []
    for contamination_label, contamination in [
        ("matched_to_true_rate", max(true_rate, 1e-4)),
        ("auto", "auto"),
    ]:
        clf = IsolationForest(
            n_estimators=300, contamination=contamination,
            random_state=SEED, n_jobs=-1,
        )
        clf.fit(Xs)
        pred = (clf.predict(Xs) == -1).astype(int)  # -1 = anomaly
        scores = -clf.decision_function(Xs)  # higher = more anomalous

        rows.append({
            "dataset": dataset_name,
            "n_rows": len(df),
            "true_fault_rate_pct": round(true_rate * 100, 3),
            "contamination_setting": contamination_label,
            "precision": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, scores), 4),
        })
    return rows


def tune_threshold_cv(
    df: pd.DataFrame, feature_cols: list[str], label_col: str,
    dataset_name: str, n_folds: int = 5,
) -> dict:
    """Isolation Forest gives a ranking (ROC-AUC), not an operating point.
    contamination='auto'/matched-rate picks a threshold blind to labels.
    Here we keep the detector itself unsupervised (fit uses no labels) but
    calibrate the DECISION THRESHOLD with 5-fold CV over the labelled
    anomaly scores: pick the F1-optimal cut on the train folds, apply it
    to the held-out fold. This is standard practice when a small labelled
    validation set exists for an otherwise-unsupervised detector, and it
    avoids leaking the test fold's own labels into its threshold.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].astype(int).to_numpy()
    Xs = StandardScaler().fit_transform(X)

    clf = IsolationForest(n_estimators=300, contamination="auto",
                           random_state=SEED, n_jobs=-1)
    clf.fit(Xs)
    scores = -clf.decision_function(Xs)  # higher = more anomalous

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    for train_idx, test_idx in skf.split(Xs, y):
        prec, rec, thr = precision_recall_curve(y[train_idx], scores[train_idx])
        f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
        best_thr = thr[np.argmax(f1s[:-1])] if len(thr) else 0.0

        pred_test = (scores[test_idx] >= best_thr).astype(int)
        fold_metrics.append({
            "precision": precision_score(y[test_idx], pred_test, zero_division=0),
            "recall": recall_score(y[test_idx], pred_test, zero_division=0),
            "f1": f1_score(y[test_idx], pred_test, zero_division=0),
        })

    fold_df = pd.DataFrame(fold_metrics)

    # Oracle upper bound: best F1 achievable on THIS score ranking if the
    # threshold were chosen with full knowledge of all labels (leaky by
    # design — reported only as a ceiling, not a deployable number).
    prec, rec, thr = precision_recall_curve(y, scores)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    oracle_f1 = float(np.max(f1s))

    return {
        "dataset": dataset_name,
        "cv_precision_mean": round(fold_df["precision"].mean(), 4),
        "cv_precision_std": round(fold_df["precision"].std(), 4),
        "cv_recall_mean": round(fold_df["recall"].mean(), 4),
        "cv_recall_std": round(fold_df["recall"].std(), 4),
        "cv_f1_mean": round(fold_df["f1"].mean(), 4),
        "cv_f1_std": round(fold_df["f1"].std(), 4),
        "oracle_f1_upper_bound": round(oracle_f1, 4),
    }


def plot_comparison(results: pd.DataFrame) -> None:
    matched = results[results["contamination_setting"] == "matched_to_true_rate"]
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (_, row) in enumerate(matched.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics], width,
               label=f"{row['dataset']} (n={row['n_rows']:,}, "
                     f"fault rate={row['true_fault_rate_pct']}%)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Isolation Forest anomaly detection: Dataset A vs Dataset B\n"
                  "(contamination matched to true fault rate)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_precision_recall_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df_a = pd.read_csv(DATASET_A_PATH)
    df_b = pd.read_csv(DATASET_B_PATH)

    results = []
    results += evaluate_isolation_forest(
        df_a, DATASET_A_FEATURES, DATASET_A_LABEL, "Dataset A (disturbance CSV)")
    results += evaluate_isolation_forest(
        df_b, DATASET_B_FEATURES, DATASET_B_LABEL, "Dataset B (microgrid_synthetic_v1)")

    results_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "anomaly_detection_summary.csv"
    results_df.to_csv(out_path, index=False)

    plot_comparison(results_df)

    print(results_df.to_string(index=False))
    print(f"\nSaved: {out_path}")
    print(f"Saved: {FIG_DIR / '01_precision_recall_comparison.png'}")

    # --- Threshold tuning: does calibrating the operating point help? ---
    tuning_rows = [
        tune_threshold_cv(df_a, DATASET_A_FEATURES, DATASET_A_LABEL,
                           "Dataset A (disturbance CSV)"),
        tune_threshold_cv(df_b, DATASET_B_FEATURES, DATASET_B_LABEL,
                           "Dataset B (microgrid_synthetic_v1)"),
    ]
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_out = RESULTS_DIR / "anomaly_threshold_tuning.csv"
    tuning_df.to_csv(tuning_out, index=False)

    print("\n--- 5-fold CV threshold tuning (label-informed cut, unsupervised score) ---")
    print(tuning_df.to_string(index=False))
    print(f"\nSaved: {tuning_out}")


if __name__ == "__main__":
    main()
