"""
Module: Anomaly Detection / Cyber-Resilience Layer (CSE)
---------------------------------------------------------
Uses Isolation Forest to flag rows that look like sensor faults or
false/corrupted data injection, per Section 5 Module 3 of the proposal.

Important scoping note (see project data_provenance_quality_note.docx,
Section 3.1): this dataset already ships with a sensor_fault_flag ground
truth column with only 210/5000 (4.2%) positive rows, and the underlying
signals are unusually clean. That means:
    - Isolation Forest's contamination parameter should be set close to the
      true fault rate (~4.2%) rather than guessed, or evaluated across a
      small sweep, since the model has no way to "learn" a rate on its own.
    - Reported precision/recall here is likely optimistic relative to a
      messier, real deployment feed - state that caveat wherever these
      numbers are reported (Section 7 metrics).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from signal_processing import load_dataset, engineer_features

FEATURE_COLUMNS = [
    "voltage_rms_V", "current_rms_A", "freq_deviation_Hz",
    "harmonic_sum_pct", "THD_voltage_pct", "THD_current_pct",
    "thd_v_to_i_ratio",
]


def fit_anomaly_detector(df: pd.DataFrame, contamination: float = 0.042, random_state: int = 42):
    """
    Fit an Isolation Forest on electrical/signal features to flag anomalous
    (potentially faulty or spoofed) sensor readings.

    Returns the fitted scaler, model, and the dataframe with an added
    'anomaly_score' (higher = more anomalous) and 'predicted_fault' column.
    """
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    raw_scores = -model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

    out = df.copy()
    out["anomaly_score"] = raw_scores
    out["predicted_fault"] = (predictions == -1).astype(int)

    return scaler, model, out


def evaluate_against_ground_truth(df_with_predictions: pd.DataFrame):
    """Compare predicted_fault against the dataset's sensor_fault_flag ground truth."""
    y_true = df_with_predictions["sensor_fault_flag"]
    y_pred = df_with_predictions["predicted_fault"]
    y_score = df_with_predictions["anomaly_score"]

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=["normal", "fault"]))
    print()
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC (using anomaly_score): {auc:.4f}")
    except ValueError as e:
        print(f"Could not compute ROC-AUC: {e}")


if __name__ == "__main__":
    df = load_dataset("/mnt/user-data/uploads/microgrid_power_quality_dataset.csv")
    df = engineer_features(df)

    true_fault_rate = df["sensor_fault_flag"].mean()
    print(f"True sensor fault rate in data: {true_fault_rate:.4f}")

    scaler, model, df_pred = fit_anomaly_detector(df, contamination=true_fault_rate)
    evaluate_against_ground_truth(df_pred)

    df_pred.to_csv("/home/user/pipeline/features_stage2_anomaly.csv", index=False)
    print("Saved features_stage2_anomaly.csv")
