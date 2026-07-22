"""
Power-Quality Disturbance Prediction Model (4-class)
=====================================================

Implements the "Prediction Model (CSE)" module from the Cyber-Resilient,
Explainable Digital Twin proposal: predicts disturbance_type from
electrical + weather features (dual-input, per the proposal's Module 4).

    Classes: None, Voltage_Sag, Harmonic_Distortion,
             Combined_Weather_Electrical

    Features (electrical): voltage_rms_V, current_rms_A, frequency_Hz,
        harmonic_3rd_pct, harmonic_5th_pct, harmonic_7th_pct,
        THD_voltage_pct, THD_current_pct
    Features (weather): temperature_C, irradiance_Wm2

Deliberately excluded from features:
    - sensor_fault_flag       (cyber layer's own output, not a PQ cause)
    - disturbance_label       (binary collapse of the target itself)
    - battery_degradation_rate, battery_capacity_loss_pct,
      economic_cost_BDT       (downstream consequences computed BY the
                                digital-twin module from the disturbance,
                                not causal inputs to predicting it)

>>>>>  WARNING - trained on SYNTHETIC data (Dataset A, 5,000 rows).
       Pipeline validation only.

Run:
    python src/disturbance_classifier.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: we only save PNGs, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                              classification_report, confusion_matrix,
                              f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "external" / "microgrid_power_quality_dataset.csv"
MODEL_DIR = ROOT / "models" / "disturbance"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "disturbance"; FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 42, 123, 2026]

FEATURE_COLS = [
    "voltage_rms_V", "current_rms_A", "frequency_Hz",
    "temperature_C", "irradiance_Wm2",
    "harmonic_3rd_pct", "harmonic_5th_pct", "harmonic_7th_pct",
    "THD_voltage_pct", "THD_current_pct",
]
TARGET_COL = "disturbance_type"

MODELS = {
    "LogReg": lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ]),
    "RF": lambda seed: RandomForestClassifier(
        n_estimators=300, random_state=seed, n_jobs=-1),
    "HistGB": lambda seed: HistGradientBoostingClassifier(random_state=seed),
    "XGBoost": lambda seed: XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=seed, n_jobs=-1),
    "MLP": lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                               max_iter=1000, random_state=seed)),
    ]),
}


def load_data() -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = pd.read_csv(DATA_PATH)
    # pandas' default na_values list includes the literal string "None",
    # which is a real category here (no disturbance), not a missing value.
    df[TARGET_COL] = df[TARGET_COL].fillna("None")
    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    return X, y, le


def run() -> pd.DataFrame:
    X, y, le = load_data()
    rows = []
    best_acc = -1.0
    best_model = None
    best_name = None
    best_split = None

    for name, factory in MODELS.items():
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y)
            model = factory(seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            macro_f1 = f1_score(y_te, y_pred, average="macro")
            rows.append({
                "model": name, "seed": seed,
                "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            })

            if seed == SEEDS[0] and acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name
                best_split = (X_te, y_te, y_pred)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "disturbance_per_seed.csv", index=False)

    summary = (results_df.groupby("model")[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(RESULTS_DIR / "disturbance_summary.csv")
    print(summary)

    # Save best model (by seed-0 accuracy, matching repo convention)
    with open(MODEL_DIR / f"{best_name}_seed0.pkl", "wb") as f:
        pickle.dump({"model": best_model, "label_encoder": le,
                     "features": FEATURE_COLS}, f)

    # Confusion matrix + per-class report for the best model
    X_te, y_te, y_pred = best_split
    print(f"\nBest model: {best_name} (seed {SEEDS[0]}, acc={best_acc:.4f})")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(f"Disturbance classifier confusion matrix\n"
                 f"({best_name}, test acc={best_acc:.3f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Accuracy-per-model bar chart across seeds
    fig, ax = plt.subplots(figsize=(6, 4))
    means = results_df.groupby("model")["accuracy"].mean().sort_values()
    stds = results_df.groupby("model")["accuracy"].std().reindex(means.index)
    ax.barh(means.index, means.values, xerr=stds.values, color="#2a9d8f")
    ax.set_xlabel("Test accuracy (mean ± std over 5 seeds)")
    ax.set_title("Disturbance prediction: accuracy per model")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_accuracy_per_model.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {RESULTS_DIR / 'disturbance_summary.csv'}")
    print(f"Saved: {MODEL_DIR / f'{best_name}_seed0.pkl'}")
    print(f"Saved: {FIG_DIR / '01_confusion_matrix.png'}")
    print(f"Saved: {FIG_DIR / '02_accuracy_per_model.png'}")

    return results_df


if __name__ == "__main__":
    run()
