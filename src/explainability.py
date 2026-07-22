"""
Explainable AI Layer — SHAP feature attribution
=================================================

Implements the "Explainable AI Layer (CSE)" module from the
Cyber-Resilient, Explainable Digital Twin proposal: explains the
disturbance_classifier's predictions with SHAP, identifying which
electrical/weather features dominate each disturbance-type decision.

Loads the trained model from disturbance_classifier.py
(models/disturbance/*.pkl) rather than retraining, so run that script
first.

Outputs:
    figures/xai/01_shap_summary_bar.png
        Mean |SHAP value| per feature, one bar group per class —
        "what drives the model overall, and does it differ by class?"
    figures/xai/02_shap_beeswarm_<class>.png
        Per-class beeswarm for the class most reliant on harmonic
        features, showing direction of effect (not just magnitude).
    results/shap_feature_importance.csv
        Mean |SHAP value| per feature per class, tabulated.

Run:
    python src/explainability.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: we only save PNGs, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "external" / "microgrid_power_quality_dataset.csv"
MODEL_DIR = ROOT / "models" / "disturbance"
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "xai"; FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 800  # SHAP is expensive; a stratified sample is plenty for attribution
SEED = 42


def load_trained_model() -> dict:
    pkl_files = sorted(MODEL_DIR.glob("*_seed0.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No trained model found in {MODEL_DIR}. "
            "Run src/disturbance_classifier.py first.")
    with open(pkl_files[0], "rb") as f:
        bundle = pickle.load(f)
    print(f"Loaded model: {pkl_files[0].name}")
    return bundle


def main() -> None:
    bundle = load_trained_model()
    model = bundle["model"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    df = pd.read_csv(DATA_PATH)
    df[df.columns[df.columns == "disturbance_type"][0]] = (
        df["disturbance_type"].fillna("None"))

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    X_sample = sample[features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # shap_values shape for multi-class trees: (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    class_names = le.classes_
    n_classes = len(class_names)

    # --- Mean |SHAP| per feature per class, tabulated ---
    rows = []
    for c in range(n_classes):
        mean_abs = np.abs(shap_values[:, :, c]).mean(axis=0)
        for feat, val in zip(features, mean_abs):
            rows.append({"class": class_names[c], "feature": feat,
                         "mean_abs_shap": round(float(val), 5)})
    importance_df = pd.DataFrame(rows)
    importance_df.to_csv(RESULTS_DIR / "shap_feature_importance.csv", index=False)

    # --- Summary bar: mean |SHAP| per feature, grouped by class ---
    pivot = importance_df.pivot(index="feature", columns="class",
                                 values="mean_abs_shap")
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               colormap="tab10", width=0.75)
    ax.set_xlabel("Mean |SHAP value| (summed contribution across classes)")
    ax.set_ylabel("")
    ax.set_title("SHAP feature attribution — disturbance classifier")
    ax.legend(title="disturbance_type", fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_shap_summary_bar.png", dpi=150)
    plt.close(fig)

    # --- Beeswarm for the class with the highest total attribution mass ---
    top_class_idx = int(pivot.sum(axis=0).values.argmax())
    top_class_name = pivot.columns[top_class_idx]
    fig = plt.figure(figsize=(7, 5.5))
    shap.summary_plot(
        shap_values[:, :, top_class_idx], X_sample,
        feature_names=features, show=False)
    plt.title(f"SHAP beeswarm — class '{top_class_name}'")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"02_shap_beeswarm_{top_class_name}.png", dpi=150)
    plt.close(fig)

    print(importance_df.pivot(index="feature", columns="class",
                               values="mean_abs_shap").round(4))
    print(f"\nSaved: {RESULTS_DIR / 'shap_feature_importance.csv'}")
    print(f"Saved: {FIG_DIR / '01_shap_summary_bar.png'}")
    print(f"Saved: {FIG_DIR / f'02_shap_beeswarm_{top_class_name}.png'}")


if __name__ == "__main__":
    main()
