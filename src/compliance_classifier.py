"""
IEEE 519 Compliance Classifier (3-class: PASS / MARGINAL / FAIL)
================================================================

Predicts the IEEE 519 compliance verdict from OPERATIONAL SIGNALS ALONE,
without using THD or harmonic amplitudes as features. This is the useful
prediction problem: given voltage, current, load, storage state, and
weather, can the model tell an operator whether the grid is about to
violate IEEE 519 before a THD analyzer confirms it?

>>>>>  WARNING - trained on SYNTHETIC data. Results are pipeline
       validation, NOT publishable evidence. Replace with real Simulink
       output from the IEEE 14-bus model before publication.

Pipeline:
    1. Load synthetic dataset, verify watermark.
    2. Drop THD-leakage columns from feature set.
    3. One-hot encode categorical columns.
    4. Train 5 classifiers x 5 seeds with deterministic 80/20 stratified split.
    5. Report accuracy + macro-F1 + per-class F1 + confusion matrix.

Run:
    python src/compliance_classifier.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "microgrid_synthetic_v1.parquet"
MODEL_DIR = ROOT / "models" / "compliance"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "compliance"; FIG_DIR.mkdir(parents=True, exist_ok=True)


SEEDS = [0, 1, 42, 123, 2026]

# Columns that DIRECTLY REVEAL the target and must be dropped from features.
# THD, individual harmonic percentages, and IEEE 519 compliance itself.
LEAKAGE_COLS = [
    "V_THD_pct", "I_THD_pct",
    "harm_5th_pct", "harm_7th_pct", "harm_11th_pct", "harm_13th_pct",
    "stability_label",     # correlated auxiliary label
]

DROP_COLS = [
    "timestamp",           # raw datetime, not directly modellable
    "source",              # watermark string
    "IEEE_519_compliance", # target
] + LEAKAGE_COLS


def load_dataset() -> pd.DataFrame:
    print(f"[data] loading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    assert (df["source"] == "SYNTHETIC_GENERATOR_v1").all(), \
        "watermark check failed - this is not the expected synthetic dataset"
    print(f"[data] {len(df):,} rows x {df.shape[1]} cols loaded")
    print(f"[data] target class balance:")
    print(df["IEEE_519_compliance"].value_counts().to_string())
    return df


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y_int, feature_names). Categorical -> one-hot; bools -> int."""
    X_df = df.drop(columns=DROP_COLS).copy()

    # bool -> int
    for c in X_df.select_dtypes(include=["bool"]).columns:
        X_df[c] = X_df[c].astype(int)

    # object (string) -> one-hot
    obj_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        X_df = pd.get_dummies(X_df, columns=obj_cols, drop_first=False)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    y_labels = df["IEEE_519_compliance"].values
    le = LabelEncoder()
    y = le.fit_transform(y_labels)   # FAIL=0, MARGINAL=1, PASS=2 (alphabetical)

    print(f"[features] {X.shape[1]} features after preprocessing")
    print(f"[features] label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return X, y, feature_names, le


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "LogReg":  Pipeline([("sc", StandardScaler()),
                             ("clf", LogisticRegression(
                                 max_iter=2000,
                                 random_state=seed, n_jobs=-1))]),
        "RF":      Pipeline([("clf", RandomForestClassifier(
                                 n_estimators=300, max_depth=None,
                                 n_jobs=-1, random_state=seed))]),
        "HistGB":  Pipeline([("clf", HistGradientBoostingClassifier(
                                 max_iter=300, random_state=seed))]),
        "MLP":     Pipeline([("sc", StandardScaler()),
                             ("clf", MLPClassifier(
                                 hidden_layer_sizes=(128, 64), max_iter=250,
                                 random_state=seed, early_stopping=True))]),
    }


def try_xgboost(seed: int, n_classes: int) -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
        return Pipeline([("clf", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=seed, n_jobs=-1,
            verbosity=0))])
    except ImportError:
        return None


def run() -> None:
    df = load_dataset()
    X, y, feat_names, le = build_features(df)
    class_names = le.classes_.tolist()

    records: list[dict] = []
    test_preds: dict[str, list[np.ndarray]] = {}
    test_truths: list[np.ndarray] = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=seed)
        test_truths.append(y_te)

        models = make_models(seed)
        xgb = try_xgboost(seed, n_classes=len(class_names))
        if xgb is not None:
            models["XGBoost"] = xgb

        print(f"\n----- seed = {seed} -----")
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            yp = pipe.predict(X_te)
            acc = accuracy_score(y_te, yp)
            f1m = f1_score(y_te, yp, average="macro")
            per_cls_f1 = f1_score(y_te, yp, average=None,
                                  labels=range(len(class_names)))
            rec = {"model": name, "seed": seed,
                   "acc": acc, "f1_macro": f1m}
            for cls_name, f1_v in zip(class_names, per_cls_f1):
                rec[f"f1_{cls_name}"] = f1_v
            records.append(rec)
            test_preds.setdefault(name, []).append(yp)
            print(f"  {name:8s}  acc={acc:.4f}  f1_macro={f1m:.4f}  "
                  f"[{'  '.join(f'{c}:{v:.2f}' for c, v in zip(class_names, per_cls_f1))}]")

            if seed == 0:
                with open(MODEL_DIR / f"{name}_seed0.pkl", "wb") as f:
                    pickle.dump(pipe, f)

    res = pd.DataFrame(records)

    summary = res.groupby("model").agg({
        "acc":      ["mean", "std"],
        "f1_macro": ["mean", "std"],
        **{f"f1_{c}": ["mean"] for c in class_names},
    }).round(4)
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns]
    summary = summary.sort_values("acc_mean", ascending=False)

    print("\n" + "=" * 72)
    print(f" SUMMARY  (mean +/- std across {len(SEEDS)} seeds)")
    print("=" * 72)
    print(summary.to_string())
    print("=" * 72)

    res.to_csv(RESULTS_DIR / "compliance_per_seed.csv", index=False)
    summary.to_csv(RESULTS_DIR / "compliance_summary.csv")

    # Confusion matrix on seed 0's best model
    best_model_name = summary.index[0]
    print(f"\n[report] confusion matrix (best={best_model_name}, seed=0)")
    y_true0 = test_truths[0]
    y_pred0 = test_preds[best_model_name][0]
    cm = confusion_matrix(y_true0, y_pred0, labels=range(len(class_names)))
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print()
    print(classification_report(y_true0, y_pred0,
                                target_names=class_names, digits=3))

    # ---- plots
    _plot_confusion(cm, class_names, best_model_name)
    _plot_per_model_acc(summary, best_model_name)

    print(f"\nSaved: {RESULTS_DIR}/compliance_summary.csv")
    print(f"Saved: {RESULTS_DIR}/compliance_per_seed.csv")
    print(f"Saved: {MODEL_DIR}/*.pkl")
    print(f"Saved: {FIG_DIR}/*.png")


def _plot_confusion(cm: np.ndarray, class_names: list[str], model: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"IEEE 519 compliance — confusion matrix\n({model}, seed=0)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_confusion_matrix.png", dpi=150)
    plt.close()


def _plot_per_model_acc(summary: pd.DataFrame, best: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = summary.index.tolist()
    means = summary["acc_mean"].values
    stds  = summary["acc_std"].values
    colors = ["#e76f51" if m == best else "#264653" for m in models]
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m*100:.2f}%", ha="center", fontsize=9)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, max(means) * 1.10)
    ax.set_title("IEEE 519 compliance classifier accuracy (5 seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_accuracy_per_model.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run()
