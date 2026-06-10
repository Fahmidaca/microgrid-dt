"""Baseline ML models on UCI Electrical Grid Stability.

Trains 5 classifiers with N seeds each, reports mean +/- std accuracy and
macro-F1 on a held-out test split. Saves trained models for downstream use
in the robustness analysis.

Models:
    1. Logistic Regression       (linear baseline, what Schafer 2016 used)
    2. Random Forest
    3. XGBoost
    4. MLP (sklearn)
    5. Gradient Boosted Trees    (sklearn HistGB)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loaders import load_uci_grid


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 42, 123, 2026]
FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "LogReg":  Pipeline([("sc", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000,
                                                        random_state=seed))]),
        "RF":      Pipeline([("clf", RandomForestClassifier(
                                n_estimators=300, max_depth=None,
                                n_jobs=-1, random_state=seed))]),
        "HistGB":  Pipeline([("clf", HistGradientBoostingClassifier(
                                max_iter=300, random_state=seed))]),
        "MLP":     Pipeline([("sc", StandardScaler()),
                             ("clf", MLPClassifier(
                                hidden_layer_sizes=(64, 32), max_iter=300,
                                random_state=seed))]),
    }


def try_xgboost(seed: int) -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
        return Pipeline([("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
            verbosity=0))])
    except ImportError:
        return None


def run() -> pd.DataFrame:
    df = load_uci_grid()
    X = df[FEATURES].values
    y = (df["stabf"] == "unstable").astype(int).values

    records: list[dict] = []
    test_preds: dict[str, list[np.ndarray]] = {}
    test_truths: list[np.ndarray] = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)
        test_truths.append(y_te)

        models = make_models(seed)
        xgb = try_xgboost(seed)
        if xgb is not None:
            models["XGBoost"] = xgb

        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            yp = pipe.predict(X_te)
            acc = accuracy_score(y_te, yp)
            f1m = f1_score(y_te, yp, average="macro")
            records.append({"model": name, "seed": seed,
                            "acc": acc, "f1_macro": f1m})
            test_preds.setdefault(name, []).append(yp)
            print(f"  seed={seed}  {name:8s}  acc={acc:.4f}  f1={f1m:.4f}")

            # save model trained on seed=0 only (for downstream robustness)
            if seed == 0:
                with open(MODEL_DIR / f"{name}_seed0.pkl", "wb") as f:
                    pickle.dump(pipe, f)

    res = pd.DataFrame(records)

    summary = res.groupby("model").agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
    ).round(4).sort_values("acc_mean", ascending=False)

    print("\n=========== SUMMARY (mean +/- std across", len(SEEDS), "seeds) ===========")
    print(summary.to_string())

    res.to_csv(RESULTS_DIR / "baseline_per_seed.csv", index=False)
    summary.to_csv(RESULTS_DIR / "baseline_summary.csv")

    # save predictions for paired McNemar test downstream
    with open(RESULTS_DIR / "baseline_preds.pkl", "wb") as f:
        pickle.dump({"preds": test_preds, "truths": test_truths,
                     "seeds": SEEDS}, f)

    print(f"\nSaved: {RESULTS_DIR}/baseline_summary.csv")
    print(f"Saved: {RESULTS_DIR}/baseline_per_seed.csv")
    print(f"Saved: {RESULTS_DIR}/baseline_preds.pkl")
    print(f"Saved models in: {MODEL_DIR}")
    return res


if __name__ == "__main__":
    run()
