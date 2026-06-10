"""Reaction-Time Robustness Margin Analysis (the paper's novel contribution).

For each test configuration, compute the smallest reaction-time perturbation
that flips the trained model's prediction. This *robustness margin* tells a
grid planner how much consumer reaction-time degradation a given grid can
absorb before its stability status changes.

Concretely, for a fixed (p, g) tuple and a model M:

    margin(x) = min  ||delta||_inf
                  delta
                s.t. M(x + (delta, 0, 0))  !=  M(x)
                     0 <= tau_i + delta_i <= 10  (sim range)

We solve this with a binary search on the L_inf magnitude of a worst-case
sign pattern (one of 16 +/- combinations on tau1..tau4). Margin distributions
are reported per class and per model.

This is the paper's novel contribution: prior work reports only raw
classification accuracy; we quantify the *operational safety buffer* that
each model preserves on top of correctness.
"""

from __future__ import annotations

import pickle
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from loaders import load_uci_grid


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "robustness"; FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["tau1", "tau2", "tau3", "tau4",
            "p1", "p2", "p3", "p4",
            "g1", "g2", "g3", "g4"]
TAU_IDX = [0, 1, 2, 3]
TAU_MIN, TAU_MAX = 0.5, 10.0

SIGN_PATTERNS = list(product([-1, +1], repeat=4))   # 16 patterns


def predict_with_delta(model, x, delta, sign):
    """Apply tau perturbation: tau_i := clip(tau_i + sign_i * delta, [0.5, 10])."""
    x = x.copy()
    for i, k in enumerate(TAU_IDX):
        x[k] = float(np.clip(x[k] + sign[i] * delta, TAU_MIN, TAU_MAX))
    return int(model.predict(x.reshape(1, -1))[0])


def margin_for_sample(model, x, y_orig, max_delta: float = 5.0,
                      tol: float = 0.01) -> float:
    """Smallest L_inf tau-perturbation that flips the model's prediction.
    Searches each of 16 sign patterns; returns the smallest flipping delta.
    Returns max_delta if no sign pattern within range can flip it."""
    best = max_delta
    for sign in SIGN_PATTERNS:
        # quick check: does even max_delta flip the prediction?
        if predict_with_delta(model, x, max_delta, sign) == y_orig:
            continue
        # binary search for smallest delta that flips
        lo, hi = 0.0, max_delta
        while hi - lo > tol:
            mid = (lo + hi) / 2
            y_new = predict_with_delta(model, x, mid, sign)
            if y_new == y_orig:
                lo = mid
            else:
                hi = mid
        best = min(best, hi)
    return best


def main(n_samples_per_class: int = 200) -> None:
    df = load_uci_grid()
    X = df[FEATURES].values
    y = (df["stabf"] == "unstable").astype(int).values

    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)

    # Sample N stable + N unstable for the analysis (it is O(16 * binsearch))
    rng = np.random.default_rng(0)
    stable_idx   = np.where(y_te == 0)[0]
    unstable_idx = np.where(y_te == 1)[0]
    pick_stable   = rng.choice(stable_idx,
                               size=min(n_samples_per_class, len(stable_idx)),
                               replace=False)
    pick_unstable = rng.choice(unstable_idx,
                               size=min(n_samples_per_class, len(unstable_idx)),
                               replace=False)
    pick = np.concatenate([pick_stable, pick_unstable])

    rows = []
    model_paths = sorted(MODEL_DIR.glob("*_seed0.pkl"))
    if not model_paths:
        raise FileNotFoundError("No trained models in models/. Run baselines.py first.")

    for mp in model_paths:
        name = mp.stem.replace("_seed0", "")
        with open(mp, "rb") as f:
            model = pickle.load(f)
        print(f"\n[{name}] computing tau-robustness margin on "
              f"{len(pick)} test samples...")

        # only analyse samples the model gets CORRECT — margin is meaningful
        # only when the starting prediction matches ground truth
        preds = model.predict(X_te[pick])
        correct_mask = (preds == y_te[pick])
        n_correct = int(correct_mask.sum())
        print(f"  ({n_correct}/{len(pick)} correctly classified)")

        margins = []
        for i, idx in enumerate(pick):
            if not correct_mask[i]:
                continue
            m = margin_for_sample(model, X_te[idx], y_te[idx])
            margins.append({"model": name, "true_class": int(y_te[idx]),
                            "margin": m})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pick)} done")
        rows.extend(margins)

    df_margins = pd.DataFrame(rows)
    df_margins.to_csv(RESULTS_DIR / "robustness_margins.csv", index=False)

    print("\n=== Robustness summary (mean margin, larger = more robust) ===\n")
    summary = (df_margins
               .groupby(["model", "true_class"])
               .agg(margin_mean=("margin", "mean"),
                    margin_median=("margin", "median"),
                    margin_p05=("margin", lambda s: s.quantile(0.05)),
                    margin_p95=("margin", lambda s: s.quantile(0.95)),
                    n=("margin", "size"))
               .round(3))
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "robustness_summary.csv")

    # plot: margin distribution per model
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = df_margins["model"].unique()
    data = [df_margins.loc[df_margins.model == m, "margin"].values for m in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch, col in zip(bp["boxes"],
                          ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]):
        patch.set_facecolor(col)
    ax.set_ylabel(r"Reaction-time robustness margin $\|\delta\tau\|_\infty$ (s)")
    ax.set_title("Per-model tau-robustness margin\n"
                 "(larger = grid stability prediction harder to flip)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_margin_per_model.png", dpi=150)
    plt.close()

    # plot: stable vs unstable margins per model
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35; xpos = np.arange(len(models))
    for k, (cls, label, col) in enumerate(
            [(0, "stable", "#2a9d8f"), (1, "unstable", "#e76f51")]):
        means = [df_margins.loc[(df_margins.model == m) &
                                (df_margins.true_class == cls), "margin"].mean()
                 for m in models]
        ax.bar(xpos + (k - 0.5) * width, means, width, label=label, color=col)
    ax.set_xticks(xpos); ax.set_xticklabels(models)
    ax.set_ylabel("Mean tau-margin (s)")
    ax.set_title("Mean tau-robustness margin per model and true class")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_margin_per_class.png", dpi=150)
    plt.close()

    print(f"\nSaved: {RESULTS_DIR}/robustness_margins.csv")
    print(f"Saved: {RESULTS_DIR}/robustness_summary.csv")
    print(f"Saved figs: {FIG_DIR}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n_samples_per_class=n)
