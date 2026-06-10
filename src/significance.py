"""Statistical significance framework for UCI Grid Stability classifiers.

Provides:
    1. Pairwise McNemar test (corrected, with continuity adjustment)
       -> p-value for "model A and model B make different errors"
    2. Bootstrap 95% confidence interval on accuracy and macro-F1
    3. Holm-Bonferroni correction across multiple pairs

Reads predictions saved by baselines.py and prints a publication-ready table.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_preds():
    with open(RESULTS_DIR / "baseline_preds.pkl", "rb") as f:
        return pickle.load(f)


def mcnemar_pvalue(y_true, y_a, y_b) -> tuple[float, int, int]:
    """McNemar's test on a paired contingency table.
    Returns (p_value, n_a_wins, n_b_wins) where:
        n_a_wins = times A is correct AND B is wrong
        n_b_wins = times B is correct AND A is wrong
    """
    a_correct = (y_a == y_true)
    b_correct = (y_b == y_true)
    n_a_wins = int(((a_correct) & (~b_correct)).sum())   # b00 -> A right, B wrong
    n_b_wins = int(((~a_correct) & (b_correct)).sum())   # b01 -> A wrong, B right
    n_both_right = int((a_correct & b_correct).sum())
    n_both_wrong = int(((~a_correct) & (~b_correct)).sum())
    table = [[n_both_right, n_a_wins],
             [n_b_wins,     n_both_wrong]]
    # exact binomial McNemar when n_a_wins + n_b_wins <= 25
    result = mcnemar(table, exact=(n_a_wins + n_b_wins <= 25), correction=True)
    return result.pvalue, n_a_wins, n_b_wins


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 2000,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Returns (point_estimate, lower_95, upper_95)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y_true)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(metric(y_true[idx], y_pred[idx]))
    point = metric(y_true, y_pred)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        adj_p = min(pvals[i] * (m - rank), 1.0)
        running_max = max(running_max, adj_p)
        adj[i] = running_max
    return adj.tolist()


def main() -> None:
    data = load_preds()
    truths = data["truths"]
    preds  = data["preds"]
    seeds  = data["seeds"]
    model_names = list(preds.keys())

    # Concatenate predictions across all seeds for one paired comparison
    y_true_concat = np.concatenate(truths)
    pred_concat = {m: np.concatenate(preds[m]) for m in model_names}

    print(f"\n=== Bootstrap 95% CIs (n_boot=2000) over {len(y_true_concat)} preds ===\n")
    rng = np.random.default_rng(123)
    rows = []
    for m in model_names:
        acc_p, acc_lo, acc_hi = bootstrap_ci(
            y_true_concat, pred_concat[m], accuracy_score, rng=rng)
        f1_p, f1_lo, f1_hi = bootstrap_ci(
            y_true_concat, pred_concat[m],
            lambda yt, yp: f1_score(yt, yp, average="macro"), rng=rng)
        rows.append({"model": m,
                     "acc": acc_p, "acc_lo": acc_lo, "acc_hi": acc_hi,
                     "f1":  f1_p,  "f1_lo":  f1_lo,  "f1_hi":  f1_hi})
    ci_df = pd.DataFrame(rows).round(4)
    print(ci_df.to_string(index=False))
    ci_df.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)

    print("\n=== Pairwise McNemar tests (corrected, all seeds pooled) ===\n")
    pair_results = []
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            if i >= j: continue
            p, a_wins, b_wins = mcnemar_pvalue(
                y_true_concat, pred_concat[a], pred_concat[b])
            pair_results.append({
                "model_A": a, "model_B": b,
                "A_wins (A right, B wrong)": a_wins,
                "B_wins (B right, A wrong)": b_wins,
                "p_raw": p,
            })

    # Holm-Bonferroni correction across all pairs
    pvals = [r["p_raw"] for r in pair_results]
    adj   = holm_bonferroni(pvals)
    for r, ap in zip(pair_results, adj):
        r["p_holm"] = ap
        r["significant_005"] = ap < 0.05
    pair_df = pd.DataFrame(pair_results).round({"p_raw": 6, "p_holm": 6})
    print(pair_df.to_string(index=False))
    pair_df.to_csv(RESULTS_DIR / "mcnemar_pairs.csv", index=False)

    print(f"\nSaved: {RESULTS_DIR}/bootstrap_ci.csv")
    print(f"Saved: {RESULTS_DIR}/mcnemar_pairs.csv")


if __name__ == "__main__":
    main()
