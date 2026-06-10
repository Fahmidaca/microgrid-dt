"""Print a publication-ready summary of all results.

Reads results/ and figures/ from the pipeline and produces a single text
block suitable for pasting into the paper's Results section.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    print("=" * 72)
    print("UCI Grid Stability — Pipeline Results Summary")
    print("=" * 72)

    bs = RESULTS / "baseline_summary.csv"
    if bs.exists():
        print("\n[1] Baseline classification (mean +- std over 5 seeds)")
        print("-" * 72)
        print(pd.read_csv(bs, index_col=0).to_string())

    ci = RESULTS / "bootstrap_ci.csv"
    if ci.exists():
        print("\n[2] Bootstrap 95% CI on accuracy and macro-F1 (n_boot=2000)")
        print("-" * 72)
        print(pd.read_csv(ci).to_string(index=False))

    mc = RESULTS / "mcnemar_pairs.csv"
    if mc.exists():
        print("\n[3] Pairwise McNemar (Holm-Bonferroni corrected)")
        print("-" * 72)
        df = pd.read_csv(mc)
        print(df[["model_A", "model_B", "p_holm", "significant_005"]]
              .to_string(index=False))

    rs = RESULTS / "robustness_summary.csv"
    if rs.exists():
        print("\n[4] Tau-robustness margin (NOVEL CONTRIBUTION)")
        print("-" * 72)
        df = pd.read_csv(rs)
        df["true_class"] = df["true_class"].map({0: "stable", 1: "unstable"})
        print(df.to_string(index=False))
    else:
        print("\n[4] Tau-robustness margin: results pending (run src/robustness.py)")

    print("\n" + "=" * 72)
    print("Plot files:")
    print("-" * 72)
    for p in sorted((ROOT / "figures").rglob("*.png")):
        print(f"  {p.relative_to(ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
