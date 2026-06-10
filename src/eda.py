"""Exploratory Data Analysis on UCI Electrical Grid Stability.

Generates:
    1. Class distribution check
    2. Feature distributions (tau, p, g groups)
    3. Correlation matrix
    4. Tau vs. stability boundary scatter (the key relationship for the paper)
    5. PCA 2D projection coloured by class

Outputs all plots to figures/eda/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from loaders import load_uci_grid


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["stabf"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("Class distribution (UCI Grid Stability)")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=150)
    plt.close()


def plot_tau_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, col in zip(axes.ravel(), ["tau1", "tau2", "tau3", "tau4"]):
        ax.hist(df.loc[df.stabf == "stable",   col], bins=40,
                alpha=0.6, label="stable",   color="#2a9d8f")
        ax.hist(df.loc[df.stabf == "unstable", col], bins=40,
                alpha=0.6, label="unstable", color="#e76f51")
        ax.set_title(f"{col} (reaction time, s)"); ax.set_xlabel(col); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_tau_distributions.png", dpi=150)
    plt.close()


def plot_correlation(df: pd.DataFrame) -> None:
    num = df.drop(columns=["stabf"])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Feature correlation matrix")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_correlation_matrix.png", dpi=150)
    plt.close()


def plot_tau_stability_boundary(df: pd.DataFrame) -> None:
    """The KEY plot for the paper: shows how tau magnitude relates to stability.
    tau_mean = (tau1+tau2+tau3+tau4)/4, tau_max = max across the four."""
    df = df.copy()
    df["tau_mean"] = df[["tau1", "tau2", "tau3", "tau4"]].mean(axis=1)
    df["tau_max"]  = df[["tau1", "tau2", "tau3", "tau4"]].max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in [(axes[0], "tau_mean", "Mean reaction time vs. stab"),
                           (axes[1], "tau_max",  "Max reaction time vs. stab")]:
        ax.scatter(df.loc[df.stabf == "stable",   col],
                   df.loc[df.stabf == "stable",   "stab"],
                   s=3, alpha=0.3, label="stable",   color="#2a9d8f")
        ax.scatter(df.loc[df.stabf == "unstable", col],
                   df.loc[df.stabf == "unstable", "stab"],
                   s=3, alpha=0.3, label="unstable", color="#e76f51")
        ax.axhline(0, color="black", lw=0.8, ls="--", label="boundary (stab=0)")
        ax.set_xlabel(col); ax.set_ylabel("stab (max real eigenvalue)")
        ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_tau_stability_boundary.png", dpi=150)
    plt.close()


def plot_pca(df: pd.DataFrame) -> None:
    X = df.drop(columns=["stab", "stabf"]).values
    y = (df["stabf"] == "unstable").astype(int).values
    Xs = StandardScaler().fit_transform(X)
    pc = PCA(n_components=2).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pc[y == 0, 0], pc[y == 0, 1], s=4, alpha=0.5,
               label="stable",   color="#2a9d8f")
    ax.scatter(pc[y == 1, 0], pc[y == 1, 1], s=4, alpha=0.5,
               label="unstable", color="#e76f51")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    ax.set_title("PCA projection of 12 features (Standardised)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_pca_projection.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_uci_grid()
    print(f"\nLoaded {len(df)} rows, {df.shape[1]} cols")

    print("\n--- 1. Class distribution ---")
    print(df["stabf"].value_counts())
    plot_class_distribution(df)

    print("\n--- 2. Tau distributions ---")
    plot_tau_distributions(df)

    print("\n--- 3. Correlation matrix ---")
    plot_correlation(df)

    print("\n--- 4. Tau vs stability boundary ---")
    plot_tau_stability_boundary(df)

    print("\n--- 5. PCA projection ---")
    plot_pca(df)

    print(f"\nAll EDA plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
