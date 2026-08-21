"""
homer_economic_optimization.py
================================
Reads the real HOMER Pro "Optimization Results" export
(data/external/homer_npc_coe_optimization.csv) - a comparison of
candidate microgrid architectures by Net Present Cost (NPC) and
Cost of Energy (COE). This is a genuine HOMER techno-economic output,
not a derived/estimated column - unlike economic_cost_BDT in the
Part 6 disturbance dataset, these numbers come directly from HOMER's
own optimizer.

Run:
    python eee_sim/homer_economic_optimization.py

Output:
    figures/eee_sim/06_coe_vs_renewable_fraction.png
    results/homer_optimization_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eee_sim"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)
CSV = ROOT / "data" / "external" / "homer_npc_coe_optimization.csv"


def run():
    df = pd.read_csv(CSV, skiprows=1)

    df = df.rename(columns={
        "Architecture/PV (kW)": "PV_kW",
        "Architecture/WT": "WT_count",
        "Architecture/FC (kW)": "FC_kW",
        "Architecture/BESS": "BESS_kWh_units",
        "Architecture/Electrolyzer (kW)": "Electrolyzer_kW",
        "Cost/NPC ($)": "NPC_USD",
        "Cost/COE ($)": "COE_USD_per_kWh",
        "Cost/Operating cost ($/yr)": "OpCost_USD_per_yr",
        "Cost/Initial capital ($)": "CapitalCost_USD",
        "System/Ren Frac (%)": "RenewableFraction_pct",
    })

    summary = df[["PV_kW", "WT_count", "FC_kW", "BESS_kWh_units", "Electrolyzer_kW",
                  "NPC_USD", "COE_USD_per_kWh",
                  "OpCost_USD_per_yr", "CapitalCost_USD", "RenewableFraction_pct"]].copy()
    summary.insert(0, "Architecture", [f"Config {i+1}" for i in range(len(summary))])
    summary.to_csv(RES_DIR / "homer_optimization_summary.csv", index=False)

    print("\n=========== HOMER Optimization Results (real output) ===========")
    for _, row in summary.iterrows():
        pv = row["PV_kW"] if pd.notna(row["PV_kW"]) else 0
        print(f"{row['Architecture']}: PV={pv:.0f}kW  WT={row['WT_count']}  FC={row['FC_kW']:.0f}kW  "
              f"BESS={row['BESS_kWh_units']}  RenFrac={row['RenewableFraction_pct']:.1f}%  "
              f"COE=${row['COE_USD_per_kWh']:.4f}/kWh  NPC=${row['NPC_USD']:,.0f}")
    print("===================================================================\n")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    ax1.bar(x, df["RenewableFraction_pct"], color=colors, alpha=0.7, label="Renewable Fraction (%)")
    ax1.set_ylabel("Renewable Fraction (%)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"Config {i+1}" for i in x])
    ax1.set_xlabel("HOMER-optimized architecture")

    ax2 = ax1.twinx()
    ax2.plot(x, df["COE_USD_per_kWh"], color="#264653", marker="o", lw=2, label="Cost of Energy ($/kWh)")
    ax2.set_ylabel("Cost of Energy (USD/kWh)")

    ax1.set_title("HOMER optimization: renewable fraction vs. cost of energy\n(real HOMER Pro output, not estimated)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "06_coe_vs_renewable_fraction.png", dpi=150)
    plt.close()

    print(f"Saved: {RES_DIR / 'homer_optimization_summary.csv'}")
    print(f"Saved: {FIG_DIR / '06_coe_vs_renewable_fraction.png'}")
    return summary


if __name__ == "__main__":
    run()
