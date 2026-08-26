"""
mesa_del_sol_validation.py
===========================
Independent, real-world sanity check on the EEE twin's voltage/frequency
assumptions, using telemetry from an unrelated, DOI-cited, real microgrid
(Mesa Del Sol, University of New Mexico; Bashir et al. 2023, Dryad,
doi:10.5061/dryad.fqz612jzb) - not the Jamalpur dataset, not HOMER, no
shared authorship or pipeline with this project.

Why this dataset, and what it can and cannot show:
  - It is a DIFFERENT microgrid on a DIFFERENT continent (60 Hz nominal,
    ~480 V class bus, US commercial campus), so it cannot validate
    Jamalpur-specific numbers (THD levels, BDT costs, disturbance
    frequency). Any such comparison would be apples-to-oranges and this
    script does not attempt it.
  - What it CAN legitimately show: whether "the grid holds voltage and
    frequency at an exact constant except during one scripted event" -
    the twin's plant_step() assumption (Vnom fixed, f0 fixed, sag is a
    single deterministic step) - looks anything like what a real,
    operating microgrid actually does from second to second. That is a
    structural/qualitative check, expressed in normalized (per-unit)
    terms so the different nominal voltage/frequency don't matter.

Data used: one representative month (April 2023, 267,625 rows at 10 s
resolution) from the 15-month release. The raw CSV is NOT committed to
this repo (467 MB across all months; it belongs to the Dryad DOI above -
redistribute by citation, not by copy). Download it yourself from the
DOI and place the monthly file at:
    data/external/mesa_del_sol/Apr_2023.csv
(gitignored - see .gitignore)

Run:
    python eee_sim/mesa_del_sol_validation.py

Output:
    results/mesa_del_sol_validation.csv
    figures/eee_sim/09_mesa_del_sol_validation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eee_sim"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "data" / "external" / "mesa_del_sol" / "Apr_2023.csv"

# the EEE twin's own assumptions (microgrid_pq_twin.py Params), for direct
# comparison - not fitted to this dataset in any way
TWIN_F0_HZ = 50.0
TWIN_VOLTAGE_RIPPLE_PU = 0.0     # plant_step: V is exactly Vnom outside the sag
TWIN_FREQ_DEVIATION_PU = 0.0    # plant_step: f is exactly f0, always


SENTINEL = -999999.0  # this dataset's own missing/comms-loss placeholder


def load() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the Mesa Del Sol dataset from "
            "doi:10.5061/dryad.fqz612jzb and place the monthly CSV there."
        )
    df = pd.read_csv(DATA_PATH)
    n_raw = len(df)
    # a block of trailing rows in this file carries an out-of-range date
    # (e.g. "2023/04/31") alongside SENTINEL values in every column - a
    # padding/comms-loss artifact of the source logger, not real telemetry.
    # Drop rows with the sentinel rather than parsing the corrupted dates.
    numeric_cols = [c for c in df.columns if c != "Timestamp"]
    df = df[~(df[numeric_cols] == SENTINEL).any(axis=1)].reset_index(drop=True)
    print(f"Loaded {n_raw:,} rows, dropped {n_raw - len(df):,} sentinel/placeholder "
          f"rows ({SENTINEL}), {len(df):,} valid samples remain.")
    return df


def pu_stats(series: pd.Series, nominal: float) -> dict:
    x = series.dropna().values
    return {
        "nominal": nominal,
        "mean": x.mean(),
        "std": x.std(),
        "std_pu_pct": 100 * x.std() / nominal,
        "min": x.min(),
        "max": x.max(),
        "range_pu_pct": 100 * (x.max() - x.min()) / nominal,
        "p1": np.percentile(x, 1),
        "p99": np.percentile(x, 99),
    }


def run():
    df = load()
    n = len(df)
    days = n * 10 / 86400  # 10 s sampling interval

    # --- frequency: two independent measurement points in the same dataset
    freq_cols = {
        "MG-LV-MSB_Frequency": "Main LV switchboard",
        "Island_mode_MCCB_Frequency": "Island-mode MCCB",
    }
    freq_rows = {}
    for col, label in freq_cols.items():
        s = pu_stats(df[col], nominal=60.0)
        s["label"] = label
        freq_rows[col] = s

    # --- voltage: two independent bus measurement points
    volt_cols = {
        "MG-LV-MSB_AC_Voltage": "Main LV switchboard",
        "Receiving_Point_AC_Voltage": "Utility receiving point",
    }
    volt_rows = {}
    for col, label in volt_cols.items():
        nominal = df[col].mean()  # this system's own operating nominal (~480 V class)
        s = pu_stats(df[col], nominal=nominal)
        s["label"] = label
        volt_rows[col] = s

    rows = []
    for col, s in freq_rows.items():
        rows.append({"quantity": f"frequency ({s['label']})", "column": col, **{k: v for k, v in s.items() if k != "label"}})
    for col, s in volt_rows.items():
        rows.append({"quantity": f"voltage ({s['label']})", "column": col, **{k: v for k, v in s.items() if k != "label"}})
    summary = pd.DataFrame(rows)
    summary.to_csv(RES_DIR / "mesa_del_sol_validation.csv", index=False)

    print("\n=========== Mesa Del Sol real-microgrid validation "
          f"({days:.1f} days, {n:,} samples @ 10 s, April 2023) ===========")
    print(summary.to_string(index=False))
    print(f"\nEEE twin assumption: frequency deviation = {TWIN_FREQ_DEVIATION_PU:.3f} pu "
          "(exactly f0, always)")
    print(f"EEE twin assumption: voltage ripple outside the scripted sag = "
          f"{TWIN_VOLTAGE_RIPPLE_PU:.3f} pu (exactly Vnom)")
    worst_freq_std = max(s["std_pu_pct"] for s in freq_rows.values())
    worst_volt_std = max(s["std_pu_pct"] for s in volt_rows.values())
    print(f"\n=> Real Mesa Del Sol data shows continuous frequency variability "
          f"of up to {worst_freq_std:.3f}% (pu std) and continuous voltage "
          f"variability of up to {worst_volt_std:.3f}% (pu std), even though "
          "no fault or sag event is scripted for most of the record.")
    print("This is real, independent evidence that the twin's deterministic "
          "constant-V / constant-f assumption between events is an "
          "idealisation - a concrete, evidenced limitation rather than a "
          "hand-waved one.")
    print("==================================================================\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, label) in zip(axes[0], freq_cols.items()):
        ax.hist(df[col].dropna(), bins=80, color="#264653", alpha=0.85)
        ax.axvline(60.0, color="#e76f51", lw=1.5, ls="--", label="60 Hz nominal")
        ax.set_title(f"Frequency: {label}")
        ax.set_xlabel("Hz")
        ax.legend(fontsize=8)
    for ax, (col, label) in zip(axes[1], volt_cols.items()):
        nominal = df[col].mean()
        ax.hist(df[col].dropna(), bins=80, color="#2a9d8f", alpha=0.85)
        ax.axvline(nominal, color="#e76f51", lw=1.5, ls="--", label=f"mean ({nominal:.0f} V)")
        ax.set_title(f"Voltage: {label}")
        ax.set_xlabel("V")
        ax.legend(fontsize=8)
    fig.suptitle("Mesa Del Sol real microgrid telemetry (April 2023) - "
                  "the twin models both of these as exact constants outside a single scripted sag")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_mesa_del_sol_validation.png", dpi=150)
    plt.close()

    print(f"Saved: {RES_DIR / 'mesa_del_sol_validation.csv'}")
    print(f"Saved: {FIG_DIR / '09_mesa_del_sol_validation.png'}")
    return summary


if __name__ == "__main__":
    run()
