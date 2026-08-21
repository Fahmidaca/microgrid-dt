"""
microgrid_pq_twin_scenarios.py
===============================
Drives the electrical-level power-quality digital twin
(`microgrid_pq_twin.py`'s plant / APF / THD-analyzer / battery-cost engine)
with real operating points pulled from a HOMER Pro / HOMER Grid annual
hourly simulation (data/external/homer_hourly_simulation.csv, 8,760 rows,
44 columns), instead of the single arbitrary demo timeline in the
original script.

Why this exists: HOMER output is hourly-averaged system-level energy
balance (no harmonics, no waveform detail) - it cannot itself produce
a THD number. This script picks three representative hours from that
HOMER year that match the proposal's three test scenarios, reads off
each hour's load and weather operating point, and feeds those into the
existing fast (10 kHz) electrical waveform simulation to see what power
quality looks like at that specific point in the annual profile.

Scenario selection (all computed from the HOMER data, not hand-picked):
  1. Harmonic distortion from nonlinear loads  -> the hour with the
     highest AC Primary Load (heaviest loading condition).
  2. Voltage sag from renewable fluctuation    -> the hour with the
     single largest hour-over-hour DROP in Total Renewable Power Output.
  3. Combined weather-electrical disturbance   -> the hour with low
     solar resource, above-median wind speed, and above-median load
     occurring together (a "bad weather + high demand" hour).

Run:
    python eee_sim/microgrid_pq_twin_scenarios.py

Output:
    results/eee_sim_scenarios_summary.csv
    figures/eee_sim/05_scenario_thd_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microgrid_pq_twin import APF, DigitalTwin, Params, plant_step, thd_window

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eee_sim"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)
HOMER_CSV = ROOT / "data" / "external" / "homer_hourly_simulation.csv"


# =========================================================================
#  LOAD HOMER DATA & PICK THE THREE SCENARIO HOURS
# =========================================================================
def load_homer(path: Path = HOMER_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, header=0)
    df = df.drop(index=0).reset_index(drop=True)  # units row
    for c in df.columns:
        if c != "Time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def pick_scenario_hours(df: pd.DataFrame) -> dict[str, pd.Series]:
    load = df["AC Primary Load"]
    renew = df["Total Renewable Power Output"]
    wind = df["Wind Speed"]
    solar = df["Global Solar"]

    # 1. Harmonic distortion from nonlinear loads: peak-load hour
    idx_harmonic = load.idxmax()

    # 2. Voltage sag from renewable fluctuation: largest hour-over-hour drop
    renew_delta = renew.diff()
    idx_sag = renew_delta.idxmin()

    # 3. Combined weather-electrical disturbance: low solar + high wind +
    #    high load, all at once
    mask = (solar < solar.median()) & (wind > wind.median()) & (load > load.median())
    candidates = df.loc[mask]
    if len(candidates) == 0:
        idx_combined = ((load / load.max()) * (wind / wind.max()) * (1 - solar / solar.max())).idxmax()
    else:
        score = (candidates["AC Primary Load"] / load.max()) * (candidates["Wind Speed"] / wind.max())
        idx_combined = score.idxmax()

    return {
        "Harmonic_Distortion": df.loc[idx_harmonic],
        "Voltage_Sag": df.loc[idx_sag],
        "Combined_Weather_Electrical": df.loc[idx_combined],
    }


# =========================================================================
#  RUN THE ELECTRICAL-LEVEL TWIN AT ONE HOMER-DERIVED OPERATING POINT
# =========================================================================
def run_scenario(name: str, row: pd.Series, load_ref_kW: float) -> dict:
    P = Params()

    # scale the nonlinear-load fundamental current by how loaded this
    # HOMER hour is relative to the dataset's median hour (the operating
    # point the base Ifund=15A parameter was set for)
    load_scale = float(np.clip(row["AC Primary Load"] / load_ref_kW, 0.3, 2.5))
    P.Ifund = 15.0 * load_scale

    # weather operating point straight from the HOMER hour
    G_hour = max(float(row["Global Solar"]) * 1000.0, 0.0)   # kW/m2 -> W/m2
    T_hour = float(row["Ambient Temperature"])

    # sag depth for the Voltage_Sag scenario scales with how severe the
    # renewable drop was that hour; other scenarios keep a small nominal sag
    if name == "Voltage_Sag":
        renew_frac_of_load = min(float(row["Total Renewable Power Output"]) / max(load_ref_kW, 1.0), 1.0)
        P.sag_depth = float(np.clip(0.15 + 0.35 * renew_frac_of_load, 0.15, 0.5))
    elif name == "Combined_Weather_Electrical":
        P.sag_depth = 0.20
    else:
        P.sag_depth = 0.05

    fs = 1 / P.Ts
    N_total = int(P.Tend / P.Ts) + 1
    t_arr = np.arange(N_total) * P.Ts
    G_arr = np.full(N_total, G_hour)
    T_arr = np.full(N_total, T_hour)

    apf = APF(Ts=P.Ts, f0=P.f0)
    twin = DigitalTwin(P)

    Nfft = 1000
    v_buf = np.zeros(Nfft); i_buf = np.zeros(Nfft); buf_idx = 0
    thd_v_cur = thd_i_cur = 0.0
    THD_i_log = np.zeros(N_total); THD_v_log = np.zeros(N_total)
    cost_log = np.zeros(N_total); SoH_log = np.zeros(N_total)
    Ppv_log = np.zeros(N_total)

    for k in range(N_total):
        t = t_arr[k]
        sag = 1.0 if (P.sag_start <= t < P.sag_start + P.sag_dur) else 0.0
        apf_on = 1.0 if t >= P.apf_on_t else 0.0

        Vabc, Iabc, Ppv = plant_step(t, G_arr[k], T_arr[k], sag, P)
        Isource = apf.step(Iabc, t, apf_on)

        v_buf[buf_idx] = Vabc[0]; i_buf[buf_idx] = Isource[0]; buf_idx += 1
        if buf_idx >= Nfft:
            thd_v_cur = thd_window(v_buf, fs)
            thd_i_cur = thd_window(i_buf, fs)
            buf_idx = 0

        tw_out = twin.step(thd_i_cur, Ppv, t)
        THD_i_log[k] = thd_i_cur; THD_v_log[k] = thd_v_cur
        cost_log[k] = tw_out["cost_BDT_yr"]; SoH_log[k] = tw_out["SoH_loss_pct"]
        Ppv_log[k] = Ppv

    pre = pd.Series(THD_i_log)[(t_arr > 0.10) & (t_arr < P.apf_on_t - 0.05)]
    post = pd.Series(THD_i_log)[t_arr > P.apf_on_t + 0.10]

    return {
        "scenario": name,
        "homer_timestamp": str(row["Time"]),
        "homer_AC_load_kW": float(row["AC Primary Load"]),
        "homer_renewable_penetration_pct": float(row["Renewable Penetration"]),
        "homer_global_solar_kWm2": float(row["Global Solar"]),
        "homer_wind_speed_ms": float(row["Wind Speed"]),
        "load_scale_applied": load_scale,
        "sag_depth_applied": P.sag_depth,
        "thd_i_pct_before_APF": float(pre.mean() * 100),
        "thd_i_pct_after_APF": float(post.mean() * 100),
        "thd_v_pct_before_APF": float(pd.Series(THD_v_log)[t_arr < P.apf_on_t].mean() * 100),
        "SoH_loss_final_pct": float(SoH_log[-1]),
        "cost_BDT_per_yr_final": float(cost_log[-1]),
    }, (t_arr, THD_i_log, THD_v_log)


def run() -> pd.DataFrame:
    if not HOMER_CSV.exists():
        raise FileNotFoundError(f"HOMER dataset not found at {HOMER_CSV}")

    df = load_homer(HOMER_CSV)
    load_ref_kW = float(df["AC Primary Load"].median())
    scenarios = pick_scenario_hours(df)

    rows = []
    curves = {}
    for name, row in scenarios.items():
        result, curve = run_scenario(name, row, load_ref_kW)
        rows.append(result)
        curves[name] = curve
        print(f"\n--- {name} ---")
        print(f"  HOMER hour: {result['homer_timestamp']}  "
              f"load={result['homer_AC_load_kW']:.1f} kW  "
              f"RE penetration={result['homer_renewable_penetration_pct']:.1f}%")
        print(f"  THD_i before/after APF: {result['thd_i_pct_before_APF']:.2f}% -> "
              f"{result['thd_i_pct_after_APF']:.2f}%")
        print(f"  Projected cost at this operating point: "
              f"{result['cost_BDT_per_yr_final']:.0f} BDT/yr")

    summary = pd.DataFrame(rows)
    summary.to_csv(RES_DIR / "eee_sim_scenarios_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Harmonic_Distortion": "#e76f51",
              "Voltage_Sag": "#264653",
              "Combined_Weather_Electrical": "#2a9d8f"}
    for name, (t_arr, thd_i, _) in curves.items():
        ax.plot(t_arr, thd_i * 100, label=name.replace("_", " "), color=colors[name], lw=1.3)
    ax.axhline(5, color="black", ls=":", lw=1, label="IEEE 519 limit (5%)")
    p0 = Params()
    ax.axvline(p0.apf_on_t, color="grey", ls="--", lw=1, label="APF ON")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("THD$_i$ (%)")
    ax.set_title("THD$_i$ under three HOMER-derived operating points")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_scenario_thd_comparison.png", dpi=150)
    plt.close()

    print(f"\nSaved: {RES_DIR / 'eee_sim_scenarios_summary.csv'}")
    print(f"Saved: {FIG_DIR / '05_scenario_thd_comparison.png'}")
    return summary


if __name__ == "__main__":
    run()
