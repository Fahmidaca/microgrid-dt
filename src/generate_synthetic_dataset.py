"""
Synthetic Microgrid Dataset Generator
=====================================

Generates a large synthetic time-series dataset of microgrid operational
measurements, covering renewable-generation, load, power-quality, storage,
and compliance labels.

>>>>>  WARNING  <<<<<
This dataset is SYNTHETIC. It is intended ONLY for:
  1. Building and testing the ML/analysis pipeline (proof-of-concept).
  2. Data-augmentation on top of a real base dataset.
  3. Explicit "simulation study" experiments.

DO NOT use this dataset's numbers as the primary evidence in any paper
without also running the real Simulink simulator on the IEEE 14-bus model
and using its output for headline claims. Reviewers WILL ask for the raw
simulation source of every published number.

Every row of the generated CSV is watermarked with a `source` column set to
"SYNTHETIC_GENERATOR_v1" so that any downstream code / paper table can
verify the origin at a glance.

Run:
    python src/generate_synthetic_dataset.py                 # default 50k rows
    python src/generate_synthetic_dataset.py 100000          # custom size
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"; DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "synthetic"; FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
#  CONSTANTS (Bangladesh 50 Hz microgrid)
# =========================================================================
F0_NOMINAL = 50.0
V_NOMINAL = 230.0
SAMPLE_DT_SEC = 60  # one sample per minute of operation

SCENARIOS = [
    "base_case_no_RE",     # 0 % renewable
    "PV_only_20pct",
    "PV_only_40pct",
    "wind_only_20pct",
    "wind_only_40pct",
    "hybrid_PV_wind_30pct",
    "hybrid_PV_wind_50pct",
    "hybrid_with_nonlinear_loads",
    "hybrid_with_ESS_mitigation",
    "AI_optimized_control",
]


# =========================================================================
#  PHYSICAL RELATIONSHIPS BETWEEN FEATURES
# =========================================================================
def scenario_re_share(name: str) -> tuple[float, float, float, bool, bool]:
    """(pv_share, wind_share, nonlinear_load_frac, ESS_on, AI_control)"""
    m = {
        "base_case_no_RE":               (0.00, 0.00, 0.10, False, False),
        "PV_only_20pct":                 (0.20, 0.00, 0.15, False, False),
        "PV_only_40pct":                 (0.40, 0.00, 0.15, False, False),
        "wind_only_20pct":               (0.00, 0.20, 0.15, False, False),
        "wind_only_40pct":               (0.00, 0.40, 0.20, False, False),
        "hybrid_PV_wind_30pct":          (0.15, 0.15, 0.20, False, False),
        "hybrid_PV_wind_50pct":          (0.25, 0.25, 0.25, False, False),
        "hybrid_with_nonlinear_loads":   (0.25, 0.25, 0.55, False, False),
        "hybrid_with_ESS_mitigation":    (0.25, 0.25, 0.55, True,  False),
        "AI_optimized_control":          (0.30, 0.30, 0.55, True,  True),
    }
    return m[name]


def solar_irradiance(hour_of_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal daytime irradiance with cloud noise (Dhaka clear-sky-ish)."""
    base = np.clip(np.sin((hour_of_day - 6) / 12 * np.pi), 0, None) * 900
    cloud_noise = rng.normal(0, 60, size=len(hour_of_day))
    return np.clip(base + cloud_noise, 0, 1050)


def wind_speed(hour_of_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Diurnal wind pattern with turbulence."""
    base = 5 + 2 * np.sin((hour_of_day - 14) / 24 * 2 * np.pi)  # peaks afternoon
    return np.clip(base + rng.normal(0, 1.2, size=len(hour_of_day)), 0.1, 20)


def ambient_temperature(hour_of_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dhaka June daily temperature curve."""
    base = 30 + 5 * np.sin((hour_of_day - 15) / 24 * 2 * np.pi)
    return base + rng.normal(0, 0.6, size=len(hour_of_day))


def load_profile(hour_of_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Typical urban daily load curve (residential + commercial mix)."""
    morning = np.exp(-((hour_of_day - 8) ** 2) / 5) * 30
    evening = np.exp(-((hour_of_day - 20) ** 2) / 8) * 55
    baseload = 40
    return baseload + morning + evening + rng.normal(0, 4, size=len(hour_of_day))


# =========================================================================
#  MAIN GENERATOR
# =========================================================================
def generate(n_rows: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Assign rows to scenarios (roughly balanced)
    per_scen = n_rows // len(SCENARIOS)
    scen_col = np.repeat(SCENARIOS, per_scen)
    # Pad if not exactly divisible
    if len(scen_col) < n_rows:
        scen_col = np.concatenate([scen_col,
                                   rng.choice(SCENARIOS, n_rows - len(scen_col))])
    rng.shuffle(scen_col)

    # Time axis: minute-resolution timestamps starting at a fixed date
    start = pd.Timestamp("2026-06-01 00:00:00")
    ts = start + pd.to_timedelta(np.arange(n_rows) * SAMPLE_DT_SEC, unit="s")
    hour_of_day = ts.hour + ts.minute / 60.0
    day_of_year = ts.dayofyear

    # Weather / environment
    G_irrad = solar_irradiance(hour_of_day.values, rng)              # W/m^2
    T_amb = ambient_temperature(hour_of_day.values, rng)             # degC
    humidity = np.clip(65 + rng.normal(0, 12, n_rows), 20, 100)      # %
    wind_ms = wind_speed(hour_of_day.values, rng)                    # m/s

    # Baseline load
    load_kW = load_profile(hour_of_day.values, rng)

    # Vectorised scenario mapping
    pv_share = np.zeros(n_rows)
    wind_share = np.zeros(n_rows)
    nonlin_frac = np.zeros(n_rows)
    ESS_on = np.zeros(n_rows, dtype=bool)
    AI_ctrl = np.zeros(n_rows, dtype=bool)
    for s in SCENARIOS:
        mask = scen_col == s
        p, w, nl, ess, ai = scenario_re_share(s)
        pv_share[mask] = p
        wind_share[mask] = w
        nonlin_frac[mask] = nl
        ESS_on[mask] = ess
        AI_ctrl[mask] = ai

    # PV and wind generation
    PV_kW = pv_share * load_kW * (G_irrad / 800.0) \
            * (1 - 0.004 * (T_amb + G_irrad / 800 * 25 - 25))
    PV_kW = np.clip(PV_kW, 0, None)
    wind_kW = wind_share * load_kW * np.clip((wind_ms / 12) ** 3, 0, 1.5)
    RE_kW = PV_kW + wind_kW
    RE_penetration_pct = np.clip(100 * RE_kW / np.clip(load_kW, 1, None), 0, 100)

    # Power quality: THD driven by RE penetration + nonlinear load fraction,
    # then reduced by ESS mitigation and further by AI control.
    v_thd_base = 1.0 + 0.08 * RE_penetration_pct + 4.0 * nonlin_frac
    v_thd = v_thd_base - 2.5 * ESS_on - 1.8 * AI_ctrl \
            + rng.normal(0, 0.35, n_rows)
    v_thd = np.clip(v_thd, 0.1, 20)

    i_thd = v_thd * 1.6 + rng.normal(0, 0.4, n_rows)
    i_thd = np.clip(i_thd, 0.1, 25)

    harm_5th  = i_thd * 0.55 + rng.normal(0, 0.15, n_rows)
    harm_7th  = i_thd * 0.38 + rng.normal(0, 0.12, n_rows)
    harm_11th = i_thd * 0.22 + rng.normal(0, 0.08, n_rows)
    harm_13th = i_thd * 0.18 + rng.normal(0, 0.07, n_rows)

    # Voltage: sag events sprinkled in
    V_a = V_NOMINAL + rng.normal(0, 3, n_rows)
    V_b = V_NOMINAL + rng.normal(0, 3, n_rows)
    V_c = V_NOMINAL + rng.normal(0, 3, n_rows)
    sag_mask = rng.random(n_rows) < 0.02
    V_a[sag_mask] *= 0.70
    V_b[sag_mask] *= 0.72
    V_c[sag_mask] *= 0.71
    V_unbalance_pct = 100 * np.max(
        np.abs(np.stack([V_a, V_b, V_c]) - np.stack([V_a, V_b, V_c]).mean(0)),
        axis=0) / V_NOMINAL

    # Current: from load + nonlinear content
    I_base = load_kW * 1000 / (np.sqrt(3) * V_NOMINAL * 0.9)
    I_a = I_base * (1 + rng.normal(0, 0.03, n_rows))
    I_b = I_base * (1 + rng.normal(0, 0.03, n_rows))
    I_c = I_base * (1 + rng.normal(0, 0.03, n_rows))
    I_neutral = np.abs(I_a - I_b) * 0.15 + rng.normal(0, 1, n_rows)

    # Power flow
    S_apparent_kVA = np.sqrt(3) * V_NOMINAL * I_base / 1000
    pf = 0.95 - 0.35 * nonlin_frac + 0.15 * ESS_on \
         + 0.10 * AI_ctrl + rng.normal(0, 0.02, n_rows)
    pf = np.clip(pf, 0.60, 0.99)
    P_active_kW = S_apparent_kVA * pf
    Q_reactive_kVAR = S_apparent_kVA * np.sqrt(1 - pf ** 2)

    # Frequency: nominal + deviations from RE variability
    freq_dev = 0.02 * (RE_penetration_pct / 100) + rng.normal(0, 0.02, n_rows)
    freq_dev -= 0.03 * ESS_on + 0.02 * AI_ctrl
    freq_Hz = F0_NOMINAL + freq_dev
    RoCoF = np.gradient(freq_Hz) / SAMPLE_DT_SEC

    # Battery
    batt_SOC = np.clip(60 + 20 * np.sin(hour_of_day.values / 24 * 2 * np.pi)
                       + rng.normal(0, 8, n_rows), 5, 100)
    batt_SOH = np.clip(100 - 0.0005 * np.arange(n_rows) / n_rows
                       - 0.02 * i_thd, 70, 100)
    batt_V = 400 + 0.5 * (batt_SOC - 50)
    batt_T_C = T_amb + 5 + 8 * (i_thd / 10) ** 2
    batt_I = P_active_kW * 1000 / batt_V * np.where(ESS_on, 1, 0.2)
    batt_P_kW = batt_V * batt_I / 1000

    # Grid mode
    grid_connected = rng.random(n_rows) > 0.05  # 5 % of the time islanded
    APF_on = AI_ctrl | ESS_on
    fault_flag = rng.random(n_rows) < 0.005

    # IEEE 519 compliance verdict
    compliance = np.where(
        (v_thd < 5) & (i_thd < 5), "PASS",
        np.where((v_thd < 6) & (i_thd < 8), "MARGINAL", "FAIL"))

    # Stability label: unstable if frequency wanders OR any PQ metric is stressed.
    # Thresholds tuned to give ~10 % unstable so classifiers train meaningfully.
    stability_label = ((np.abs(freq_dev) > 0.06) | (v_thd > 4.5)
                       | (i_thd > 6.5) | (V_unbalance_pct > 2.5)
                       | fault_flag).astype(int)

    # Economics
    energy_cost_BDT = P_active_kW * (SAMPLE_DT_SEC / 3600) * 12  # 12 BDT/kWh tariff
    cum_cost_BDT = np.cumsum(energy_cost_BDT)

    df = pd.DataFrame({
        # time
        "timestamp": ts,
        "hour_of_day": hour_of_day,
        "day_of_year": day_of_year,
        # environment
        "irradiance_Wpm2": G_irrad.round(1),
        "ambient_T_C": T_amb.round(2),
        "humidity_pct": humidity.round(1),
        "wind_speed_mps": wind_ms.round(2),
        # voltage
        "V_rms_a_V": V_a.round(2),
        "V_rms_b_V": V_b.round(2),
        "V_rms_c_V": V_c.round(2),
        "V_unbalance_pct": V_unbalance_pct.round(3),
        # current
        "I_rms_a_A": I_a.round(3),
        "I_rms_b_A": I_b.round(3),
        "I_rms_c_A": I_c.round(3),
        "I_neutral_A": I_neutral.round(3),
        # power
        "P_active_kW": P_active_kW.round(3),
        "Q_reactive_kVAR": Q_reactive_kVAR.round(3),
        "S_apparent_kVA": S_apparent_kVA.round(3),
        "power_factor": pf.round(3),
        # frequency
        "freq_Hz": freq_Hz.round(4),
        "freq_dev_Hz": freq_dev.round(4),
        "RoCoF_Hz_per_s": RoCoF.round(5),
        # harmonics
        "V_THD_pct": v_thd.round(3),
        "I_THD_pct": i_thd.round(3),
        "harm_5th_pct": harm_5th.round(3),
        "harm_7th_pct": harm_7th.round(3),
        "harm_11th_pct": harm_11th.round(3),
        "harm_13th_pct": harm_13th.round(3),
        # renewables
        "PV_kW": PV_kW.round(3),
        "wind_kW": wind_kW.round(3),
        "RE_penetration_pct": RE_penetration_pct.round(2),
        # storage
        "batt_SOC_pct": batt_SOC.round(2),
        "batt_SOH_pct": batt_SOH.round(3),
        "batt_V_V": batt_V.round(2),
        "batt_I_A": batt_I.round(3),
        "batt_T_C": batt_T_C.round(2),
        "batt_P_kW": batt_P_kW.round(3),
        # load
        "load_kW": load_kW.round(3),
        "nonlinear_load_frac": nonlin_frac.round(3),
        # mode flags
        "grid_connected": grid_connected,
        "APF_on": APF_on,
        "ESS_on": ESS_on,
        "AI_control_on": AI_ctrl,
        "fault_flag": fault_flag,
        # labels
        "operating_scenario": scen_col,
        "IEEE_519_compliance": compliance,
        "stability_label": stability_label,
        # economics
        "energy_cost_BDT": energy_cost_BDT.round(3),
        "cum_cost_BDT": cum_cost_BDT.round(2),
        # WATERMARK
        "source": "SYNTHETIC_GENERATOR_v1",
    })

    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print(f" SYNTHETIC MICROGRID DATASET  ({len(df):,} rows x {df.shape[1]} cols)")
    print("=" * 72)
    print(f" Time span: {df.timestamp.min()}  ->  {df.timestamp.max()}")
    print(f" Sample resolution: {SAMPLE_DT_SEC} s")
    print()
    print(" Per-scenario counts:")
    print(df["operating_scenario"].value_counts().to_string())
    print()
    print(" IEEE 519 compliance breakdown:")
    print(df["IEEE_519_compliance"].value_counts().to_string())
    print()
    print(f" Stability label balance: "
          f"{100 * df.stability_label.mean():.1f} % unstable")
    print()
    print(" Feature summary (numeric only, top 10):")
    print(df.select_dtypes("number").describe().T.head(10).round(3).to_string())
    print("=" * 72)


def make_plots(df: pd.DataFrame) -> None:
    # 1. Distribution of THD across scenarios
    fig, ax = plt.subplots(figsize=(10, 4.5))
    scenarios = df.operating_scenario.unique()
    data = [df.loc[df.operating_scenario == s, "V_THD_pct"].values
            for s in scenarios]
    bp = ax.boxplot(data, tick_labels=scenarios, patch_artist=True)
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(scenarios)))
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
    ax.axhline(5, color="red", ls="--", lw=1, label="IEEE 519 5 % limit")
    ax.set_ylabel("V-THD (%)"); ax.set_title("Voltage THD per operating scenario")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_vthd_by_scenario.png", dpi=150)
    plt.close()

    # 2. Frequency vs. RE penetration
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(df.RE_penetration_pct, df.freq_dev_Hz, s=2, alpha=0.15,
               color="#2a9d8f")
    ax.set_xlabel("Renewable penetration (%)"); ax.set_ylabel("Freq. deviation (Hz)")
    ax.set_title("Frequency deviation grows with renewable share")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_freq_vs_RE.png", dpi=150)
    plt.close()

    # 3. Daily load + PV + wind sample day
    day1 = df.iloc[:24 * 60]  # first day (1-minute samples)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(day1.hour_of_day, day1.load_kW,
            label="Load", color="#e76f51", lw=1.2)
    ax.plot(day1.hour_of_day, day1.PV_kW,
            label="PV", color="#f4a261", lw=1.2)
    ax.plot(day1.hour_of_day, day1.wind_kW,
            label="Wind", color="#264653", lw=1.2)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Power (kW)")
    ax.set_title("Sample day: load, PV, and wind profile")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_daily_profile.png", dpi=150)
    plt.close()

    # 4. Correlation heatmap of key numerics
    key_cols = ["V_THD_pct", "I_THD_pct", "freq_dev_Hz", "RoCoF_Hz_per_s",
                "RE_penetration_pct", "batt_SOC_pct", "batt_T_C",
                "power_factor", "V_unbalance_pct", "stability_label"]
    corr = df[key_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Correlation of key features (synthetic)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_correlations.png", dpi=150)
    plt.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000

    print(f"\n[generator] producing {n:,} synthetic microgrid rows...")
    df = generate(n_rows=n)

    csv_path = DATA_DIR / "microgrid_synthetic_v1.csv"
    df.to_csv(csv_path, index=False)
    csv_mb = csv_path.stat().st_size / 1024 / 1024
    print(f"[generator] wrote {csv_path.name} ({csv_mb:.1f} MB, uncompressed)")

    # Compressed parquet - smaller, faster to load, git-friendly
    try:
        pq_path = DATA_DIR / "microgrid_synthetic_v1.parquet"
        df.to_parquet(pq_path, compression="snappy", index=False)
        pq_mb = pq_path.stat().st_size / 1024 / 1024
        print(f"[generator] wrote {pq_path.name} ({pq_mb:.1f} MB, "
              f"{csv_mb / pq_mb:.1f}x smaller)")
    except ImportError:
        print("[generator] parquet skipped (install pyarrow for compressed output)")

    summarize(df)
    make_plots(df)
    print(f"[generator] plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
