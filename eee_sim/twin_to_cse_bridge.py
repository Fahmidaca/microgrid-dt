"""
twin_to_cse_bridge.py
=======================
Closes the loop between the EEE side (physics digital twin) and the
CSE side (ML trained on real Jamalpur field data), which until now
have been two separate pipelines running on unrelated data.

What this does: injects four kinds of disturbance into the EEE
electrical-level twin (None / Harmonic_Distortion / Voltage_Sag /
Combined_Weather_Electrical - the same four classes the Part 6
disturbance dataset uses), extracts the same 10-feature vector the
CSE disturbance_classifier.py and anomaly_detection.py were trained
on, and asks the *already-trained-on-real-field-data* models to
classify the *physics-simulated* waveform. This tests whether a
classifier trained on real Jamalpur readings generalizes to a
first-principles simulation of the same phenomena - a genuine
sim-to-real cross-check, not just two demos bolted under one title.

Feature extraction point: pre-APF (the raw nonlinear-load current),
since the Jamalpur sensor has no local mitigation device - detecting
the disturbance is the CSE side's job, upstream of any EEE-side
suppression. This also sets up the natural "digital twin loop" story:
CSE detects -> EEE's APF (see microgrid_pq_twin.py) suppresses.

Requires: models/disturbance/RF_seed0.pkl must exist
(run src/disturbance_classifier.py first).

Run:
    python eee_sim/twin_to_cse_bridge.py

Output:
    results/twin_to_cse_bridge_predictions.csv
    results/twin_to_cse_bridge_domain_shift.csv
    figures/eee_sim/07_bridge_confusion_matrix.png
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from microgrid_pq_twin import Params, plant_step

# microgrid_pq_twin.plant_step assumes an ideal (zero-impedance) voltage
# source, so THD_voltage_pct is always exactly 0 - structurally impossible
# for load-current harmonics to show up on the voltage waveform. Real grids
# have nonzero source impedance, which is exactly why real Jamalpur data
# shows THD_voltage_pct as the single biggest discriminator between
# Harmonic_Distortion (mean 11.3%) and Voltage_Sag (mean 2.9%). We add a
# standard linearly-frequency-scaling source reactance (X_h = h * X1, the
# textbook approximation for an inductive Thevenin source) on top of the
# existing plant_step, rather than touching the shared function itself so
# every other script's already-verified results are untouched.
_X1_OHM = 1.6  # fundamental-frequency source reactance (assumed, not measured)


def plant_step_with_source_impedance(t, G, T_amb, sag, P):
    Vabc, Iabc, Ppv = plant_step(t, G, T_amb, sag, P)
    w0 = 2 * np.pi * P.f0
    th = w0 * t
    Vharm = np.zeros(3)
    for n, hm in zip(P.harm_ord, P.harm_amp):
        Iharm_amp = P.Ifund * hm
        Vh = Iharm_amp * (n * _X1_OHM)
        Vharm[0] += Vh * np.sin(n * th)
        Vharm[1] += Vh * np.sin(n * (th - 2 * np.pi / 3))
        Vharm[2] += Vh * np.sin(n * (th + 2 * np.pi / 3))
    return Vabc + Vharm, Iabc, Ppv

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eee_sim"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)
DISTURBANCE_MODEL = ROOT / "models" / "disturbance" / "RF_seed0.pkl"
JAMALPUR_CSV = ROOT / "data" / "external" / "microgrid_power_quality_dataset.csv"

FEATURE_COLS = [
    "voltage_rms_V", "current_rms_A", "frequency_Hz",
    "temperature_C", "irradiance_Wm2",
    "harmonic_3rd_pct", "harmonic_5th_pct", "harmonic_7th_pct",
    "THD_voltage_pct", "THD_current_pct",
]

RNG = np.random.default_rng(42)
N_PER_CLASS = 25


def harmonic_ratios(x: np.ndarray, fs: float, orders=(3, 5, 7)) -> dict:
    """FFT-bin magnitude at each harmonic order, relative to the fundamental."""
    N = len(x)
    df = fs / N
    X = np.abs(np.fft.fft(x)) * (2 / N)
    fund_bin = round(50 / df)
    fund = X[fund_bin]
    if fund < 1e-6:
        return {k: 0.0 for k in orders}
    out = {}
    for k in orders:
        b = round(k * 50 / df)
        out[k] = float(X[b] / fund) if b < N // 2 else 0.0
    return out


def thd_of(x: np.ndarray, fs: float) -> float:
    N = len(x)
    df = fs / N
    X = np.abs(np.fft.fft(x)) * (2 / N)
    fund = X[round(50 / df)]
    if fund < 1e-6:
        return 0.0
    hsum = sum(X[round(k * 50 / df)] ** 2 for k in range(2, 26) if round(k * 50 / df) < N // 2)
    return float(np.sqrt(hsum) / fund)


#  Calibration: THD = sqrt(sum(harm_amp_i^2)), independent of Ifund/load level
#  (this is exactly why THD came out scale-invariant to load in the scenario
#  study - see README Part 5). The uncalibrated default harm_amp=(0.20, 0.14,
#  0.09, 0.077) gives THD = 27.13%, far above anything in the real Jamalpur
#  data (baseline ~2-4%, disturbance rows ~7% per docs/DATA_PROVENANCE). We
#  scale the *same* 6-pulse harmonic-order shape (5th > 7th > 11th > 13th,
#  the physically justified part) down to real-world THD magnitudes, rather
#  than inventing a new curve.
_BASE_HARM_AMP = (0.20, 0.14, 0.09, 0.077)
_BASE_THD = float(np.sqrt(sum(a ** 2 for a in _BASE_HARM_AMP)))  # 0.2713
_TARGET_THD_PCT = {
    "None": 3.0,
    "Harmonic_Distortion": 9.0,
    "Voltage_Sag": 3.0,                    # a sag is a magnitude event, not a harmonic one
    "Combined_Weather_Electrical": 7.0,
}


def simulate_one(scenario: str, rng: np.random.Generator) -> dict:
    """Run ~0.3s of the (pre-APF) electrical twin under one scenario, with
    per-repetition random jitter, and extract the 10-feature CSE vector."""
    P = Params()
    P.Tend = 0.30
    P.sag_start = 999.0   # disabled unless overridden below
    P.apf_on_t = 999.0    # never turns on - we want the raw, unmitigated signal

    if scenario == "None":
        irradiance = rng.uniform(250, 550)
        temp = rng.uniform(28, 32)
        load_scale = rng.uniform(0.8, 1.1)
        sag_active, sag_depth = False, 0.0
    elif scenario == "Harmonic_Distortion":
        irradiance = rng.uniform(200, 500)
        temp = rng.uniform(28, 33)
        load_scale = rng.uniform(1.4, 2.0)          # heavy nonlinear load
        sag_active, sag_depth = False, 0.0
    elif scenario == "Voltage_Sag":
        irradiance = rng.uniform(30, 150)             # renewable fluctuation
        temp = rng.uniform(28, 32)
        load_scale = rng.uniform(0.8, 1.1)
        sag_active, sag_depth = True, rng.uniform(0.20, 0.40)
    elif scenario == "Combined_Weather_Electrical":
        irradiance = rng.uniform(10, 100)
        temp = rng.uniform(31, 36)
        load_scale = rng.uniform(1.3, 1.8)
        sag_active, sag_depth = True, rng.uniform(0.15, 0.30)
    else:
        raise ValueError(scenario)

    P.Ifund = 15.0 * load_scale
    target_thd_pct = _TARGET_THD_PCT[scenario] * rng.uniform(0.85, 1.15)
    k = (target_thd_pct / 100.0) / _BASE_THD
    P.harm_amp = tuple(a * k for a in _BASE_HARM_AMP)
    if sag_active:
        P.sag_start, P.sag_dur, P.sag_depth = 0.10, 0.10, sag_depth
    else:
        P.sag_start, P.sag_dur = 0.10, 0.10  # window below still lines up; depth=0 (no dip) since sag_active=False keeps default flag off

    fs = 1 / P.Ts
    N_total = int(P.Tend / P.Ts) + 1
    t_arr = np.arange(N_total) * P.Ts
    Va = np.zeros(N_total); Ia = np.zeros(N_total)
    for k_i in range(N_total):
        t = t_arr[k_i]
        sag = 1.0 if (sag_active and P.sag_start <= t < P.sag_start + P.sag_dur) else 0.0
        Vabc, Iabc, _ = plant_step_with_source_impedance(t, irradiance, temp, sag, P)
        Va[k_i] = Vabc[0]; Ia[k_i] = Iabc[0]

    # analysis window = exactly the [sag_start, sag_start+sag_dur) interval,
    # so sag scenarios are actually captured mid-event, not after it ends
    w0, w1 = int(P.sag_start / P.Ts), int((P.sag_start + P.sag_dur) / P.Ts)
    Va_w, Ia_w = Va[w0:w1], Ia[w0:w1]
    v_rms = float(np.sqrt(np.mean(Va_w ** 2)))
    i_rms = float(np.sqrt(np.mean(Ia_w ** 2)))
    h = harmonic_ratios(Ia_w, fs, orders=(3, 5, 7))
    thd_v = thd_of(Va_w, fs) * 100
    thd_i = thd_of(Ia_w, fs) * 100

    return {
        "voltage_rms_V": v_rms,
        "current_rms_A": i_rms,
        "frequency_Hz": float(rng.normal(50.0, 0.02)),  # twin has no freq dynamics; matches Jamalpur's own tight spread
        "temperature_C": temp,
        "irradiance_Wm2": irradiance,
        "harmonic_3rd_pct": h[3] * 100,
        "harmonic_5th_pct": h[5] * 100,
        "harmonic_7th_pct": h[7] * 100,
        "THD_voltage_pct": thd_v,
        "THD_current_pct": thd_i,
        "true_scenario": scenario,
    }


def run():
    if not DISTURBANCE_MODEL.exists():
        raise FileNotFoundError(
            f"{DISTURBANCE_MODEL} not found - run src/disturbance_classifier.py first")

    scenarios = ["None", "Harmonic_Distortion", "Voltage_Sag", "Combined_Weather_Electrical"]
    rows = [simulate_one(s, RNG) for s in scenarios for _ in range(N_PER_CLASS)]
    df = pd.DataFrame(rows)

    with open(DISTURBANCE_MODEL, "rb") as f:
        saved = pickle.load(f)
    clf, le = saved["model"], saved["label_encoder"]

    X = df[FEATURE_COLS].values
    y_true = df["true_scenario"].values
    y_pred = le.inverse_transform(clf.predict(X))

    df["predicted_scenario"] = y_pred
    df.to_csv(RES_DIR / "twin_to_cse_bridge_predictions.csv", index=False)

    acc = accuracy_score(y_true, y_pred)
    print("\n=========== EEE twin -> CSE disturbance classifier (sim-to-real test) ===========")
    print(f"Overall accuracy: {acc:.3f}  (n={len(df)}, {N_PER_CLASS} simulated repetitions/class)")
    print(classification_report(y_true, y_pred))

    cm_labels = scenarios
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=cm_labels, ax=ax, xticks_rotation=45, colorbar=False,
        text_kw={"fontsize": 14})
    ax.set_title("CSE classifier (trained on real Jamalpur data)\napplied to EEE twin simulations",
                  fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_bridge_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- anomaly detector: does it flag disturbed windows more than baseline? ---
    jam = pd.read_csv(JAMALPUR_CSV)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(jam[FEATURE_COLS].values)
    df["anomaly_flag"] = (iso.predict(X) == -1)
    anomaly_rate = df.groupby("true_scenario")["anomaly_flag"].mean().reindex(scenarios)
    print("\n--- Anomaly-flag rate per scenario (Isolation Forest trained on real Jamalpur data) ---")
    print(anomaly_rate.round(3).to_string())

    # --- domain-shift check: how far are simulated features from real Jamalpur ranges? ---
    shift = pd.DataFrame({
        "feature": FEATURE_COLS,
        "jamalpur_mean": [jam[c].mean() for c in FEATURE_COLS],
        "jamalpur_std": [jam[c].std() for c in FEATURE_COLS],
        "twin_mean": [df[c].mean() for c in FEATURE_COLS],
        "twin_std": [df[c].std() for c in FEATURE_COLS],
    })
    shift["mean_gap_in_jamalpur_std"] = (
        (shift["twin_mean"] - shift["jamalpur_mean"]) / shift["jamalpur_std"].replace(0, np.nan)
    ).round(2)
    shift.to_csv(RES_DIR / "twin_to_cse_bridge_domain_shift.csv", index=False)
    print("\n--- Domain-shift check (twin-simulated vs real Jamalpur feature ranges) ---")
    print(shift.round(3).to_string(index=False))

    print(f"\nSaved: {RES_DIR / 'twin_to_cse_bridge_predictions.csv'}")
    print(f"Saved: {RES_DIR / 'twin_to_cse_bridge_domain_shift.csv'}")
    print(f"Saved: {FIG_DIR / '07_bridge_confusion_matrix.png'}")
    return df


if __name__ == "__main__":
    run()
