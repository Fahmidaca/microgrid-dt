"""
Module: Signal Processing (EEE)
--------------------------------
Loads the power quality dataset and derives / validates the signal-level
features described in Section 5 of the proposal:
    - THD (already present in the dataset as THD_voltage_pct / THD_current_pct)
    - Harmonic components (3rd, 5th, 7th - already present)
    - RMS values (voltage_rms_V, current_rms_A - already present)
    - Frequency deviation feature (derived: |f - 50| Hz)

Since the uploaded dataset already provides RMS/THD/harmonic columns directly
(rather than raw waveform samples), this module focuses on:
    1. Loading + validating the dataset schema
    2. Deriving a couple of useful engineered features (frequency deviation,
       harmonic-to-fundamental ratio) that the ML modules downstream will use
    3. Producing a clean, typed dataframe ready for anomaly detection /
       prediction modules

If you later add raw waveform samples (voltage/current time series at a
sampling rate), swap load_dataset() for a waveform loader and use
extract_features_from_waveform() instead - it's included below and uses
numpy's FFT so you don't have to rewrite the rest of the pipeline.
"""

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "voltage_rms_V", "current_rms_A", "frequency_Hz", "temperature_C",
    "irradiance_Wm2", "harmonic_3rd_pct", "harmonic_5th_pct", "harmonic_7th_pct",
    "THD_voltage_pct", "THD_current_pct", "sensor_fault_flag",
    "disturbance_type", "disturbance_label", "battery_degradation_rate",
    "battery_capacity_loss_pct", "economic_cost_BDT",
]

NOMINAL_FREQ_HZ = 50.0


def load_dataset(path: str) -> pd.DataFrame:
    """Load the power quality CSV and validate its schema."""
    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # disturbance_type is blank (NaN) for non-disturbance rows by convention
    # in this dataset - make that explicit rather than leaving NaN around.
    df["disturbance_type"] = df["disturbance_type"].fillna("None")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived signal-processing features used by downstream modules."""
    out = df.copy()

    # Frequency deviation from nominal - a classic power quality indicator
    out["freq_deviation_Hz"] = (out["frequency_Hz"] - NOMINAL_FREQ_HZ).abs()

    # Aggregate harmonic distortion proxy (sum of measured harmonic orders)
    out["harmonic_sum_pct"] = (
        out["harmonic_3rd_pct"] + out["harmonic_5th_pct"] + out["harmonic_7th_pct"]
    )

    # Voltage/current THD ratio - flags cases where current distortion
    # dominates (common with nonlinear loads / inverter switching)
    out["thd_v_to_i_ratio"] = out["THD_voltage_pct"] / out["THD_current_pct"].replace(0, np.nan)
    out["thd_v_to_i_ratio"] = out["thd_v_to_i_ratio"].fillna(0)

    return out


def extract_features_from_waveform(
    voltage_samples: np.ndarray, sampling_rate_hz: float, fundamental_hz: float = 50.0
) -> dict:
    """
    Optional path: if raw waveform samples become available later, use this
    to derive RMS, THD, and harmonic magnitudes via FFT instead of relying on
    pre-computed columns.

    Parameters
    ----------
    voltage_samples : 1D array of a single-cycle-or-longer voltage waveform
    sampling_rate_hz : sampling rate of voltage_samples
    fundamental_hz : expected fundamental frequency (50 or 60 Hz)

    Returns
    -------
    dict with rms, thd_pct, and harmonic_2..7_pct
    """
    n = len(voltage_samples)
    rms = float(np.sqrt(np.mean(voltage_samples ** 2)))

    fft_vals = np.fft.rfft(voltage_samples)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    magnitudes = np.abs(fft_vals) / n * 2

    def nearest_bin(target_freq):
        return int(np.argmin(np.abs(fft_freqs - target_freq)))

    fundamental_mag = magnitudes[nearest_bin(fundamental_hz)]
    harmonics = {}
    for order in range(2, 8):
        harmonics[f"harmonic_{order}_pct"] = (
            100.0 * magnitudes[nearest_bin(fundamental_hz * order)] / fundamental_mag
            if fundamental_mag > 0 else 0.0
        )

    thd_pct = 100.0 * np.sqrt(
        sum((magnitudes[nearest_bin(fundamental_hz * k)]) ** 2 for k in range(2, 8))
    ) / fundamental_mag if fundamental_mag > 0 else 0.0

    return {"rms": rms, "thd_pct": thd_pct, **harmonics}


if __name__ == "__main__":
    df = load_dataset("/mnt/user-data/uploads/microgrid_power_quality_dataset.csv")
    df = engineer_features(df)
    print(df.shape)
    print(df[["frequency_Hz", "freq_deviation_Hz", "harmonic_sum_pct", "thd_v_to_i_ratio"]].describe())
    df.to_csv("/home/user/pipeline/features_stage1.csv", index=False)
    print("Saved features_stage1.csv")
