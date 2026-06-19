"""
microgrid_pq_twin.py
====================
Runnable Python port of build_microgrid_pq_digital_twin.m
(Bangladesh renewable microgrid power-quality digital twin).

This is a *behavioural* simulator - same engineering math as the Simulink
MATLAB-Function blocks, but in numpy + scipy, so it runs on any machine
with Python 3.10+ (no MATLAB/Simulink licence required).

Pipeline:
    PV + weather  ->  Nonlinear load harmonics  ->  SRF Active Power Filter
        ->  FFT-based THD analysis  ->  Battery-degradation & BDT cost twin

Run:
    python eee_sim/microgrid_pq_twin.py

Output:
    figures/eee_sim/01_currents.png
    figures/eee_sim/02_thd_timeseries.png
    figures/eee_sim/03_soh_and_cost.png
    figures/eee_sim/04_spectrum_before_after.png
    results/eee_sim_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eee_sim"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
#  PARAMETERS  (mirror of microgrid_params.m)
# =========================================================================
@dataclass
class Params:
    Ts: float = 1e-4              # solver step  [s] (10 kHz)
    f0: float = 50.0              # grid fundamental [Hz] (Bangladesh)
    Vnom: float = 230.0           # nominal phase RMS [V]
    Tend: float = 1.00            # total sim time [s]

    # voltage sag event
    sag_start: float = 0.30
    sag_dur: float = 0.10
    sag_depth: float = 0.30       # V drops to 70 percent during sag

    # APF activation step
    apf_on_t: float = 0.50

    # nonlinear load (6-pulse rectifier)
    Ifund: float = 15.0
    harm_ord = (5, 7, 11, 13)
    harm_amp = (0.20, 0.14, 0.09, 0.077)

    # PV array (10 kWp)
    Pstc: float = 10_000.0
    gam_T: float = -0.0040
    NOCT: float = 45.0

    # battery + Bangladesh economics
    Batt_kWh: float = 20.0
    cost_kWh: float = 24_000.0    # BDT/kWh replacement
    base_fade: float = 8e-7       # SoH fraction lost per kWh throughput


# =========================================================================
#  PLANT  (3-phase voltage, nonlinear load current, PV power)
# =========================================================================
def plant_step(t: float, G: float, T_amb: float, sag: float, P: Params):
    w0 = 2 * np.pi * P.f0
    Vpk = P.Vnom * np.sqrt(2)
    sag_factor = 1 - P.sag_depth * sag
    th = w0 * t
    Vabc = sag_factor * Vpk * np.array(
        [np.sin(th), np.sin(th - 2 * np.pi / 3), np.sin(th + 2 * np.pi / 3)])

    # fundamental + 5/7/11/13 harmonic load current
    Iabc = P.Ifund * np.array(
        [np.sin(th), np.sin(th - 2 * np.pi / 3), np.sin(th + 2 * np.pi / 3)])
    for n, hm in zip(P.harm_ord, P.harm_amp):
        amp = P.Ifund * hm
        Iabc[0] += amp * np.sin(n * th)
        Iabc[1] += amp * np.sin(n * (th - 2 * np.pi / 3))
        Iabc[2] += amp * np.sin(n * (th + 2 * np.pi / 3))

    # PV power with NOCT cell-temperature derate
    Tcell = T_amb + (G / 800.0) * (P.NOCT - 20.0)
    Ppv = max(0.0, P.Pstc * (G / 1000.0) * (1 + P.gam_T * (Tcell - 25.0)))
    return Vabc, Iabc, Ppv


# =========================================================================
#  SRF ACTIVE POWER FILTER  (Park transform + LPF + inverse Park)
# =========================================================================
class APF:
    def __init__(self, Ts: float, f0: float = 50.0, fc: float = 25.0):
        self.f0 = f0
        self.alpha = 2 * np.pi * fc * Ts / (1 + 2 * np.pi * fc * Ts)
        self.Id_lp = 0.0
        self.Iq_lp = 0.0

    def step(self, Iabc: np.ndarray, t: float, apf_on: float) -> np.ndarray:
        w0 = 2 * np.pi * self.f0
        th = w0 * t
        c0 = np.cos(th); s0 = np.sin(th)
        c_m = np.cos(th - 2 * np.pi / 3); s_m = np.sin(th - 2 * np.pi / 3)
        c_p = np.cos(th + 2 * np.pi / 3); s_p = np.sin(th + 2 * np.pi / 3)

        Id =  (2/3) * (Iabc[0] * c0 + Iabc[1] * c_m + Iabc[2] * c_p)
        Iq = -(2/3) * (Iabc[0] * s0 + Iabc[1] * s_m + Iabc[2] * s_p)

        self.Id_lp += self.alpha * (Id - self.Id_lp)
        self.Iq_lp += self.alpha * (Iq - self.Iq_lp)

        Idh = Id - self.Id_lp
        Iqh = Iq - self.Iq_lp

        Iah = Idh * c0  - Iqh * s0
        Ibh = Idh * c_m - Iqh * s_m
        Ich = Idh * c_p - Iqh * s_p
        Icomp = np.array([Iah, Ibh, Ich])
        return Iabc - apf_on * Icomp


# =========================================================================
#  FFT-BASED THD ANALYZER  (5-cycle window, integer cycles)
# =========================================================================
def thd_window(x: np.ndarray, fs: float) -> float:
    N = len(x)
    df = fs / N
    fund_bin = round(50 / df)
    X = np.abs(np.fft.fft(x)) * (2 / N)
    fund = X[fund_bin]
    if fund < 1e-6:
        return 0.0
    hsum = 0.0
    for k in range(2, 26):
        b = round(k * 50 / df)
        if b < N // 2:
            hsum += X[b] ** 2
    return float(np.sqrt(hsum) / fund)


# =========================================================================
#  DIGITAL TWIN  (THD -> ageing -> BDT cost)
# =========================================================================
class DigitalTwin:
    def __init__(self, P: Params):
        self.P = P
        self.SoH_loss = 0.0

    def step(self, THD_i: float, Ppv: float, t: float) -> dict:
        P = self.P
        Pbatt_kW = max(Ppv, 1.0) / 1000.0
        dE_kWh = Pbatt_kW * P.Ts / 3600.0
        dT = 25 * THD_i ** 2
        accel = 2 ** (dT / 10.0)
        self.SoH_loss += P.base_fade * dE_kWh * accel
        year_scale = (365 * 24 * 3600) / max(t, P.Ts)
        cost_BDT_yr = self.SoH_loss * P.Batt_kWh * P.cost_kWh * year_scale
        return {"SoH_loss_pct": self.SoH_loss * 100,
                "batt_T": 30 + dT,
                "cost_BDT_yr": cost_BDT_yr}


# =========================================================================
#  DRIVER
# =========================================================================
def run(P: Params | None = None) -> pd.DataFrame:
    if P is None:
        P = Params()

    fs = 1 / P.Ts
    N_total = int(P.Tend / P.Ts) + 1
    t_arr = np.arange(N_total) * P.Ts

    # weather profiles: rising irradiance, slow temperature drift
    G_arr = np.linspace(800, 950, N_total)
    T_arr = np.linspace(32, 34, N_total)

    apf = APF(Ts=P.Ts, f0=P.f0)
    twin = DigitalTwin(P)

    Va_log = np.zeros(N_total)
    Ia_load_log = np.zeros(N_total)
    Ia_source_log = np.zeros(N_total)
    Ppv_log = np.zeros(N_total)
    THD_v_log = np.zeros(N_total)
    THD_i_log = np.zeros(N_total)
    cost_log = np.zeros(N_total)
    SoH_log = np.zeros(N_total)

    Nfft = 1000              # 5 cycles at 10 kHz
    v_buf = np.zeros(Nfft)
    i_buf = np.zeros(Nfft)
    buf_idx = 0
    thd_v_cur = 0.0
    thd_i_cur = 0.0

    for k in range(N_total):
        t = t_arr[k]
        G = G_arr[k]
        T_amb = T_arr[k]
        sag = 1.0 if (P.sag_start <= t < P.sag_start + P.sag_dur) else 0.0
        apf_on = 1.0 if t >= P.apf_on_t else 0.0

        Vabc, Iabc, Ppv = plant_step(t, G, T_amb, sag, P)
        Isource = apf.step(Iabc, t, apf_on)

        v_buf[buf_idx] = Vabc[0]
        i_buf[buf_idx] = Isource[0]
        buf_idx += 1
        if buf_idx >= Nfft:
            thd_v_cur = thd_window(v_buf, fs)
            thd_i_cur = thd_window(i_buf, fs)
            buf_idx = 0

        tw_out = twin.step(thd_i_cur, Ppv, t)

        Va_log[k] = Vabc[0]
        Ia_load_log[k] = Iabc[0]
        Ia_source_log[k] = Isource[0]
        Ppv_log[k] = Ppv
        THD_v_log[k] = thd_v_cur
        THD_i_log[k] = thd_i_cur
        cost_log[k] = tw_out["cost_BDT_yr"]
        SoH_log[k] = tw_out["SoH_loss_pct"]

    df = pd.DataFrame({
        "t": t_arr, "Va": Va_log,
        "Ia_load": Ia_load_log, "Ia_source": Ia_source_log,
        "Ppv": Ppv_log, "THD_v": THD_v_log, "THD_i": THD_i_log,
        "SoH_loss_pct": SoH_log, "cost_BDT_yr": cost_log,
    })

    # =====================================================================
    #  ACCEPTANCE NUMBERS  (the script header's promise)
    # =====================================================================
    print("\n=========== Bangladesh PV Microgrid PQ Twin (Python) ===========")
    pre  = df.loc[(df.t > 0.10) & (df.t < P.apf_on_t - 0.05), "THD_i"]
    post = df.loc[df.t > P.apf_on_t + 0.10, "THD_i"]
    print(f"Total simulated time     : {P.Tend:.2f} s "
          f"({N_total} samples @ {fs:.0f} Hz)")
    print(f"APF turn-on time         : t = {P.apf_on_t:.2f} s")
    print(f"THD_i BEFORE APF (mean)  : {pre.mean()*100:6.2f} %")
    print(f"THD_i AFTER  APF (mean)  : {post.mean()*100:6.2f} %")
    print(f"THD_v BEFORE APF (mean)  : {df.loc[df.t < P.apf_on_t, 'THD_v'].mean()*100:6.2f} %")
    print(f"PV power range           : {Ppv_log.min():.0f} - {Ppv_log.max():.0f} W")
    print(f"Final cumulative SoH loss: {SoH_log[-1]:.6f} %")
    print(f"Projected battery cost   : {cost_log[-1]:.0f} BDT/yr "
          f"(BDT {cost_log[-1]/12:.0f}/month)")
    print("===============================================================\n")

    # =====================================================================
    #  PLOTS
    # =====================================================================
    around = (df.t > P.apf_on_t - 0.06) & (df.t < P.apf_on_t + 0.06)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.loc[around, "t"], df.loc[around, "Ia_load"],
            color="#e76f51", lw=1.0, label="Load current $i_a$ (uncompensated)")
    ax.plot(df.loc[around, "t"], df.loc[around, "Ia_source"],
            color="#264653", lw=1.0, label="Source current $i_a$ (after APF)")
    ax.axvline(P.apf_on_t, color="green", lw=1, ls="--", label="APF ON")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Current (A)")
    ax.set_title("Phase-A current: APF turn-on transient (zoom $\\pm 60$ ms)")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_currents.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.t, df.THD_i * 100, color="#e76f51", lw=1.5,
            label="$\\mathrm{THD}_i$ (source current)")
    ax.plot(df.t, df.THD_v * 100, color="#2a9d8f", lw=1.0,
            label="$\\mathrm{THD}_v$ (voltage)")
    ax.axhline(5, color="black", ls=":", lw=1, label="IEEE 519 limit (5 %)")
    ax.axvline(P.apf_on_t, color="green", lw=1, ls="--", label="APF ON")
    ax.axvspan(P.sag_start, P.sag_start + P.sag_dur,
               color="orange", alpha=0.2, label="30 %  voltage sag")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("THD (%)")
    ax.set_title("Total Harmonic Distortion vs. time")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_thd_timeseries.png", dpi=150)
    plt.close()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(df.t, df.SoH_loss_pct, color="#e76f51", lw=1.5)
    a1.set_xlabel("Time (s)"); a1.set_ylabel("Cumulative SoH loss (%)")
    a1.set_title("Battery state-of-health degradation")
    a1.grid(alpha=0.3)
    a2.plot(df.t, df.cost_BDT_yr / 1000, color="#264653", lw=1.5)
    a2.set_xlabel("Time (s)"); a2.set_ylabel("Projected cost (kBDT/yr)")
    a2.set_title("Projected annual battery replacement cost (Bangladesh tariff)")
    a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_soh_and_cost.png", dpi=150)
    plt.close()

    pre_win  = df.loc[df.t >= 0.20, "Ia_load"].values[:Nfft]
    post_win = df.loc[df.t >= P.apf_on_t + 0.15, "Ia_source"].values[:Nfft]
    if len(pre_win) >= Nfft and len(post_win) >= Nfft:
        pre_X  = np.abs(np.fft.fft(pre_win))[:Nfft // 2] * (2 / Nfft)
        post_X = np.abs(np.fft.fft(post_win))[:Nfft // 2] * (2 / Nfft)
        freqs  = np.fft.fftfreq(Nfft, P.Ts)[:Nfft // 2]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stem(freqs, pre_X,  linefmt="#e76f51", markerfmt=" ",
                basefmt=" ", label="Before APF")
        ax.stem(freqs, post_X, linefmt="#264653", markerfmt=" ",
                basefmt=" ", label="After APF")
        ax.set_xlim(0, 1500)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude (A)")
        ax.set_title("Source-current spectrum: APF effect on 5th, 7th, 11th, 13th harmonics")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(FIG_DIR / "04_spectrum_before_after.png", dpi=150)
        plt.close()

    # =====================================================================
    #  SUMMARY CSV
    # =====================================================================
    summary = {
        "thd_i_pct_before_APF":  float(pre.mean() * 100),
        "thd_i_pct_after_APF":   float(post.mean() * 100),
        "thd_v_pct_before_APF":  float(df.loc[df.t < P.apf_on_t, "THD_v"].mean() * 100),
        "PV_peak_W":             float(Ppv_log.max()),
        "SoH_loss_final_pct":    float(SoH_log[-1]),
        "cost_BDT_per_yr_final": float(cost_log[-1]),
        "T_end_s":               P.Tend,
        "fs_Hz":                 fs,
    }
    pd.DataFrame([summary]).to_csv(RES_DIR / "eee_sim_summary.csv", index=False)
    print(f"Saved figs : {FIG_DIR}")
    print(f"Saved CSV  : {RES_DIR / 'eee_sim_summary.csv'}")
    return df


if __name__ == "__main__":
    run()
