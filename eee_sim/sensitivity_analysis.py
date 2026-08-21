"""
sensitivity_analysis.py
=========================
The EEE twin's cost projection (microgrid_pq_twin.py) reports a single
deterministic number (~27,000 BDT/yr) from assumed, uncited parameters:
battery replacement cost per kWh (cost_kWh) and the base capacity-fade
rate (base_fade). The CSE side reports every accuracy number as a
mean +/- std over 5 seeds with bootstrap CIs; the EEE side has had no
equivalent uncertainty treatment until now.

This runs the twin many times with cost_kWh and base_fade independently
perturbed +/-20% (Monte Carlo, uniform), holding the electrical physics
fixed, and reports the resulting spread in projected cost and SoH loss -
so the paper can say "27,000 BDT/yr, 90% CI [X, Y]" instead of implying
false precision on assumed inputs.

Run:
    python eee_sim/sensitivity_analysis.py [n_runs]

Output:
    results/eee_sim_sensitivity.csv
    figures/eee_sim/08_cost_sensitivity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microgrid_pq_twin import APF, DigitalTwin, Params, plant_step, thd_window

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "eee_sim"; FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(parents=True, exist_ok=True)

# baseline assumed values from microgrid_pq_twin.Params, +/-20% Monte Carlo range
BASELINE_COST_KWH = 24_000.0     # BDT/kWh replacement
BASELINE_BASE_FADE = 8e-7        # SoH fraction lost per kWh throughput
PERTURB_PCT = 0.20


def run_once(cost_kwh: float, base_fade: float) -> dict:
    P = Params()
    P.cost_kWh = cost_kwh
    P.base_fade = base_fade

    fs = 1 / P.Ts
    N_total = int(P.Tend / P.Ts) + 1
    t_arr = np.arange(N_total) * P.Ts
    G_arr = np.linspace(800, 950, N_total)
    T_arr = np.linspace(32, 34, N_total)

    apf = APF(Ts=P.Ts, f0=P.f0)
    twin = DigitalTwin(P)
    Nfft = 1000
    v_buf = np.zeros(Nfft); i_buf = np.zeros(Nfft); buf_idx = 0
    thd_i_cur = 0.0
    cost_final = 0.0; soh_final = 0.0

    for k in range(N_total):
        t = t_arr[k]
        sag = 1.0 if (P.sag_start <= t < P.sag_start + P.sag_dur) else 0.0
        apf_on = 1.0 if t >= P.apf_on_t else 0.0
        Vabc, Iabc, Ppv = plant_step(t, G_arr[k], T_arr[k], sag, P)
        Isource = apf.step(Iabc, t, apf_on)
        v_buf[buf_idx] = Vabc[0]; i_buf[buf_idx] = Isource[0]; buf_idx += 1
        if buf_idx >= Nfft:
            thd_i_cur = thd_window(i_buf, fs)
            buf_idx = 0
        out = twin.step(thd_i_cur, Ppv, t)
        cost_final, soh_final = out["cost_BDT_yr"], out["SoH_loss_pct"]

    return {"cost_kWh": cost_kwh, "base_fade": base_fade,
            "cost_BDT_yr": cost_final, "SoH_loss_final_pct": soh_final}


def run(n_runs: int = 200):
    rng = np.random.default_rng(42)
    cost_kwh_samples = rng.uniform(
        BASELINE_COST_KWH * (1 - PERTURB_PCT), BASELINE_COST_KWH * (1 + PERTURB_PCT), n_runs)
    base_fade_samples = rng.uniform(
        BASELINE_BASE_FADE * (1 - PERTURB_PCT), BASELINE_BASE_FADE * (1 + PERTURB_PCT), n_runs)

    rows = [run_once(c, f) for c, f in zip(cost_kwh_samples, base_fade_samples)]
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "eee_sim_sensitivity.csv", index=False)

    baseline = run_once(BASELINE_COST_KWH, BASELINE_BASE_FADE)
    cost_vals = df["cost_BDT_yr"].values
    ci_lo, ci_hi = np.percentile(cost_vals, [5, 95])

    print(f"\n=========== Cost sensitivity to assumed parameters ({n_runs} runs, "
          f"+/-{PERTURB_PCT*100:.0f}% on cost_kWh and base_fade) ===========")
    print(f"Baseline (point estimate): {baseline['cost_BDT_yr']:.0f} BDT/yr")
    print(f"Monte Carlo mean:          {cost_vals.mean():.0f} BDT/yr")
    print(f"Monte Carlo std:           {cost_vals.std():.0f} BDT/yr")
    print(f"90% interval:              [{ci_lo:.0f}, {ci_hi:.0f}] BDT/yr")
    print("===========================================================================\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.hist(cost_vals, bins=25, color="#264653", alpha=0.8)
    a1.axvline(baseline["cost_BDT_yr"], color="#e76f51", lw=2, ls="--", label="baseline point estimate")
    a1.axvline(ci_lo, color="grey", lw=1, ls=":", label="90% interval")
    a1.axvline(ci_hi, color="grey", lw=1, ls=":")
    a1.set_xlabel("Projected battery cost (BDT/yr)")
    a1.set_ylabel("count")
    a1.set_title(f"Cost distribution under +/-{PERTURB_PCT*100:.0f}% parameter uncertainty")
    a1.legend(fontsize=8)

    sc = a2.scatter(df["cost_kWh"], df["base_fade"], c=df["cost_BDT_yr"], cmap="viridis", s=18)
    a2.set_xlabel("assumed cost_kWh (BDT/kWh)")
    a2.set_ylabel("assumed base_fade (SoH frac / kWh throughput)")
    a2.set_title("Cost outcome across the assumed-parameter space")
    plt.colorbar(sc, ax=a2, label="cost (BDT/yr)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_cost_sensitivity.png", dpi=150)
    plt.close()

    print(f"Saved: {RES_DIR / 'eee_sim_sensitivity.csv'}")
    print(f"Saved: {FIG_DIR / '08_cost_sensitivity.png'}")
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(n)
