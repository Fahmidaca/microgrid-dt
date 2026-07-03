"""
Time-Series Forecasting of Grid Power Quality (V-THD)
=====================================================

Multi-horizon forecast of Voltage THD from the past 30 minutes of grid
operational signals. Four models are compared:

    1. Persistence  (naive: y_hat(t+k) = y(t))
    2. MLP          (flat window -> dense)
    3. LSTM
    4. GRU

Forecast horizons: 5, 15, and 30 minutes ahead.

Why this matters for the "digital twin" story:
- Grid operators care about seeing THD drift toward the IEEE 519 5% limit
  BEFORE it violates. A 15-min lead-time enables preemptive ESS or APF
  dispatch.
- Persistence is the honest zero-cost baseline. Any deep model must beat it.
- We additionally report early-warning F1 for the binary event
  "V-THD will breach 5% within the next horizon".

>>>>>  WARNING - trained on SYNTHETIC data. See data/synthetic/README.md.
        The pipeline is production-quality; the numbers are pipeline
        validation only. Replace with real Simulink output before publication.

Run:
    python src/forecasting.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "microgrid_synthetic_v1.parquet"
RESULTS_DIR = ROOT / "results"; RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures" / "forecasting"; FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "forecasting"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LOOKBACK = 30                          # 30-minute window
HORIZONS = [5, 15, 30]                 # minutes ahead
IEEE519_LIMIT = 5.0                    # V-THD % breach threshold
BATCH = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input features (excluding target) - operational signals a grid controller sees.
INPUT_FEATURES = [
    "irradiance_Wpm2", "ambient_T_C", "humidity_pct", "wind_speed_mps",
    "V_rms_a_V", "V_rms_b_V", "V_rms_c_V", "V_unbalance_pct",
    "I_rms_a_A", "I_rms_b_A", "I_rms_c_A", "I_neutral_A",
    "P_active_kW", "Q_reactive_kVAR", "power_factor",
    "freq_Hz", "freq_dev_Hz", "RoCoF_Hz_per_s",
    "PV_kW", "wind_kW", "RE_penetration_pct",
    "batt_SOC_pct", "batt_T_C", "batt_P_kW",
    "load_kW", "nonlinear_load_frac",
]
TARGET = "V_THD_pct"


# =========================================================================
#  DATA
# =========================================================================
def load_and_prepare() -> tuple[pd.DataFrame, np.ndarray]:
    print(f"[data] loading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
    assert (df["source"] == "SYNTHETIC_GENERATOR_v1").all()
    print(f"[data] {len(df):,} rows in chronological order")
    return df, df[TARGET].values


def make_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where X has shape (N, LOOKBACK, F) and y has shape (N, len(HORIZONS))."""
    X_full = df[INPUT_FEATURES + [TARGET]].values.astype(np.float32)
    N = len(df)
    max_h = max(HORIZONS)
    valid_len = N - LOOKBACK - max_h

    X = np.zeros((valid_len, LOOKBACK, X_full.shape[1]), dtype=np.float32)
    y = np.zeros((valid_len, len(HORIZONS)), dtype=np.float32)
    for i in range(valid_len):
        X[i] = X_full[i : i + LOOKBACK]
        for j, h in enumerate(HORIZONS):
            y[i, j] = X_full[i + LOOKBACK + h - 1, -1]  # -1 = TARGET column
    return X, y


def chronological_split(X: np.ndarray, y: np.ndarray, val_frac=0.1, test_frac=0.2):
    N = len(X)
    n_test  = int(N * test_frac)
    n_val   = int(N * val_frac)
    n_train = N - n_val - n_test
    return ((X[:n_train],       y[:n_train]),
            (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            (X[n_train + n_val:],        y[n_train + n_val:]))


# =========================================================================
#  MODELS
# =========================================================================
class PersistenceModel:
    """Predict y_hat(t+k) = y(t) for all horizons."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        last_target = X[:, -1, -1]                       # last observed V-THD
        return np.stack([last_target] * len(HORIZONS), axis=1)


class MLPForecaster(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )
    def forward(self, x): return self.net(x)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, out_dim: int, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])                  # last-timestep hidden


class GRUForecaster(nn.Module):
    def __init__(self, n_features: int, out_dim: int, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2,
                          batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# =========================================================================
#  TRAINING
# =========================================================================
def train_pytorch(model, tr_loader, va_loader, name: str) -> nn.Module:
    model = model.to(DEVICE)
    optim_ = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    hist_tr, hist_va = [], []
    since = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim_.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim_.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(loss_fn(model(xb), yb).item())
        va_loss = float(np.mean(val_losses))

        hist_tr.append(tr_loss)
        hist_va.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            print(f"  [{name}] ep {ep:2d}  train {tr_loss:.4f}  val {va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{name}] done in {time.time() - since:.1f}s "
          f"(best val={best_val:.4f})")
    return model, {"train": hist_tr, "val": hist_va}


def torch_predict(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).to(DEVICE)
        return model(xt).cpu().numpy()


# =========================================================================
#  METRICS
# =========================================================================
def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, h in enumerate(HORIZONS):
        yt, yp = y_true[:, j], y_pred[:, j]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae  = float(mean_absolute_error(yt, yp))
        r2   = float(r2_score(yt, yp))

        # early-warning F1: predict future breach (V-THD > 5 %) within horizon
        true_breach = (yt > IEEE519_LIMIT).astype(int)
        pred_breach = (yp > IEEE519_LIMIT).astype(int)
        ew_f1 = float(f1_score(true_breach, pred_breach, zero_division=0))
        breach_rate = float(true_breach.mean())

        rows.append({"horizon_min": h,
                     "RMSE": round(rmse, 4),
                     "MAE": round(mae, 4),
                     "R2": round(r2, 4),
                     "early_warning_F1": round(ew_f1, 4),
                     "breach_rate": round(breach_rate, 4)})
    return pd.DataFrame(rows)


# =========================================================================
#  DRIVER
# =========================================================================
def run() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df, _ = load_and_prepare()
    print(f"[data] building sliding windows (lookback={LOOKBACK}, horizons={HORIZONS})")
    X, y = make_windows(df)
    print(f"[data] windows: X {X.shape}  y {y.shape}")

    # Chronological split (NO random split - would leak future info into training)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = chronological_split(X, y)
    print(f"[data] chronological split: train={len(X_tr):,}  "
          f"val={len(X_va):,}  test={len(X_te):,}")

    # Normalize input features and (separately) target. Fit ONLY on train.
    n_features = X.shape[2]
    feat_scaler = StandardScaler().fit(X_tr.reshape(-1, n_features))
    def sx(A): return feat_scaler.transform(
        A.reshape(-1, n_features)).reshape(A.shape).astype(np.float32)
    X_tr_s, X_va_s, X_te_s = sx(X_tr), sx(X_va), sx(X_te)

    y_scaler = StandardScaler().fit(y_tr)
    y_tr_s = y_scaler.transform(y_tr).astype(np.float32)
    y_va_s = y_scaler.transform(y_va).astype(np.float32)

    def unsc(y_s): return y_scaler.inverse_transform(y_s)

    # ---------- 1. Persistence baseline (in ORIGINAL units, no training) ----------
    print("\n===== Persistence baseline =====")
    persist = PersistenceModel()
    yp = persist.predict(X_te)
    print(horizon_metrics(y_te, yp).to_string(index=False))
    all_metrics = {"Persistence": horizon_metrics(y_te, yp)}
    all_preds   = {"Persistence": yp}

    # ---------- 2. MLP ----------
    print("\n===== MLP =====")
    mlp = MLPForecaster(in_dim=LOOKBACK * n_features, out_dim=len(HORIZONS))
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_s),
                                         torch.from_numpy(y_tr_s)),
                           batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va_s),
                                         torch.from_numpy(y_va_s)),
                           batch_size=BATCH, shuffle=False)
    mlp, mlp_hist = train_pytorch(mlp, tr_loader, va_loader, "MLP")
    yp = unsc(torch_predict(mlp, X_te_s))
    print(horizon_metrics(y_te, yp).to_string(index=False))
    all_metrics["MLP"] = horizon_metrics(y_te, yp)
    all_preds["MLP"] = yp
    torch.save(mlp.state_dict(), MODEL_DIR / "MLP_state.pt")

    # ---------- 3. LSTM ----------
    print("\n===== LSTM =====")
    lstm = LSTMForecaster(n_features=n_features, out_dim=len(HORIZONS))
    lstm, lstm_hist = train_pytorch(lstm, tr_loader, va_loader, "LSTM")
    yp = unsc(torch_predict(lstm, X_te_s))
    print(horizon_metrics(y_te, yp).to_string(index=False))
    all_metrics["LSTM"] = horizon_metrics(y_te, yp)
    all_preds["LSTM"] = yp
    torch.save(lstm.state_dict(), MODEL_DIR / "LSTM_state.pt")

    # ---------- 4. GRU ----------
    print("\n===== GRU =====")
    gru = GRUForecaster(n_features=n_features, out_dim=len(HORIZONS))
    gru, gru_hist = train_pytorch(gru, tr_loader, va_loader, "GRU")
    yp = unsc(torch_predict(gru, X_te_s))
    print(horizon_metrics(y_te, yp).to_string(index=False))
    all_metrics["GRU"] = horizon_metrics(y_te, yp)
    all_preds["GRU"] = yp
    torch.save(gru.state_dict(), MODEL_DIR / "GRU_state.pt")

    # ---------- summary + save ----------
    merged = []
    for name, dfm in all_metrics.items():
        for _, r in dfm.iterrows():
            merged.append({"model": name, **r.to_dict()})
    summary = pd.DataFrame(merged)
    summary.to_csv(RESULTS_DIR / "forecasting_summary.csv", index=False)

    print("\n" + "=" * 76)
    print(" FORECASTING SUMMARY  (all models, all horizons)")
    print("=" * 76)
    print(summary.to_string(index=False))
    print("=" * 76)

    # ---------- plots ----------
    _plot_predictions(y_te, all_preds)
    _plot_metric_bars(summary, "RMSE")
    _plot_metric_bars(summary, "early_warning_F1")
    _plot_curves({"MLP": mlp_hist, "LSTM": lstm_hist, "GRU": gru_hist})

    print(f"\nSaved: {RESULTS_DIR}/forecasting_summary.csv")
    print(f"Saved: {FIG_DIR}/*.png")
    print(f"Saved: {MODEL_DIR}/*.pt")


def _plot_predictions(y_te: np.ndarray, all_preds: dict) -> None:
    """Show 30-min-ahead prediction over a slice of the test set."""
    slc = slice(0, 800)
    for h_idx, h in enumerate(HORIZONS):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(y_te[slc, h_idx], color="black", lw=1.2, label="True V-THD")
        colors = {"Persistence": "#e9c46a", "MLP": "#f4a261",
                  "LSTM": "#2a9d8f", "GRU": "#264653"}
        for name, yp in all_preds.items():
            ax.plot(yp[slc, h_idx], color=colors[name], lw=0.9,
                    alpha=0.85, label=name)
        ax.axhline(IEEE519_LIMIT, color="red", ls="--", lw=1,
                   label="IEEE 519 5% limit")
        ax.set_xlabel("Test sample index (minutes)")
        ax.set_ylabel("V-THD (%)")
        ax.set_title(f"V-THD forecast, horizon = +{h} min")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right", ncol=3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"01_predictions_h{h:02d}min.png", dpi=150)
        plt.close()


def _plot_metric_bars(summary: pd.DataFrame, metric: str) -> None:
    pivot = summary.pivot(index="horizon_min", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(9, 4.5), rot=0,
                    color=["#e9c46a", "#f4a261", "#2a9d8f", "#264653"])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("Forecast horizon (minutes ahead)")
    ax.set_title(f"{metric.replace('_', ' ')} per model and horizon")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"02_{metric}_bars.png", dpi=150)
    plt.close()


def _plot_curves(hists: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"MLP": "#f4a261", "LSTM": "#2a9d8f", "GRU": "#264653"}
    for name, h in hists.items():
        ax.plot(h["train"], color=colors[name], lw=1.3, label=f"{name} train")
        ax.plot(h["val"], color=colors[name], lw=1.3, ls="--",
                label=f"{name} val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled targets)")
    ax.set_title("Training curves — deep forecasters")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_training_curves.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run()
