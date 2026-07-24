"""
Visualization Dashboard (CSE) — Proposal Module 7
====================================================

The one piece of the CSE-side task list that had no interface yet:
"Real-time simulation results, Alerts and predictions, Cost impact
visualization." This wires together the models already built in
anomaly_detection.py, disturbance_classifier.py, and explainability.py
into a single interactive view over the Part 6 disturbance dataset.

Run:
    streamlit run src/dashboard.py

>>>>>  WARNING - SYNTHETIC-STYLE playback, not a live feed. There is no
       real-time sensor connection here; the "current reading" is
       whichever row of the (field-measured, per team confirmation —
       see docs/DATA_PROVENANCE_AND_QUALITY.md) disturbance dataset the
       slider is on. Treat this as a pipeline-validation UI, not a
       deployed monitoring system.

>>>>>  WARNING - economic_cost_BDT and battery_capacity_loss_pct are
       team-calculated/estimated columns, not independent measurements
       (see docs/DATA_PROVENANCE_AND_QUALITY.md). The cost panel below
       labels them as such rather than presenting them as ground truth.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "external" / "microgrid_power_quality_dataset.csv"
MODEL_DIR = ROOT / "models" / "disturbance"

FEATURE_COLS = [
    "voltage_rms_V", "current_rms_A", "frequency_Hz",
    "temperature_C", "irradiance_Wm2",
    "harmonic_3rd_pct", "harmonic_5th_pct", "harmonic_7th_pct",
    "THD_voltage_pct", "THD_current_pct",
]
NOMINAL = {"voltage_rms_V": 230.0, "current_rms_A": 15.0, "frequency_Hz": 50.0}
SEED = 42

st.set_page_config(page_title="Microgrid PQ Digital Twin", layout="wide")


# --------------------------------------------------------------------- #
# Cached loaders — model / data only load & fit once per session
# --------------------------------------------------------------------- #
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["disturbance_type"] = df["disturbance_type"].fillna("None")
    return df


@st.cache_resource
def load_disturbance_model(_df: pd.DataFrame) -> dict:
    """Load a saved model if one exists locally (fast path for local dev,
    where you've already run disturbance_classifier.py). Otherwise train
    one directly from the data — this is the path that runs on a fresh
    deploy (e.g. Streamlit Cloud), since models/*.pkl is gitignored on
    purpose and a fresh git clone never has it. Same recipe as
    disturbance_classifier.py's RF, just trained in-memory instead of
    loaded from disk."""
    pkl_files = sorted(MODEL_DIR.glob("*_seed0.pkl"))
    if pkl_files:
        with open(pkl_files[0], "rb") as f:
            return pickle.load(f)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(_df["disturbance_type"])
    model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(_df[FEATURE_COLS], y)
    return {"model": model, "label_encoder": le, "features": FEATURE_COLS}


@st.cache_resource
def fit_anomaly_detector(_df: pd.DataFrame) -> tuple[IsolationForest, StandardScaler]:
    """Same recipe as anomaly_detection.py's Dataset A run, contamination
    matched to the dataset's true sensor_fault_flag rate."""
    X = _df[FEATURE_COLS].to_numpy(dtype=float)
    true_rate = _df["sensor_fault_flag"].mean()
    scaler = StandardScaler().fit(X)
    clf = IsolationForest(
        n_estimators=300, contamination=max(true_rate, 1e-4),
        random_state=SEED, n_jobs=-1,
    ).fit(scaler.transform(X))
    return clf, scaler


@st.cache_resource
def build_shap_explainer(_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


# --------------------------------------------------------------------- #
# Load everything once
# --------------------------------------------------------------------- #
df = load_data()
bundle = load_disturbance_model(df)
disturbance_model = bundle["model"]
label_encoder = bundle["label_encoder"]
iso_forest, scaler = fit_anomaly_detector(df)
explainer = build_shap_explainer(disturbance_model)

st.title("Microgrid Power-Quality Digital Twin — Live View")
st.caption(
    "Part 6 dashboard · disturbance dataset (field-measured, per team "
    "confirmation) · playback, not a live sensor feed — see "
    "`docs/DATA_PROVENANCE_AND_QUALITY.md` for what's measured vs. "
    "team-calculated in this data."
)

# --------------------------------------------------------------------- #
# Top-level dataset overview
# --------------------------------------------------------------------- #
overview_cols = st.columns(4)
overview_cols[0].metric("Rows in dataset", f"{len(df):,}")
overview_cols[1].metric("Sensor fault rate", f"{df['sensor_fault_flag'].mean()*100:.2f}%")
overview_cols[2].metric(
    "Disturbance rate", f"{(df['disturbance_type'] != 'None').mean()*100:.1f}%")
overview_cols[3].metric(
    "Disturbance classes", f"{df['disturbance_type'].nunique()}")

st.divider()

# --------------------------------------------------------------------- #
# Row selector — the "current reading" the rest of the page reacts to
# --------------------------------------------------------------------- #
st.subheader("1. Pick a reading")
row_idx = st.slider("Row index (simulated current timestep)",
                     0, len(df) - 1, value=0, step=1)
row = df.iloc[row_idx]
X_row = pd.DataFrame([row[FEATURE_COLS]])

left, right = st.columns([3, 2])

with left:
    st.subheader("2. Current reading")
    m = st.columns(5)
    for i, col in enumerate(["voltage_rms_V", "current_rms_A", "frequency_Hz",
                              "THD_voltage_pct", "THD_current_pct"]):
        delta = None
        if col in NOMINAL:
            delta = f"{row[col] - NOMINAL[col]:+.2f} vs nominal"
        m[i].metric(col, f"{row[col]:.2f}", delta)

    weather = st.columns(2)
    weather[0].metric("Temperature (°C)", f"{row['temperature_C']:.1f}")
    weather[1].metric("Irradiance (W/m²)", f"{row['irradiance_Wm2']:.1f}")

with right:
    st.subheader("3. Alerts")

    pred_idx = disturbance_model.predict(X_row)[0]
    pred_proba = disturbance_model.predict_proba(X_row)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    pred_confidence = pred_proba[pred_idx]

    X_row_scaled = scaler.transform(X_row.to_numpy())
    anomaly_score = -iso_forest.decision_function(X_row_scaled)[0]
    is_anomaly = iso_forest.predict(X_row_scaled)[0] == -1

    if pred_label != "None":
        st.error(f"⚠️ Disturbance predicted: **{pred_label}** "
                 f"({pred_confidence*100:.1f}% confidence)")
    else:
        st.success(f"✅ No disturbance predicted "
                   f"({pred_confidence*100:.1f}% confidence)")

    if is_anomaly:
        st.warning(f"🚨 Flagged as anomalous reading "
                    f"(score {anomaly_score:.3f}) — possible sensor "
                    f"fault or bad data")
    else:
        st.info(f"Reading within normal range (anomaly score {anomaly_score:.3f})")

    if bool(row["sensor_fault_flag"]):
        st.caption("Ground truth for this row: sensor_fault_flag = 1 (known fault)")

st.divider()

# --------------------------------------------------------------------- #
# Explainability — why did the model predict what it predicted?
# --------------------------------------------------------------------- #
st.subheader("4. Why this prediction — SHAP attribution")

shap_values = explainer.shap_values(X_row)
if isinstance(shap_values, list):
    # older SHAP API: list of (n_samples, n_features) arrays, one per class
    shap_row = shap_values[pred_idx][0]
else:
    # newer SHAP API: single (n_samples, n_features, n_classes) array
    shap_row = shap_values[0, :, pred_idx]

shap_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "shap_value": shap_row,
}).sort_values("shap_value", key=abs, ascending=True)

fig_shap = px.bar(
    shap_df, x="shap_value", y="feature", orientation="h",
    color="shap_value", color_continuous_scale="RdBu_r",
    title=f"Feature contribution to predicting '{pred_label}' on this row",
)
fig_shap.update_layout(coloraxis_showscale=False, height=350)
st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# --------------------------------------------------------------------- #
# Cost impact — explicitly labelled as team-calculated, not measured
# --------------------------------------------------------------------- #
st.subheader("5. Cost impact (team-calculated — not an independent measurement)")
st.caption(
    "`economic_cost_BDT` and `battery_capacity_loss_pct` are derived "
    "columns, not sensor readings — see the provenance doc before "
    "treating these as ground truth."
)

cost_cols = st.columns(3)
cost_cols[0].metric("This row's cost (BDT)", f"{row['economic_cost_BDT']:.2f}")
cost_cols[1].metric("Battery degradation rate", f"{row['battery_degradation_rate']:.5f}")
cost_cols[2].metric(
    "Cumulative battery capacity loss",
    f"{row['battery_capacity_loss_pct']*100:.3f}%")

cum_df = df.iloc[:row_idx + 1]
fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(
    x=cum_df.index, y=cum_df["economic_cost_BDT"].cumsum(),
    mode="lines", name="Cumulative cost (BDT)", line=dict(color="#e76f51")))
fig_cost.update_layout(
    title="Cumulative economic cost up to the selected row",
    xaxis_title="Row index", yaxis_title="Cumulative BDT", height=300)
st.plotly_chart(fig_cost, use_container_width=True)

st.divider()

# --------------------------------------------------------------------- #
# Historical trend with disturbance regions highlighted
# --------------------------------------------------------------------- #
st.subheader("6. Historical trend")
trend_metric = st.selectbox(
    "Metric", ["THD_voltage_pct", "voltage_rms_V", "current_rms_A"], index=0)

fig_trend = px.line(df, y=trend_metric, title=f"{trend_metric} across all rows")
fig_trend.add_vline(x=row_idx, line_dash="dash", line_color="red",
                     annotation_text="selected row")
for dtype, color in zip(
    ["Voltage_Sag", "Harmonic_Distortion", "Combined_Weather_Electrical"],
    ["orange", "purple", "brown"],
):
    idx = df.index[df["disturbance_type"] == dtype]
    if len(idx):
        fig_trend.add_trace(go.Scatter(
            x=idx, y=df.loc[idx, trend_metric], mode="markers",
            marker=dict(size=4, color=color), name=dtype))
fig_trend.update_layout(height=350)
st.plotly_chart(fig_trend, use_container_width=True)
