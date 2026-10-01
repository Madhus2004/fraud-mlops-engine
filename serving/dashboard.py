import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from core.db import fetch_logs, init_db
from core.features import FEATURE_COLUMNS


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# First row of the public credit-card dataset, used as an editable starting point.
DEFAULTS = {
    "Time": 100.0, "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781, "V5": -0.3383, "V6": 0.4623,
    "V7": 0.2395, "V8": 0.0986, "V9": 0.3637, "V10": 0.0907, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704, "V17": 0.2079,
    "V18": 0.0257, "V19": 0.4039, "V20": 0.2514, "V21": -0.0183, "V22": 0.2778,
    "V23": -0.1104, "V24": 0.0669, "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
}

st.set_page_config(page_title="Fraud Engine", page_icon="🛡️", layout="wide")
init_db()

st.title("🛡️ Real-Time Fraud Detection Engine")
st.caption("Serving plane only. Drift detection, retraining and model promotion run offline on a local worker.")

# Active model card
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    m = health.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active model", health["active_model_version"])
    c2.metric("Decision threshold", f"{health['decision_threshold']:.4f}")
    c3.metric("Holdout PR-AUC", m.get("holdout_pr_auc", "n/a"))
    c4.metric("Holdout recall", m.get("holdout_recall", "n/a"))
except Exception as exc:
    st.warning(f"API not reachable yet: {exc}")

tab_test, tab_logs = st.tabs(["🧪 Test a prediction", "📋 Inference logs"])

# ---- Tab 1 (runs first so a new prediction shows up in Tab 2 on the same run) ----
with tab_test:
    with st.form("predict_form"):
        a, b = st.columns(2)
        amount = a.number_input("Amount ($)", value=DEFAULTS["Amount"], min_value=0.0)
        time_s = b.number_input("Time (seconds)", value=DEFAULTS["Time"], min_value=0.0)
        with st.expander("PCA features V1–V28"):
            cols = st.columns(4)
            vs = {f"V{i}": cols[(i - 1) % 4].number_input(f"V{i}", value=DEFAULTS[f"V{i}"], format="%.4f")
                  for i in range(1, 29)}
        go = st.form_submit_button("Evaluate risk")
    if go:
        payload = {"Time": time_s, "Amount": amount, **vs}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            if r.ok:
                st.session_state["last"] = r.json()
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as exc:
            st.error(f"Could not reach API: {exc}")
    if "last" in st.session_state:
        res = st.session_state["last"]
        x, y, z, w = st.columns(4)
        x.metric("Transaction", res["txn_id"][:8] + "…")
        y.metric("Risk score", f"{res['risk_score']:.4f}")
        z.metric("Decision", "🚨 FLAGGED" if res["is_flagged"] else "✅ APPROVED")
        w.metric("Model", res["model_version"])
        st.json(res)

# ---- Tab 2 ----
with tab_logs:
    top = st.columns([1, 1, 2, 1, 1])
    status = top[0].selectbox("Decision", ["All", "Flagged", "Approved"])
    label = top[1].selectbox("Ground truth", ["All", "Labeled", "Unlabeled"])
    limit = top[3].slider("Max rows", 100, 5000, 1000, step=100)
    top[4].write("")
    top[4].button("🔄 Refresh")

    df = fetch_logs(limit=limit)
    if df.empty:
        st.info("No logs yet. Send a prediction from the first tab or run `python -m ops.simulator`.")
    else:
        versions = sorted(df["model_version"].dropna().unique())
        chosen = top[2].multiselect("Model version", versions, default=versions)
        df = df[df["model_version"].isin(chosen) | df["model_version"].isna()]
        if status != "All":
            df = df[df["is_flagged"] == (1 if status == "Flagged" else 0)]
        if label != "All":
            df = df[df["actual_label"].notna() == (label == "Labeled")]

        lab = df.dropna(subset=["actual_label"])
        tp = int(((lab["is_flagged"] == 1) & (lab["actual_label"] == 1)).sum())
        fp = int(((lab["is_flagged"] == 1) & (lab["actual_label"] == 0)).sum())
        fn = int(((lab["is_flagged"] == 0) & (lab["actual_label"] == 1)).sum())
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Logs", f"{len(df):,}")
        m2.metric("Flag rate", f"{df['is_flagged'].mean() * 100:.1f}%" if len(df) else "n/a")
        m3.metric("Labeled", f"{len(lab):,}")
        m4.metric("Live precision", f"{tp / (tp + fp):.2%}" if tp + fp else "n/a")
        m5.metric("Live recall", f"{tp / (tp + fn):.2%}" if tp + fn else "n/a")

        if len(df):
            fig = px.histogram(df, x="risk_score", color=df["is_flagged"].map({1: "Flagged", 0: "Approved"}),
                               nbins=40, barmode="overlay", opacity=0.75, title="Risk score distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.drop(columns=["features_json"]), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.drop(columns=["features_json"]).to_csv(index=False),
                           "inference_logs.csv", "text/csv")

        if len(df):
            pick = st.selectbox("Inspect a transaction's raw features", df["txn_id"].tolist())
            row = df[df["txn_id"] == pick].iloc[0]
            st.json(json.loads(row["features_json"]))