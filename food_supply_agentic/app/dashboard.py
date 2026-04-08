"""
dashboard.py — Streamlit Dashboard
Run with: streamlit run app/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing      import load_and_process
from src.demand_forecasting   import forecast_demand
from src.inventory_simulation import simulate_inventory
from src.decision_engine      import run_decision_engine, decisions_to_dataframe
from src.optimization         import rebalance

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Supply Chain",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous Inventory Rebalancing — Agentic AI")
st.caption("Multi-agent observe → predict → decide → act pipeline")

# ── Run Pipeline (cached) ─────────────────────────────────────────────────────
@st.cache_data
def run_pipeline():
    df        = load_and_process(data_dir="data")
    df        = forecast_demand(df)
    df        = simulate_inventory(df)
    decisions = run_decision_engine(df)
    actions   = rebalance(decisions)
    dec_df    = decisions_to_dataframe(decisions)
    return df, dec_df, actions

with st.spinner("Running agentic pipeline..."):
    df, dec_df, actions = run_pipeline()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
stores      = sorted(df["Store"].unique())
sel_stores  = st.sidebar.multiselect("Stores", stores, default=stores)
if "Date" in df.columns:
    df["Date"]     = pd.to_datetime(df["Date"])
    dec_df["Date"] = pd.to_datetime(dec_df["Date"])
    min_d, max_d   = df["Date"].min(), df["Date"].max()
    date_range     = st.sidebar.date_input("Date Range", [min_d, max_d])

# Filtered views
df_f    = df[df["Store"].isin(sel_stores)]
dec_f   = dec_df[dec_df["Store"].isin(sel_stores)]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader("📊 System Overview")
c1, c2, c3, c4 = st.columns(4)

shortage_ct = len(dec_f[dec_f["Action"] == "SHORTAGE"])
surplus_ct  = len(dec_f[dec_f["Action"] == "SURPLUS"])
stable_ct   = len(dec_f[dec_f["Action"] == "STABLE"])
high_ct     = len(dec_f[dec_f["Urgency"] == "HIGH"])

c1.metric("🔴 Shortages",     shortage_ct)
c2.metric("🟢 Surpluses",     surplus_ct)
c3.metric("🔵 Stable",        stable_ct)
c4.metric("⚠️ High Urgency",  high_ct)

st.divider()

# ── Chart 1: Weekly Sales per Store ──────────────────────────────────────────
st.subheader("📈 Weekly Sales by Store")
fig1 = px.line(
    df_f, x="Date", y="Weekly_Sales", color="Store",
    labels={"Weekly_Sales": "Weekly Sales ($)", "Store": "Store"},
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Inventory vs Demand ──────────────────────────────────────────────
st.subheader("📦 Inventory vs Predicted Demand")
store_pick = st.selectbox("Select Store", sel_stores)
df_store   = df_f[df_f["Store"] == store_pick]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_store["Date"], y=df_store["predicted_demand"],
                           name="Predicted Demand", line=dict(color="royalblue")))
fig2.add_trace(go.Scatter(x=df_store["Date"], y=df_store["inventory"],
                           name="Inventory", line=dict(color="orange", dash="dash")))
fig2.update_layout(xaxis_title="Date", yaxis_title="Units")
st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: Decision Distribution ───────────────────────────────────────────
st.subheader("🤖 Agent Decision Distribution")
action_counts = dec_f["Action"].value_counts().reset_index()
action_counts.columns = ["Action", "Count"]
color_map = {"SHORTAGE": "#ef4444", "SURPLUS": "#22c55e", "STABLE": "#3b82f6"}
fig3 = px.bar(action_counts, x="Action", y="Count",
              color="Action", color_discrete_map=color_map)
st.plotly_chart(fig3, use_container_width=True)

# ── Transfer Actions Table ────────────────────────────────────────────────────
st.subheader("🚚 Rebalancing Transfer Actions")
if actions:
    actions_data = [{
        "Date":       a.date,
        "From Store": a.from_store,
        "To Store":   a.to_store,
        "Units":      f"{a.units:,.0f}",
        "Urgency":    a.urgency,
    } for a in actions[:50]]
    st.dataframe(pd.DataFrame(actions_data), use_container_width=True)
else:
    st.info("No transfers needed for selected filters.")

# ── Raw Decision Log ──────────────────────────────────────────────────────────
with st.expander("🗂 Full Decision Log"):
    st.dataframe(dec_f, use_container_width=True)
