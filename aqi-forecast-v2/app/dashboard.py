"""
DASHBOARD — the only file the end user actually looks at.
Run locally with:  streamlit run app/dashboard.py

Streamlit turns a plain Python script into a web page. Every st.xxx()
call below draws one piece of the page — no HTML/CSS needed.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import config
from inference import predict_next_3_days, load_latest_features

st.set_page_config(page_title=f"AQI Forecast — {config.CITY_NAME}", page_icon="🌫️", layout="wide")
st.title(f"🌫️ AQI Forecast — {config.CITY_NAME}")
st.caption("Data: Open-Meteo (free, no API key). Feature store & models: Hopsworks. "
           "Automation: GitHub Actions (hourly + daily).")

# ---- Load the forecast (cached for 1 hour so we're not re-predicting on every click) ----
@st.cache_data(ttl=3600)
def get_forecast():
    return predict_next_3_days()

try:
    result = get_forecast()
except Exception as e:
    st.error(f"Couldn't load a forecast yet. Has the feature pipeline run at least once? ({e})")
    st.stop()

# ---- Top row: current AQI + next 3 days ----
cols = st.columns(4)
cols[0].metric("Current AQI", f"{result['current_aqi']:.0f}", result["current_category"])
for i, f in enumerate(result["forecasts"]):
    cols[i + 1].metric(
        f"+{f['horizon_hours']}h ({f['target_time'].strftime('%a, %H:%M UTC')})",
        f"{f['predicted_aqi']:.0f}",
        f["category"],
    )

if any(f["alert"] for f in result["forecasts"]):
    st.error("🚨 Hazardous AQI levels predicted in the next 3 days. Limit outdoor exposure, especially for sensitive groups.")

st.divider()

# ---- Forecast line chart ----
st.subheader("Next 3 days")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[result["as_of"]] + [f["target_time"] for f in result["forecasts"]],
    y=[result["current_aqi"]] + [f["predicted_aqi"] for f in result["forecasts"]],
    mode="lines+markers+text",
    text=[f"{result['current_aqi']:.0f}"] + [f"{f['predicted_aqi']:.0f}" for f in result["forecasts"]],
    textposition="top center",
))
for lo, hi, label, color in config.AQI_CATEGORIES:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.08, line_width=0)
fig.update_layout(yaxis_title="AQI", xaxis_title="Time (UTC)", height=400)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Simple exploratory charts over recent history ----
st.subheader("Recent history")
try:
    history = load_latest_features().tail(24 * 14)  # last 14 days
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(history, x="timestamp", y="aqi", title="AQI over the last 14 days"),
                         use_container_width=True)
    with c2:
        st.plotly_chart(px.box(history, x="hour", y="aqi", title="AQI by hour of day"),
                         use_container_width=True)
except Exception as e:
    st.warning(f"History chart unavailable: {e}")
