
import sys
import os

# Allow Python to find files inside src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config
from inference import predict_next_3_days, load_latest_features


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=f"AQI Forecast - {config.CITY_NAME}",
    page_icon="🌫️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(f"🌫️ Air Quality Forecast")
st.subheader(config.CITY_NAME)

st.caption(
    "Machine Learning based AQI prediction using historical AQI, "
    "weather and pollutant features."
)

st.divider()


# ============================================================
# LOAD FORECAST
# ============================================================

@st.cache_data(ttl=3600)
def get_forecast():
    return predict_next_3_days()


try:
    result = get_forecast()

except Exception as e:
    st.error(
        "Unable to load the AQI forecast.\n\n"
        "Make sure that:\n"
        "1. Historical data has been created\n"
        "2. The models have been trained\n"
        "3. The required model files exist"
    )

    st.code(str(e))
    st.stop()


# ============================================================
# CURRENT AQI
# ============================================================

st.header("📍 Current Air Quality")

current_aqi = result["current_aqi"]
current_category = result["current_category"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Current AQI",
        value=f"{current_aqi:.0f}"
    )

with col2:
    st.metric(
        label="Air Quality",
        value=current_category
    )

with col3:
    st.metric(
        label="Last Updated",
        value=result["as_of"].strftime("%d %b %Y, %H:%M UTC")
    )


# ============================================================
# AQI FORECAST CARDS
# ============================================================

st.header("🔮 AQI Forecast")

forecast_cols = st.columns(3)

for i, forecast in enumerate(result["forecasts"]):

    with forecast_cols[i]:

        st.metric(
            label=f"+{forecast['horizon_hours']} Hours",
            value=f"{forecast['predicted_aqi']:.0f}",
            delta=forecast["category"]
        )

        st.caption(
            forecast["target_time"].strftime(
                "%d %b %Y, %H:%M UTC"
            )
        )


# ============================================================
# ALERT
# ============================================================

danger_found = any(
    forecast["alert"]
    for forecast in result["forecasts"]
)

if danger_found:

    st.error(
        "🚨 WARNING: High AQI is predicted during the next "
        "3 days. Consider reducing prolonged outdoor exposure."
    )

else:

    st.success(
        "✅ No high-AQI alert is predicted for the next 3 days."
    )


st.divider()


# ============================================================
# FORECAST CHART
# ============================================================

st.header("📈 3-Day AQI Forecast")

forecast_times = [result["as_of"]]
forecast_values = [result["current_aqi"]]

for forecast in result["forecasts"]:
    forecast_times.append(forecast["target_time"])
    forecast_values.append(forecast["predicted_aqi"])


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=forecast_times,
        y=forecast_values,
        mode="lines+markers+text",
        text=[
            f"{value:.0f}"
            for value in forecast_values
        ],
        textposition="top center",
        line=dict(width=3),
        marker=dict(size=9)
    )
)


# AQI threshold lines
fig.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="Good → Moderate"
)

fig.add_hline(
    y=100,
    line_dash="dash",
    annotation_text="Moderate → Unhealthy"
)

fig.add_hline(
    y=150,
    line_dash="dash",
    annotation_text="Unhealthy"
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="AQI",
    height=450,
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# RECENT HISTORY
# ============================================================

st.divider()

st.header("📊 Historical AQI")


try:

    history = load_latest_features()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    history = history.sort_values("timestamp")

    # Last 14 days
    history_14_days = history.tail(24 * 14)


    # --------------------------------------------------------
    # AQI HISTORY
    # --------------------------------------------------------

    fig_history = px.line(
        history_14_days,
        x="timestamp",
        y="aqi",
        title="AQI During Last 14 Days"
    )

    fig_history.update_layout(
        xaxis_title="Time",
        yaxis_title="AQI"
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )


    # --------------------------------------------------------
    # POLLUTANTS
    # --------------------------------------------------------

    st.subheader("🧪 Pollutant Levels")

    pollutant_columns = [
        "pm25",
        "pm10",
        "o3",
        "no2",
        "so2",
        "co"
    ]

    available_pollutants = [
        column
        for column in pollutant_columns
        if column in history.columns
    ]

    if available_pollutants:

        pollutant_data = history_14_days[
            ["timestamp"] + available_pollutants
        ]

        fig_pollutants = px.line(
            pollutant_data,
            x="timestamp",
            y=available_pollutants,
            title="Pollutant Trends"
        )

        st.plotly_chart(
            fig_pollutants,
            use_container_width=True
        )


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    st.subheader("🌤️ Weather Conditions")

    weather_columns = [
        "temp",
        "humidity",
        "pressure",
        "wind_speed",
        "clouds"
    ]

    available_weather = [
        column
        for column in weather_columns
        if column in history.columns
    ]

    if available_weather:

        weather_data = history_14_days[
            ["timestamp"] + available_weather
        ]

        selected_weather = st.selectbox(
            "Select weather variable",
            available_weather
        )

        fig_weather = px.line(
            weather_data,
            x="timestamp",
            y=selected_weather,
            title=f"{selected_weather} During Last 14 Days"
        )

        st.plotly_chart(
            fig_weather,
            use_container_width=True
        )


except Exception as e:

    st.warning(
        f"Historical data could not be displayed: {e}"
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander("🔍 View Latest Data"):

    try:

        latest_data = load_latest_features().tail(10)

        st.dataframe(
            latest_data,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Could not load latest data: {e}"
        )




# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AQI Forecasting System | Open-Meteo + Machine Learning + "
    "Hopsworks + Streamlit"
)
