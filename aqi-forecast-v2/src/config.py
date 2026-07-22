"""
CONFIG — every "knob" for this project lives here, in one place.
Change the city, and everything else (data fetching, features, models)
just works for the new location. Nothing else in the codebase should
contain a hard-coded number.
"""
import os

# ----------------------------------------------------------------------
# 1. WHERE — the city we're forecasting for
# ----------------------------------------------------------------------
CITY_NAME = "Hyderabad, Sindh, Pakistan"
LATITUDE = 25.3960
LONGITUDE = 68.3578

# ----------------------------------------------------------------------
# 2. DATA SOURCE — Open-Meteo (see README for why we picked this one)
#    No API key needed. These are just the two base URLs we call.
# ----------------------------------------------------------------------
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# ----------------------------------------------------------------------
# 3. FEATURE STORE — Hopsworks (free tier). This is the only account/
#    key you actually need to create for this project.
# ----------------------------------------------------------------------
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_hyderabad_sindh")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_forecast_model"

# ----------------------------------------------------------------------
# 4. WHAT WE'RE PREDICTING
# ----------------------------------------------------------------------
# We train 3 separate models: one that predicts 24 hours ahead, one for
# 48h, one for 72h. Predicting further ahead is a harder problem, so a
# specialist model per horizon tends to do better than one model trying
# to do all three at once.
FORECAST_HORIZONS_HOURS = [24, 48, 72]

# How many hours back the model is allowed to "look" when building
# lag/rolling features (see feature_engineering.py).
LAG_HOURS = [1, 6, 24, 48]
ROLLING_WINDOWS_HOURS = [6, 24]

# ----------------------------------------------------------------------
# 5. ALERTING — US EPA AQI categories (0-500 scale)
# ----------------------------------------------------------------------
AQI_CATEGORIES = [
    (0, 50, "Good", "green"),
    (51, 100, "Moderate", "yellow"),
    (101, 150, "Unhealthy for Sensitive Groups", "orange"),
    (151, 200, "Unhealthy", "red"),
    (201, 300, "Very Unhealthy", "purple"),
    (301, 500, "Hazardous", "maroon"),
]
ALERT_THRESHOLD = 150  # dashboard shows a red warning at/above this AQI
