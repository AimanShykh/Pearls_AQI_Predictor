"""
Central configuration for the AQI Forecasting pipeline.
City: Hyderabad, Sindh, Pakistan
"""
import os

# ----------------------------------------------------------------------
# Location
# ----------------------------------------------------------------------
CITY_NAME = "Hyderabad, Sindh, Pakistan"
LATITUDE = 25.3960
LONGITUDE = 68.3578

# AQICN (waqi.info) station search string. AQICN resolves nearest station
# via geo-lookup, which is more reliable than guessing a station slug.
AQICN_GEO_URL = f"https://api.waqi.info/feed/geo:{LATITUDE};{LONGITUDE}/"

# OpenWeather Air Pollution + Onecall (weather) endpoints
OWM_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OWM_AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# ----------------------------------------------------------------------
# API keys (never hard-code — always via environment / GitHub Secrets)
# ----------------------------------------------------------------------
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "")
OWM_API_KEY = os.environ.get("OWM_API_KEY", "")

# ----------------------------------------------------------------------
# Hopsworks Feature Store / Model Registry
# ----------------------------------------------------------------------
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_hyderabad_sindh")

FEATURE_GROUP_NAME = "aqi_features_hyderabad"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "aqi_forecast_model"
MODEL_VERSION = None  # None => Hopsworks auto-increments; inference always pulls latest

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
TARGET_COL = "aqi"
FORECAST_HORIZONS_HOURS = [24, 48, 72]  # next 1, 2, 3 days

LAG_HOURS = [1, 3, 6, 12, 24, 48]
ROLLING_WINDOWS_HOURS = [3, 6, 24]

RAW_POLLUTANT_COLS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
RAW_WEATHER_COLS = ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]

# ----------------------------------------------------------------------
# AQI thresholds (US EPA breakpoints) used for alerting
# ----------------------------------------------------------------------
AQI_CATEGORIES = [
    (0, 50, "Good", "green"),
    (51, 100, "Moderate", "yellow"),
    (101, 150, "Unhealthy for Sensitive Groups", "orange"),
    (151, 200, "Unhealthy", "red"),
    (201, 300, "Very Unhealthy", "purple"),
    (301, 500, "Hazardous", "maroon"),
]
ALERT_THRESHOLD = 150  # trigger alert at/above "Unhealthy for Sensitive Groups"
