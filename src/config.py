"""
CONFIG — every "knob" for this project lives here, in one place.
"""
import os

# ----------------------------------------------------------------------
# 1. WHERE — the city we're forecasting for
# ----------------------------------------------------------------------
CITY_NAME = "Hyderabad, Sindh, Pakistan"
LATITUDE = 25.3960
LONGITUDE = 68.3578

# ----------------------------------------------------------------------
# 2. DATA SOURCE — Open-Meteo. No API key needed.
# ----------------------------------------------------------------------
# AIR_QUALITY_URL = "https://archive-api.open-meteo.com/v1/archive"
# WEATHER_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
# ----------------------------------------------------------------------
# 2. DATA SOURCE — Open-Meteo. No API key needed.
# ----------------------------------------------------------------------
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# ----------------------------------------------------------------------
# 3. FEATURE STORE — Hopsworks (free tier)
# ----------------------------------------------------------------------
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "aqi_hyderabad_sindh")
##############################understand
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_forecast_model"

# ----------------------------------------------------------------------
# 4. WHAT WE'RE PREDICTING 
# ----------------------------------------------------------------------
FORECAST_HORIZONS_HOURS = [24, 48, 72]
LAG_HOURS = [1, 6, 24, 48] #past data of these hours
ROLLING_WINDOWS_HOURS = [6, 24] # average trnds of these hours

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
ALERT_THRESHOLD = 150

# ----------------------------------------------------------------------
# 6. LOCAL FILE PATHS — always the same spot, no matter which folder
#    you run a script from
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT, "backfill_data.parquet")
LOCAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "models") 

