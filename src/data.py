"""
DATA.PY — everything about turning raw numbers from Open-Meteo into a
clean table the model can learn from.

This file has two halves:
  PART A — FETCHING: call Open-Meteo, get raw AQI + weather numbers
  PART B — FEATURE ENGINEERING: turn raw numbers into "clues" (features)
           that help a model predict the future

Why one file instead of many? Because these two steps are tightly
linked — if you change what you fetch, you usually need to change what
you engineer. Keeping them together makes that obvious.
"""
import requests
import numpy as np
import pandas as pd
import config
# import os
# import json
# from datetime import datetime, timedelta

# ========================================================================
# PART A — FETCHING RAW DATA FROM OPEN-METEO
# ========================================================================
#
# Open-Meteo needs NO API key. You just build a URL with your
# coordinates and the fields you want, and it hands back JSON.
#
# We need two things from it:
#   1. Air quality  -> gives us the actual AQI number (us_aqi) + pollutants
#   2. Weather      -> temperature, humidity, wind etc. (AQI depends on
#                       weather: no wind = pollution doesn't disperse)

def fetch_current_conditions() -> dict:
    """
    Get RIGHT NOW's AQI + weather for our city, as one flat dictionary.
    This is what the hourly feature pipeline calls every hour.
    """
    air = _fetch_air_quality(past_days=0, forecast_days=1)
    weather = _fetch_weather(past_days=0, forecast_days=1)

    # Both responses come back as hour-by-hour arrays. We just want the
    # single row that matches "now" (the closest past hour).
    now = pd.Timestamp.utcnow().floor("h")
    air_row = air[air["timestamp"] == now]
    weather_row = weather[weather["timestamp"] == now]

    if air_row.empty or weather_row.empty:
        # fall back to the latest available hour if "now" isn't in yet
        air_row = air.tail(1)
        weather_row = weather[weather["timestamp"] == air_row["timestamp"].iloc[0]]

    row = {**air_row.iloc[0].to_dict(), **weather_row.iloc[0].to_dict()}
    return row


def fetch_forecast_weather() -> pd.DataFrame:
    """
    Get the next few days of FORECASTED weather (not air quality).
    Used at prediction time: if we're forecasting AQI 3 days from now,
    it helps to know the *expected* wind/humidity then, not just today's.
    """
    return _fetch_weather(past_days=0, forecast_days=7)


# def fetch_history(days: int) -> pd.DataFrame:
#     """
#     Get the last `days` days of AQI + weather, merged into one table.
#     Used once, at the start, to build a training dataset (see backfill.py).
#     Open-Meteo allows up to 92 days of "past_days" in a single call.
#     """
#     air = _fetch_air_quality(past_days=min(days, 92), forecast_days=1)
#     weather = _fetch_weather(past_days=min(days, 92), forecast_days=1)
#     merged = pd.merge(air, weather, on="timestamp", how="inner")
#     return merged.sort_values("timestamp").reset_index(drop=True)
def fetch_history(days: int) -> pd.DataFrame:
    """
    Get the last `days` days of AQI + weather, merged into one table.
    Open-Meteo allows multi-year ranges via explicit start_date/end_date
    (unlike the simpler past_days shortcut, which caps at 92 days).
    """
    end_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    air = _fetch_air_quality(start_date, end_date)
    weather = _fetch_weather_archive(start_date, end_date)
    merged = pd.merge(air, weather, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)

def _fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical weather, further back than the forecast endpoint allows.
    Uses Open-Meteo's Archive API (ERA5 reanalysis), which supports
    dates going back to 1940."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    r = requests.get(config.WEATHER_ARCHIVE_URL, params=params, timeout=60)
    payload = r.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Weather archive request failed: {payload}")
    data = payload["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"], utc=True),
        "temp": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "pressure": data["surface_pressure"],
        "wind_speed": data["wind_speed_10m"],
        "clouds": data["cloud_cover"],
    })

# def _fetch_air_quality(past_days: int, forecast_days: int) -> pd.DataFrame:
#     params = {
#         "latitude": config.LATITUDE,
#         "longitude": config.LONGITUDE,
#         "hourly": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
#         "past_days": past_days,
#         "forecast_days": forecast_days,
#         "timezone": "UTC",
#     }
#     data = requests.get(config.AIR_QUALITY_URL, params=params, timeout=30).json()["hourly"]
#     return pd.DataFrame({
#         "timestamp": pd.to_datetime(data["time"], utc=True),
#         "aqi": data["us_aqi"],
#         "pm25": data["pm2_5"],
#         "pm10": data["pm10"],
#         "o3": data["ozone"],
#         "no2": data["nitrogen_dioxide"],
#         "so2": data["sulphur_dioxide"],
#         "co": data["carbon_monoxide"],
#     })


# def _fetch_weather(past_days: int, forecast_days: int) -> pd.DataFrame:
#     params = {
#         "latitude": config.LATITUDE,
#         "longitude": config.LONGITUDE,
#         "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover",
#         "past_days": past_days,
#         "forecast_days": forecast_days,
#         "timezone": "UTC",
#     }
#     data = requests.get(config.WEATHER_URL, params=params, timeout=30).json()["hourly"]
#     return pd.DataFrame({
#         "timestamp": pd.to_datetime(data["time"], utc=True),
#         "temp": data["temperature_2m"],
#         "humidity": data["relative_humidity_2m"],
#         "pressure": data["surface_pressure"],
#         "wind_speed": data["wind_speed_10m"],
#         "clouds": data["cloud_cover"],
#     })
def _fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    data = requests.get(config.AIR_QUALITY_URL, params=params, timeout=60).json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"], utc=True),
        "aqi": data["us_aqi"],
        "pm25": data["pm2_5"],
        "pm10": data["pm10"],
        "o3": data["ozone"],
        "no2": data["nitrogen_dioxide"],
        "so2": data["sulphur_dioxide"],
        "co": data["carbon_monoxide"],
    })


def _fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    data = requests.get(config.WEATHER_URL, params=params, timeout=60).json()["hourly"]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"], utc=True),
        "temp": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "pressure": data["surface_pressure"],
        "wind_speed": data["wind_speed_10m"],
        "clouds": data["cloud_cover"],
    })


# ========================================================================
# PART B — FEATURE ENGINEERING
# ========================================================================
#
# A model can't just see "AQI = 90 right now" and guess the future. We
# need to hand it useful clues. Every function below adds one kind of
# clue as a new column.

# This list is the exact set of columns every model will be trained and
# predict on. Keeping it in one place means training and prediction can
# never accidentally use different features (a common, sneaky ML bug).
RAW_COLUMNS = ["pm25", "pm10", "o3", "no2", "so2", "co",
               "temp", "humidity", "pressure", "wind_speed", "clouds"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clue: 'what time of day/week/year is it?' — pollution has daily
    rhythms (rush hour) and seasonal ones (burning season, monsoon)."""
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month

    # A plain "hour" number tricks a model into thinking hour 23 and
    # hour 0 are far apart, when really they're right next to each
    # other. Sin/cos encoding fixes that by placing hours on a circle.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clue: 'what was the AQI N hours ago?' — this is what gives the
    model memory. Pollution today is strongly related to pollution
    yesterday, so this is usually the single most useful feature."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for hours_back in config.LAG_HOURS:
        df[f"aqi_lag_{hours_back}h"] = df["aqi"].shift(hours_back)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clue: 'what's the AQI trend been recently?' — a rolling average
    smooths out noisy hour-to-hour spikes and shows the bigger trend."""
    df = df.copy()
    for window in config.ROLLING_WINDOWS_HOURS:
        df[f"aqi_avg_{window}h"] = df["aqi"].rolling(window, min_periods=1).mean()
    # Clue: 'is AQI rising or falling right now, and how fast?'
    df["aqi_change_rate"] = df["aqi"].diff(1)
    return df


def add_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the answer key for training: 'what WAS the AQI N hours after
    this row?'. shift(-N) looks forward N rows instead of backward.
    Only historical data has these answers (we don't know the future
    yet!) — that's exactly why we need this for training only.
    """
    df = df.copy()
    for horizon in config.FORECAST_HORIZONS_HOURS:
        df[f"target_{horizon}h"] = df["aqi"].shift(-horizon)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature steps in order. Used both for training data and
    for a live prediction row — same function, so features are always
    computed identically in both places."""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def feature_columns_present(df: pd.DataFrame) -> list:
    """The full list of model input columns, filtered to ones that
    actually exist in `df` (handy since a single live row won't have
    all lag columns filled in yet)."""
    engineered = (
        ["hour", "day_of_week", "month", "hour_sin", "hour_cos"]
        + [f"aqi_lag_{h}h" for h in config.LAG_HOURS]
        + [f"aqi_avg_{w}h" for w in config.ROLLING_WINDOWS_HOURS]
        + ["aqi_change_rate"]
    )
    all_cols = RAW_COLUMNS + engineered
    return [c for c in all_cols if c in df.columns]

