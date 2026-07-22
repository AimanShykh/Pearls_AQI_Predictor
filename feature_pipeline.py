"""
INFERENCE — used by the Streamlit dashboard (and can be run standalone) to
produce the next-3-day AQI forecast for Hyderabad, Sindh.

Strategy:
  - Pull the most recent ~96h of engineered features from Hopsworks
    (or local parquet fallback).
  - For each horizon (24h/48h/72h) load that horizon's registered model
    and predict off the *latest* feature row.
  - Blend in OpenWeather's forecast weather for the target timestamps as
    exogenous inputs where the model uses weather features.
"""
import joblib
import numpy as np
import pandas as pd

import config
from utils import log, fetch_openweather_forecast, aqi_category
from feature_engineering import FEATURE_COLUMNS

try:
    import hopsworks
except ImportError:
    hopsworks = None


def _load_model_local(horizon: int):
    return joblib.load(f"models/aqi_{horizon}h/model.joblib")


def _load_model_hopsworks(horizon: int):
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
    mr = project.get_model_registry()
    model_meta = mr.get_best_model(f"{config.MODEL_NAME}_{horizon}h", "rmse", "min")
    model_dir = model_meta.download()
    return joblib.load(f"{model_dir}/model.joblib")


def load_model(horizon: int):
    if hopsworks and config.HOPSWORKS_API_KEY:
        try:
            return _load_model_hopsworks(horizon)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Hopsworks model load failed for {horizon}h, trying local: {e}")
    return _load_model_local(horizon)


def get_latest_features() -> pd.DataFrame:
    if hopsworks and config.HOPSWORKS_API_KEY:
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        df = fg.read()
    else:
        df = pd.read_parquet("backfill_training_data.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def predict_next_3_days() -> dict:
    features_df = get_latest_features()
    latest_row = features_df.tail(1)
    latest_ts = latest_row["timestamp"].iloc[0]
    current_aqi = float(latest_row["aqi"].iloc[0])

    forecast_weather = None
    try:
        forecast_weather = fetch_openweather_forecast()
    except Exception as e:  # noqa: BLE001
        log.warning(f"Could not fetch forward weather forecast: {e}")

    predictions = []
    for horizon in config.FORECAST_HORIZONS_HOURS:
        cols = [c for c in FEATURE_COLUMNS if c in latest_row.columns]
        X = latest_row[cols].copy()

        # Swap in forecasted weather for the target time, if available —
        # improves accuracy vs. assuming weather stays constant.
        if forecast_weather is not None and not forecast_weather.empty:
            target_ts = latest_ts + pd.Timedelta(hours=horizon)
            forecast_weather["diff"] = (forecast_weather["timestamp"] - target_ts).abs()
            nearest = forecast_weather.sort_values("diff").iloc[0]
            for wcol in ["temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]:
                if wcol in X.columns:
                    X[wcol] = nearest[wcol]

        model = load_model(horizon)
        pred_aqi = float(model.predict(X)[0])
        pred_aqi = max(0.0, pred_aqi)
        category, color = aqi_category(pred_aqi)
        target_time = latest_ts + pd.Timedelta(hours=horizon)

        predictions.append({
            "horizon_hours": horizon,
            "target_time": target_time,
            "predicted_aqi": round(pred_aqi, 1),
            "category": category,
            "color": color,
            "alert": pred_aqi >= config.ALERT_THRESHOLD,
        })

    return {
        "city": config.CITY_NAME,
        "as_of": latest_ts,
        "current_aqi": current_aqi,
        "current_category": aqi_category(current_aqi)[0],
        "forecasts": predictions,
    }


if __name__ == "__main__":
    result = predict_next_3_days()
    print(f"\n{result['city']} — AQI forecast (as of {result['as_of']})")
    print(f"Current AQI: {result['current_aqi']} ({result['current_category']})\n")
    for f in result["forecasts"]:
        alert = " 🚨 ALERT" if f["alert"] else ""
        print(f"  +{f['horizon_hours']:>2}h ({f['target_time']}): {f['predicted_aqi']} — {f['category']}{alert}")
