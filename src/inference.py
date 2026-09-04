"""
INFERENCE — turns "the latest saved features" + "the latest trained
models" into an actual 3-day forecast. Used by the dashboard.
"""
import joblib
import pandas as pd

import config
import data
import os

def aqi_category(aqi_value: float):
    for lo, hi, label, color in config.AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return label, color
    return "Hazardous+", "maroon"


def load_model(horizon: int):
    """
    Load the trained model.

    If Hopsworks is configured, get the model from the registry.
    Otherwise, load the locally saved model.
    """

    if config.HOPSWORKS_API_KEY:

        import hopsworks

        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT
        )

        registry = project.get_model_registry()

        model_meta = registry.get_best_model(
            f"{config.MODEL_NAME}_{horizon}h",
            "rmse",
            "min"
        )

        model_dir = model_meta.download()

        # IMPORTANT:
        # Load the model from the downloaded Hopsworks directory
        return joblib.load(
            os.path.join(model_dir, "model.joblib")
        )

    # Local model
    model_path = os.path.join(
        config.LOCAL_MODELS_DIR,
        f"aqi_{horizon}h",
        "model.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model for {horizon}h not found at:\n{model_path}\n\n"
            f"Run training_pipeline.py first."
        )

    return joblib.load(model_path)



def load_latest_features() -> pd.DataFrame:
    if config.HOPSWORKS_API_KEY:
        import hopsworks
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        df = fg.read()
    else:
        # df = pd.read_parquet("backfill_data.parquet")
        df = pd.read_parquet(config.LOCAL_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def predict_next_3_days() -> dict:
    features_df = load_latest_features()
    latest_row = features_df.tail(1)
    now = latest_row["timestamp"].iloc[0]
    current_aqi = float(latest_row["aqi"].iloc[0])

    forecasts = []
    for horizon in config.FORECAST_HORIZONS_HOURS:
        model = load_model(horizon)
        feature_cols = data.feature_columns_present(latest_row)
        predicted_aqi = max(0.0, float(model.predict(latest_row[feature_cols])[0]))
        category, color = aqi_category(predicted_aqi)

        forecasts.append({
            "horizon_hours": horizon,
            "target_time": now + pd.Timedelta(hours=horizon),
            "predicted_aqi": round(predicted_aqi, 1),
            "category": category,
            "color": color,
            "alert": predicted_aqi >= config.ALERT_THRESHOLD,
        })

    return {
        "city": config.CITY_NAME,
        "as_of": now,
        "current_aqi": current_aqi,
        "current_category": aqi_category(current_aqi)[0],
        "forecasts": forecasts,
    }


if __name__ == "__main__":
    result = predict_next_3_days()
    print(f"{result['city']} — as of {result['as_of']}")
    print(f"Current AQI: {result['current_aqi']} ({result['current_category']})")
    for f in result["forecasts"]:
        alert = " (ALERT)" if f["alert"] else ""
        print(f"  +{f['horizon_hours']}h: {f['predicted_aqi']} — {f['category']}{alert}")
