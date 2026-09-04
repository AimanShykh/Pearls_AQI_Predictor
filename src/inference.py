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
    """
    Predict AQI for 24h, 48h and 72h ahead.

    Uses recent historical data to make sure lag and
    rolling features are available for the latest row.
    """

    # --------------------------------------------------------
    # 1. Load historical feature data
    # --------------------------------------------------------

    features_df = load_latest_features()

    if features_df.empty:
        raise RuntimeError(
            "No feature data available."
        )

    # Make sure data is sorted
    features_df = features_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # 2. Rebuild features from historical data
    # --------------------------------------------------------

    # This is important because lag/rolling features need
    # previous observations.

    required_history = max(
        max(config.LAG_HOURS),
        max(config.ROLLING_WINDOWS_HOURS)
    ) + 5

    recent_df = features_df.tail(
        max(required_history, 100)
    ).copy()

    engineered_df = data.build_features(
        recent_df
    )


    # --------------------------------------------------------
    # 3. Get latest row
    # --------------------------------------------------------

    latest_row = engineered_df.tail(1).copy()

    now = latest_row["timestamp"].iloc[0]

    current_aqi = float(
        latest_row["aqi"].iloc[0]
    )


    # --------------------------------------------------------
    # 4. Make predictions
    # --------------------------------------------------------

    forecasts = []

    for horizon in config.FORECAST_HORIZONS_HOURS:

        model = load_model(horizon)

        # Get the exact features expected by the model
        feature_cols = data.feature_columns_present(
            latest_row
        )

        # Check for missing features
        missing_features = [
            col
            for col in feature_cols
            if pd.isna(latest_row[col].iloc[0])
        ]

        if missing_features:

            raise RuntimeError(
                f"Missing feature values for {horizon}h model:\n"
                + "\n".join(missing_features)
            )


        # Prediction
        predicted_aqi = model.predict(
            latest_row[feature_cols]
        )[0]

        # AQI cannot be negative
        predicted_aqi = max(
            0.0,
            float(predicted_aqi)
        )


        # Category
        category, color = aqi_category(
            predicted_aqi
        )


        # Alert
        alert = (
            predicted_aqi >= config.ALERT_THRESHOLD
        )


        forecasts.append({

            "horizon_hours": horizon,

            "target_time":
                now + pd.Timedelta(
                    hours=horizon
                ),

            "predicted_aqi":
                round(predicted_aqi, 1),

            "category":
                category,

            "color":
                color,

            "alert":
                alert
        })


    # --------------------------------------------------------
    # 5. Return everything to Streamlit
    # --------------------------------------------------------

    return {

        "city":
            config.CITY_NAME,

        "as_of":
            now,

        "current_aqi":
            current_aqi,

        "current_category":
            aqi_category(current_aqi)[0],

        "forecasts":
            forecasts
    }


if __name__ == "__main__":
    result = predict_next_3_days()
    print(f"{result['city']} — as of {result['as_of']}")
    print(f"Current AQI: {result['current_aqi']} ({result['current_category']})")
    for f in result["forecasts"]:
        alert = " (ALERT)" if f["alert"] else ""
        print(f"  +{f['horizon_hours']}h: {f['predicted_aqi']} — {f['category']}{alert}")
