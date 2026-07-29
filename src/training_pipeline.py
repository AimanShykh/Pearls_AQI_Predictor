"""
TRAINING PIPELINE — runs once a day (see .github/workflows/training_pipeline.yml)

What it does, in plain English:
  1. Load all the saved feature rows from Hopsworks.
  2. For each forecast horizon (24h, 48h, 72h), try two models:
       - Ridge Regression: a simple straight-line model. Fast, easy to
         reason about, a good baseline.
       - Random Forest: hundreds of small decision trees voting
         together. Usually more accurate, still fairly easy to explain.
  3. Score both on data the model never saw during training.
  4. Keep whichever did better, and save it.

We deliberately train separate models per horizon rather than one
model for all three — 3-days-ahead is a harder problem than
1-day-ahead, so specialists tend to do better than a generalist.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
import data


def load_training_data() -> pd.DataFrame:
    """Prefer Hopsworks (real deployment); fall back to the local
    parquet file from backfill.py (useful for local testing)."""
    if config.HOPSWORKS_API_KEY:
        import hopsworks
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        return fg.read()
    # print("HOPSWORKS_API_KEY not set — reading local backfill_data.parquet instead.")
    # return pd.read_parquet("backfill_data.parquet")
    print(f"HOPSWORKS_API_KEY not set — reading local file instead: {config.LOCAL_DATA_PATH}")
    return pd.read_parquet(config.LOCAL_DATA_PATH)


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    IMPORTANT: for time series data, never shuffle before splitting.
    We train on the OLDER 80% and test on the NEWER 20% — this mimics
    the real situation of predicting the future from the past.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_at = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_at], df.iloc[split_at:]


def train_one_horizon(df: pd.DataFrame, horizon: int):
    """Train + evaluate both candidate models for one forecast horizon,
    return whichever one scored best."""
    target_col = f"target_{horizon}h"
    feature_cols = data.feature_columns_present(df)
    clean_df = df.dropna(subset=feature_cols + [target_col])

    if len(clean_df) < 50:
        print(f"Not enough data ({len(clean_df)} rows) to train the {horizon}h model yet — skipping.")
        return None

    train_df, test_df = time_based_split(clean_df)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    }

    best_name, best_model, best_rmse, all_metrics = None, None, float("inf"), {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        all_metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"  [{horizon}h] {name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

        if rmse < best_rmse:
            best_name, best_model, best_rmse = name, model, rmse

    print(f"  -> best model for {horizon}h horizon: {best_name}")
    return {
        "horizon": horizon,
        "model": best_model,
        "model_name": best_name,
        "metrics": all_metrics[best_name],
        "feature_columns": feature_cols,
    }


def save_model(result: dict):
    """Save the trained model to disk, and to the Hopsworks Model
    Registry if configured (so the dashboard can always fetch the
    latest version)."""
    # model_dir = f"models/aqi_{result['horizon']}h"
    model_dir = os.path.join(config.LOCAL_MODELS_DIR, f"aqi_{result['horizon']}h")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(result["model"], f"{model_dir}/model.joblib")
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump({
            "model_name": result["model_name"],
            "metrics": result["metrics"],
            "feature_columns": result["feature_columns"],
        }, f, indent=2)

    if config.HOPSWORKS_API_KEY:
        import hopsworks
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        registry = project.get_model_registry()
        hw_model = registry.python.create_model(
            name=f"{config.MODEL_NAME}_{result['horizon']}h",
            metrics=result["metrics"],
            description=f"AQI, {result['horizon']}h ahead, {config.CITY_NAME}. Algorithm: {result['model_name']}",
        )
        hw_model.save(model_dir)
        print(f"  Registered '{config.MODEL_NAME}_{result['horizon']}h' v{hw_model.version} in Hopsworks.")
    else:
        print(f"  HOPSWORKS_API_KEY not set — model saved locally only, at {model_dir}")


def run():
    print(f"Training pipeline starting for {config.CITY_NAME}")
    df = load_training_data()
    print(f"Loaded {len(df)} rows of training data.\n")

    for horizon in config.FORECAST_HORIZONS_HOURS:
        result = train_one_horizon(df, horizon)
        if result:
            save_model(result)
    print("\nDone.")


if __name__ == "__main__":
    run()
