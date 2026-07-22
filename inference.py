"""
TRAINING PIPELINE — runs daily via GitHub Actions
(.github/workflows/training_pipeline.yml).

1. Read engineered features from Hopsworks Feature View.
2. Train/evaluate several candidate models per forecast horizon
   (24h / 48h / 72h): Ridge, RandomForest, XGBoost.
3. Pick the best model per horizon by RMSE on a held-out time-based split.
4. Log metrics + save models to Hopsworks Model Registry.
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
from utils import log
from feature_engineering import FEATURE_COLUMNS

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import hopsworks
except ImportError:
    hopsworks = None


def get_training_data() -> pd.DataFrame:
    """Prefer Hopsworks; fall back to local parquet from backfill.py for local dev."""
    if hopsworks and config.HOPSWORKS_API_KEY:
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        try:
            fv = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)
        except Exception:  # noqa: BLE001
            query = fg.select_all()
            fv = fs.create_feature_view(
                name=config.FEATURE_VIEW_NAME,
                version=config.FEATURE_VIEW_VERSION,
                query=query,
            )
        df = fv.get_batch_data()
        return df
    log.warning("Hopsworks not configured — reading local backfill_training_data.parquet")
    return pd.read_parquet("backfill_training_data.parquet")


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def build_candidate_models():
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ),
    }
    if xgb is not None:
        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    return models


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_for_horizon(df: pd.DataFrame, horizon: int):
    target_col = f"target_aqi_{horizon}h"
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    data = df.dropna(subset=cols + [target_col]).copy()

    if len(data) < 50:
        log.warning(f"Not enough rows ({len(data)}) to train {horizon}h model — skipping")
        return None

    train_df, test_df = time_based_split(data)
    X_train, y_train = train_df[cols], train_df[target_col]
    X_test, y_test = test_df[cols], test_df[target_col]

    results = {}
    fitted = {}
    for name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        fitted[name] = model
        log.info(f"[{horizon}h] {name}: RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    log.info(f"[{horizon}h] best model = {best_name}")

    return {
        "horizon": horizon,
        "best_model_name": best_name,
        "best_model": fitted[best_name],
        "metrics": results,
        "feature_columns": cols,
    }


def save_and_register(result: dict):
    horizon = result["horizon"]
    model_dir = f"models/aqi_{horizon}h"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(result["best_model"], model_path)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "horizon_hours": horizon,
            "best_model_name": result["best_model_name"],
            "all_model_metrics": result["metrics"],
            "feature_columns": result["feature_columns"],
        }, f, indent=2)

    if hopsworks and config.HOPSWORKS_API_KEY:
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        mr = project.get_model_registry()
        best_metrics = result["metrics"][result["best_model_name"]]
        hw_model = mr.python.create_model(
            name=f"{config.MODEL_NAME}_{horizon}h",
            metrics=best_metrics,
            description=f"AQI forecast model, {horizon}h horizon, Hyderabad Sindh. "
                         f"Best algo: {result['best_model_name']}",
        )
        hw_model.save(model_dir)
        log.info(f"Registered model '{config.MODEL_NAME}_{horizon}h' v{hw_model.version} in Hopsworks Model Registry")
    else:
        log.warning(f"Hopsworks not configured — model saved locally only at {model_dir}")


def run():
    log.info(f"Starting training pipeline for {config.CITY_NAME}")
    df = get_training_data()
    log.info(f"Loaded {len(df)} training rows")

    for horizon in config.FORECAST_HORIZONS_HOURS:
        result = train_for_horizon(df, horizon)
        if result:
            save_and_register(result)

    log.info("Training pipeline complete.")


if __name__ == "__main__":
    run()
