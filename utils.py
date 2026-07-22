"""
Pure feature-engineering functions.
Kept separate from I/O so the same logic is used by:
  - the hourly feature pipeline (single new row + recent history)
  - the historical backfill script (bulk transform)
  - the inference script (must replicate training-time features exactly)
"""
import numpy as np
import pandas as pd
import config


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # cyclical encodings so the model sees hour 23 and hour 0 as adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:
    """Requires df sorted ascending by time with a regular hourly cadence."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    for lag in config.LAG_HOURS:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)

    for window in config.ROLLING_WINDOWS_HOURS:
        df[f"{target_col}_rollmean_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_rollstd_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )

    # AQI change rate (1st derivative) and acceleration (2nd derivative)
    df[f"{target_col}_change_rate_1h"] = df[target_col].diff(1)
    df[f"{target_col}_change_rate_3h"] = df[target_col].diff(3)
    df[f"{target_col}_acceleration"] = df[f"{target_col}_change_rate_1h"].diff(1)

    return df


def add_forecast_targets(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:
    """Create the supervised-learning targets: AQI 24h/48h/72h ahead."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    for h in config.FORECAST_HORIZONS_HOURS:
        df[f"target_{target_col}_{h}h"] = df[target_col].shift(-h)
    return df


def build_feature_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: raw hourly rows -> engineered feature table.
    raw_df must contain: timestamp, aqi, pm25, pm10, o3, no2, so2, co,
                          temp, humidity, pressure, wind_speed, wind_deg, clouds
    """
    df = add_time_features(raw_df)
    df = add_lag_and_rolling_features(df)
    return df


def build_training_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = build_feature_row(raw_df)
    df = add_forecast_targets(df)
    return df


FEATURE_COLUMNS = (
    ["hour", "day", "month", "day_of_week", "is_weekend",
     "hour_sin", "hour_cos", "month_sin", "month_cos"]
    + config.RAW_POLLUTANT_COLS
    + config.RAW_WEATHER_COLS
    + [f"aqi_lag_{h}h" for h in config.LAG_HOURS]
    + [f"aqi_rollmean_{w}h" for w in config.ROLLING_WINDOWS_HOURS]
    + [f"aqi_rollstd_{w}h" for w in config.ROLLING_WINDOWS_HOURS]
    + ["aqi_change_rate_1h", "aqi_change_rate_3h", "aqi_acceleration"]
)
