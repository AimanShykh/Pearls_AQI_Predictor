"""
FEATURE PIPELINE — runs every hour via GitHub Actions (see
.github/workflows/feature_pipeline.yml).

1. Pull current AQI/pollutants (AQICN) + weather (OpenWeather).
2. Merge into one raw row for this timestamp.
3. Pull recent history (last 72h) from Hopsworks to compute lag/rolling
   features correctly (features must see real history, not just 1 row).
4. Engineer features.
5. Upsert the new engineered row into the Hopsworks Feature Group.

100% serverless: no server to keep running — GitHub Actions cron invokes
this script, it does its work, and exits.
"""
import sys
import pandas as pd

import config
from utils import fetch_aqicn_current, fetch_openweather_pollution, fetch_openweather_weather, log
from feature_engineering import build_feature_row

try:
    import hopsworks
except ImportError:
    hopsworks = None


def get_feature_store():
    if hopsworks is None:
        raise ImportError("pip install hopsworks")
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
    return project.get_feature_store()


def get_or_create_feature_group(fs):
    return fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        primary_key=["timestamp_unix"],
        event_time="timestamp",
        description=f"Hourly engineered AQI features for {config.CITY_NAME}",
        online_enabled=True,
    )


def fetch_raw_row() -> dict:
    """Combine AQICN (ground truth AQI + pollutants) with OpenWeather (weather)."""
    aqicn = fetch_aqicn_current()
    weather = fetch_openweather_weather()

    row = {
        "timestamp": aqicn["timestamp"],
        "aqi": aqicn["aqi"],
        "pm25": aqicn["pm25"],
        "pm10": aqicn["pm10"],
        "o3": aqicn["o3"],
        "no2": aqicn["no2"],
        "so2": aqicn["so2"],
        "co": aqicn["co"],
        "temp": weather["temp"],
        "humidity": weather["humidity"],
        "pressure": weather["pressure"],
        "wind_speed": weather["wind_speed"],
        "wind_deg": weather["wind_deg"],
        "clouds": weather["clouds"],
    }

    # OpenWeather pollution as a fallback / cross-check for any missing AQICN fields
    try:
        owm_pol = fetch_openweather_pollution()
        for k in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            owm_key = "pm2_5" if k == "pm25" else k
            if row.get(k) is None:
                row[k] = owm_pol.get(owm_key)
    except Exception as e:  # noqa: BLE001
        log.warning(f"OpenWeather pollution fallback failed: {e}")

    return row


def run():
    log.info(f"Running hourly feature pipeline for {config.CITY_NAME}")

    new_row = fetch_raw_row()
    new_df = pd.DataFrame([new_row])

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    # Pull last 72h of history so lag/rolling features are computed correctly
    try:
        history_df = fg.read()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        cutoff = new_df["timestamp"].iloc[0] - pd.Timedelta(hours=96)
        history_df = history_df[history_df["timestamp"] >= cutoff]
        combined = pd.concat([history_df, new_df], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        log.warning(f"No existing history found (first run?): {e}")
        combined = new_df

    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    engineered = build_feature_row(combined)

    # Only insert the newest row — history rows are already stored
    latest = engineered.tail(1).copy()
    latest["timestamp_unix"] = (latest["timestamp"].astype("int64") // 10**9).astype(int)

    fg.insert(latest, write_options={"wait_for_job": True})
    log.info(f"Inserted feature row for {latest['timestamp'].iloc[0]} — AQI={latest['aqi'].iloc[0]}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        log.error(f"Feature pipeline failed: {exc}")
        sys.exit(1)
