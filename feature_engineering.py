"""
HISTORICAL BACKFILL — run ONCE (manually or via `workflow_dispatch`) to
populate the Feature Store with enough history to train a first model.

Data sources for history:
  - OpenWeather Air Pollution History API (free, hourly, up to ~1 year back)
  - OpenWeather weather history requires a paid plan, so as a serverless-
    friendly workaround we backfill weather from the Open-Meteo Archive API
    (https://archive-api.open-meteo.com), which is free and keyless.

Usage:
    python backfill.py --days 90
"""
import argparse
import time
import requests
import pandas as pd

import config
from utils import log
from feature_engineering import build_training_table, FEATURE_COLUMNS

try:
    import hopsworks
except ImportError:
    hopsworks = None


def fetch_owm_pollution_history(start_unix: int, end_unix: int) -> pd.DataFrame:
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "start": start_unix,
        "end": end_unix,
        "appid": config.OWM_API_KEY,
    }
    r = requests.get(config.OWM_AIR_POLLUTION_HISTORY_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for item in data["list"]:
        comp = item["components"]
        rows.append({
            "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),
            "aqi": None,  # OWM's history endpoint gives components, not AQICN's AQI scale
            "owm_aqi_index": item["main"]["aqi"],
            "pm25": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "o3": comp.get("o3"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "co": comp.get("co"),
        })
    return pd.DataFrame(rows)


def fetch_open_meteo_weather_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Free, keyless historical weather (hourly) for lat/lon."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"], utc=True),
        "temp": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "pressure": data["surface_pressure"],
        "wind_speed": data["wind_speed_10m"],
        "wind_deg": data["wind_direction_10m"],
        "clouds": data["cloud_cover"],
    })
    return df


def us_aqi_from_pm25(pm25: float) -> float:
    """Convert PM2.5 concentration (µg/m³) to US EPA AQI, used to backfill a
    target 'aqi' column consistent with AQICN's reporting scale, since OWM's
    history endpoint only returns raw pollutant concentrations."""
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
    ]
    if pm25 is None or pd.isna(pm25):
        return None
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(a_lo + (a_hi - a_lo) / (c_hi - c_lo) * (pm25 - c_lo), 1)
    return 500.0


def run(days: int):
    if not config.OWM_API_KEY:
        raise EnvironmentError("OWM_API_KEY required for backfill")

    end = pd.Timestamp.utcnow().floor("h")
    start = end - pd.Timedelta(days=days)
    log.info(f"Backfilling {config.CITY_NAME} from {start} to {end}")

    # OWM history API caps ~1 week per call on the free tier -> chunk requests
    pollution_frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=7), end)
        log.info(f"  pollution chunk {cursor} -> {chunk_end}")
        try:
            df_chunk = fetch_owm_pollution_history(int(cursor.timestamp()), int(chunk_end.timestamp()))
            pollution_frames.append(df_chunk)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  chunk failed: {e}")
        cursor = chunk_end
        time.sleep(1)  # be nice to the free-tier rate limit

    pollution_df = pd.concat(pollution_frames, ignore_index=True).drop_duplicates("timestamp")
    pollution_df["aqi"] = pollution_df["pm25"].apply(us_aqi_from_pm25)

    weather_df = fetch_open_meteo_weather_history(
        start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")
    )

    raw_df = pd.merge_asof(
        pollution_df.sort_values("timestamp"),
        weather_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    raw_df = raw_df.dropna(subset=["aqi"]).sort_values("timestamp").reset_index(drop=True)
    log.info(f"Assembled {len(raw_df)} raw hourly rows")

    training_df = build_training_table(raw_df)
    training_df["timestamp_unix"] = (training_df["timestamp"].astype("int64") // 10**9).astype(int)

    out_path = "backfill_training_data.parquet"
    training_df.to_parquet(out_path, index=False)
    log.info(f"Saved local copy -> {out_path}")

    if hopsworks and config.HOPSWORKS_API_KEY:
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            primary_key=["timestamp_unix"],
            event_time="timestamp",
            description=f"Hourly engineered AQI features for {config.CITY_NAME}",
            online_enabled=True,
        )
        fg.insert(training_df, write_options={"wait_for_job": True})
        log.info(f"Backfilled {len(training_df)} rows into Hopsworks feature group '{config.FEATURE_GROUP_NAME}'")
    else:
        log.warning("HOPSWORKS_API_KEY not set — skipped upload, local parquet only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="How many days of history to backfill")
    args = parser.parse_args()
    run(args.days)
