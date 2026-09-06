"""
FEATURE PIPELINE 
"""
import pandas as pd
import config
import data

import hopsworks


def get_feature_group():
   
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        primary_key=["timestamp_unix"],
        event_time="timestamp",
        description=f"Hourly AQI + weather features for {config.CITY_NAME}",
        online_enabled=True,
        time_travel_format="HUDI",
    )


def run():
    print(f"Fetching current conditions for {config.CITY_NAME}...")
    current_row = data.fetch_current_conditions()
    new_df = pd.DataFrame([current_row])

    fg = get_feature_group()

    # Pull recent history so lag/rolling features have real data to
    # look back on (a brand new row alone can't compute "AQI 24h ago").
    try:
        history_df = fg.read()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        cutoff = new_df["timestamp"].iloc[0] - pd.Timedelta(hours=72)
        history_df = history_df[history_df["timestamp"] >= cutoff]
        combined = pd.concat([history_df, new_df], ignore_index=True)
    except Exception:
        print("No existing history yet (probably the first-ever run) — that's fine.")
        combined = new_df

    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)  # force correct dtype
    engineered = data.build_features(combined)

    # We already have every row except the newest one saved 
    latest_row = engineered.tail(1).copy()
    latest_row["timestamp_unix"] = (latest_row["timestamp"].astype("int64") // 10**9).astype(int)

    fg.insert(latest_row, write_options={"wait_for_job": False})
    print(f"Saved feature row for {latest_row['timestamp'].iloc[0]} — AQI = {latest_row['aqi'].iloc[0]}")


if __name__ == "__main__":
    run()
