"""
BACKFILL — run this ONCE, manually, to give the model something to
learn from before the hourly pipeline has had time to accumulate weeks
of its own data.

Open-Meteo can hand back up to 92 days of recent history in a single
call — no separate "historical" service needed, unlike our first
draft that juggled 3 different providers.

Usage:
    python src/backfill.py --days 90
"""
import argparse
import config
import data


def run(days: int):
    print(f"Downloading {days} days of history for {config.CITY_NAME}...")
    raw_df = data.fetch_history(days)
    print(f"Got {len(raw_df)} hourly rows.")

    engineered = data.build_features(raw_df)
    training_df = data.add_forecast_targets(engineered)
    training_df["timestamp_unix"] = (training_df["timestamp"].astype("int64") // 10**9).astype(int)

    # Always keep a local copy — handy for testing without Hopsworks.
    # training_df.to_parquet("backfill_data.parquet", index=False)
    # print("Saved local copy -> backfill_data.parquet")
    training_df.to_parquet(config.LOCAL_DATA_PATH, index=False)
    print(f"Saved local copy -> {config.LOCAL_DATA_PATH}")
    if config.HOPSWORKS_API_KEY:
        import hopsworks
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY, project=config.HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=config.FEATURE_GROUP_NAME,
            version=config.FEATURE_GROUP_VERSION,
            primary_key=["timestamp_unix"],
            event_time="timestamp",
            description=f"Hourly AQI + weather features for {config.CITY_NAME}",
            online_enabled=True,
        )
        fg.insert(training_df, write_options={"wait_for_job": True})
        print(f"Uploaded {len(training_df)} rows into Hopsworks.")
    else:
        print("HOPSWORKS_API_KEY not set — skipped upload, local parquet file only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run(args.days)
