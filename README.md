# AQI Forecast — Hyderabad, Sindh, Pakistan 🌫️

Predicts Air Quality Index 24h / 48h / 72h ahead, running entirely on
free, serverless infrastructure — no server you manage, no API keys
except one (for storage).

## Why this version is simpler than a typical tutorial you'll find online

Most AQI-forecast tutorials wire together 2-3 different data providers
(one for current AQI, one for weather, one for history) because most
weather APIs gate their historical/forecast data behind paid tiers.

This project uses **one data source for everything: [Open-Meteo](https://open-meteo.com)**.
It's free, needs **no signup or API key**, already computes the AQI
number for you (`us_aqi`, same 0-500 scale as any AQI app), and gives
current + forecast + up-to-92-days-history in the same two endpoints.
That cuts real complexity out of the code, not just the explanation.

The only account you truly need to create is **Hopsworks** (also free),
which acts as the shared storage between the hourly data-collector and
the dashboard — see "Why a feature store?" below.

## How it works (plain English)

```
Open-Meteo API  →  Feature Pipeline  →  Hopsworks Feature Store
(no key needed)    (runs every hour,      (a shared, growing table
                     GitHub Actions)        of AQI + weather history)
                                                    │
                                                    ▼
                                          Training Pipeline
                                          (runs once a day,
                                           GitHub Actions)
                                                    │
                                                    ▼
                                          Hopsworks Model Registry
                                          (stores the trained models)
                                                    │
                                                    ▼
                                          Streamlit Dashboard
                                          (loads the latest model
                                           + features, shows forecast)
```

1. **Every hour**, a small script asks Open-Meteo "what's the AQI and
   weather right now?", turns that into model-ready features, and
   saves it to Hopsworks.
2. **Once a day**, another script pulls all the saved history, trains
   two candidate models (Ridge Regression, Random Forest) for each of
   the 3 forecast horizons, keeps whichever did better, and saves it.
3. **Whenever you open the dashboard**, it grabs the latest saved
   features and the latest saved models, and shows you the forecast.

Nothing here needs a server running 24/7 — GitHub Actions wakes up,
runs the script, and goes back to sleep. That's what "serverless" means.

## Why a feature store at all? (Why not just a CSV file?)

Two real reasons:
- The **exact same feature-computation code** (`data.py`) needs to run
  both when training the model and when making a live prediction. If
  those two ever compute features even slightly differently, the model
  gets confused — a classic ML bug called training/serving skew. A
  feature store is built to prevent that; a loose CSV file doesn't help
  with it.
- Multiple separate scripts (hourly collector, daily trainer, dashboard)
  need to reliably read/write the same growing dataset without
  clobbering each other.

## Repo layout

```
├── src/
│   ├── config.py             # city coordinates, thresholds, settings
│   ├── data.py                # fetch from Open-Meteo + all feature engineering
│   ├── feature_pipeline.py    # HOURLY job
│   ├── backfill.py            # ONE-TIME job (run manually at setup)
│   ├── training_pipeline.py   # DAILY job
│   └── inference.py           # loads model+features, produces the forecast
├── app/
│   └── dashboard.py            # Streamlit dashboard
├── .github/workflows/
│   ├── feature_pipeline.yml    # cron: every hour
│   ├── training_pipeline.yml   # cron: daily
│   └── backfill.yml            # manual trigger, run once
└── requirements.txt
```

Only 6 Python files. `data.py` is the one worth reading closely — it
has a comment above every function explaining *why* that feature
helps, not just what it does.

## Setup

### 1. Create a free Hopsworks account
Go to [hopsworks.ai](https://www.hopsworks.ai/), sign up, create a
project (e.g. `aqi_hyderabad_sindh`), then generate an API key under
Account Settings → API Keys. That's the only account you need.

### 2. Add 2 GitHub Secrets
On your repo: Settings → Secrets and variables → Actions → New secret
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`

(Open-Meteo needs no key, so there's nothing else to add.)

### 3. Run the one-time backfill
GitHub → Actions tab → **Historical Backfill (run once, manually)** →
Run workflow. This gives the model ~90 days of history to learn from.

### 4. Let automation take over
`feature_pipeline.yml` starts running hourly automatically once merged.
`training_pipeline.yml` runs once a day, trains both models per
horizon, and keeps the better one.

### 5. Deploy the dashboard
Push to GitHub → go to [share.streamlit.io](https://share.streamlit.io)
→ "New app" → point at this repo, main file `app/dashboard.py` → add
the same 2 secrets in Streamlit's secrets manager. Done.

## Local development

```bash
pip install -r requirements.txt
export HOPSWORKS_API_KEY=...  HOPSWORKS_PROJECT=...

python src/backfill.py --days 90     # one-time
python src/feature_pipeline.py       # simulate one hourly run
python src/training_pipeline.py      # train + register models
python src/inference.py               # print the forecast to console
streamlit run app/dashboard.py        # launch dashboard locally
```

If you don't set the Hopsworks variables, everything still runs using
a local `backfill_data.parquet` file and a local `models/` folder —
handy for testing the ML logic with zero setup at all.

## Modeling notes
- **Separate model per horizon** (24h/48h/72h) instead of one model
  guessing all three, to avoid compounding error on longer horizons.
- **Two candidate models**: Ridge Regression (simple, fast baseline)
  and Random Forest (usually more accurate, still explainable). The
  training script prints both models' scores so you can see the
  comparison yourself.
- **Time-based train/test split** — never shuffle time series data.
  We train on the older 80% and test on the newer 20%, mimicking the
  real task of predicting the future from the past.
- **Metrics**: RMSE (average error, punishes big misses more), MAE
  (average error, plain), R² (how much variation the model explains).
