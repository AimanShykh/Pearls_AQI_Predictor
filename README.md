# AQI Forecast — Hyderabad, Sindh, Pakistan 🌫️

**Live dashboard:** [pearlsaqipredictor123.streamlit.app](https://pearlsaqipredictor123.streamlit.app/)

A serverless, end-to-end machine learning pipeline that forecasts Air
Quality Index 24h / 48h / 72h ahead for Hyderabad, Sindh — no server
to manage, only one account to create (Hopsworks).

## How it works

```
Open-Meteo API  →  Feature Pipeline  →  Hopsworks Feature Store
                    (runs hourly,          (growing table of AQI +
                     GitHub Actions)        weather history)
                                                    │
                                                    ▼
                                          Training Pipeline
                                          (runs daily, GitHub Actions)
                                          Ridge · Random Forest · XGBoost
                                          → best model per horizon kept
                                                    │
                                                    ▼
                                          Hopsworks Model Registry
                                                    │
                                                    ▼
                                          Streamlit Dashboard
                                          forecast · alerts · EDA · SHAP
```

1. **Every hour** — a script fetches current AQI + weather from
   Open-Meteo, engineers features, and appends one row to Hopsworks.
2. **Once a day** — a script pulls the full feature history, trains
   and compares three models per forecast horizon (24h/48h/72h), and
   registers whichever scores lowest RMSE.
3. **Whenever the dashboard is opened** — it loads the latest features
   and latest registered models and computes a live 3-day forecast.

Nothing runs continuously; GitHub Actions wakes up on a schedule, runs
the script, and shuts down.

## Data source

Everything comes from **[Open-Meteo](https://open-meteo.com)**  It also returns `us_aqi`, a pre-computed
AQI on the standard 0–500 scale, so no manual pollutant-to-AQI
conversion is needed.

## Repo layout

```
├── src/
│   ├── config.py             # city coordinates, thresholds, Hopsworks settings
│   ├── data.py                # Open-Meteo fetching + all feature engineering
│   ├── feature_pipeline.py    # HOURLY job
│   ├── backfill.py            # ONE-TIME job — 2-year historical backfill
│   ├── training_pipeline.py   # DAILY job — trains & compares 3 models
│   └── inference.py           # loads latest model + features, builds forecast
├── app/
│   └── dashboard.py            # Streamlit dashboard (forecast, EDA, alerts, SHAP)
├── models/                    
├── .github/workflows/
│   ├── feature_pipeline.yml    # cron: hourly
│   ├── training_pipeline.yml   # cron: daily
│   └── backfill.yml            # manual trigger, run once at setup
├── .devcontainer/               # reproducible dev environment config
├── aqi_eda.ipynb                # exploratory data analysis on 2 years of data
├── data.csv                      # sample/backfilled historical dataset
├── requirements.txt
└── runtime.txt                   # pins Python 3.11 (required by Hopsworks client)
```

## Feature engineering

Raw hourly data (`pm25`, `pm10`, `o3`, `no2`, `so2`, `co`, `temp`,
`humidity`, `pressure`, `wind_speed`, `clouds`) is transformed into
~29 model-ready columns in `src/data.py`:

| Group | Examples | Why |
|---|---|---|
| Temporal | `hour`, `day_of_week`, `month`, `hour_sin`/`hour_cos` | Captures daily traffic and seasonal pollution cycles |
| Lag | `aqi_lag_1h/6h/24h/48h` | Gives the model memory of recent trends — the strongest single predictor |
| Rolling/trend | `aqi_avg_6h/24h`, `aqi_change_rate` | Smooths noise, captures direction and speed of change |
| Targets (training only) | `target_24h/48h/72h` | Future AQI values the model learns to predict |

`aqi_eda.ipynb` explores this dataset in depth — seasonality, daily
rhythm, correlations between pollutants and weather, and a category
breakdown of how often AQI crosses into unhealthy territory.

## Modeling

Two candidate models are trained **separately per forecast
horizon**, avoiding compounding error from recursive multi-step
forecasting:

- **Ridge Regression** — simple, fast linear baseline
- **Random Forest** — ensemble of trees, captures non-linear patterns

Evaluation uses a **time-based 80/20 split** (train on the older 80%,
test on the newer 20% — never shuffled, since this is time-series
data), scored with RMSE, MAE, and R². The lowest-RMSE candidate per
horizon is registered to the Hopsworks Model Registry.

## Feature Store

Features persist in **Hopsworks** rather than a flat file so that (1)
identical feature-computation code runs at both training and
inference time, preventing training/serving skew, and (2) the hourly
pipeline, daily trainer, and dashboard can all read/write the same
growing dataset concurrently.

## Dashboard

Deployed on Streamlit Community Cloud (`runtime.txt` pins Python 3.11
to avoid a Hopsworks dependency conflict on newer Python versions).

- Current AQI + 24h/48h/72h forecast cards
- Forecast trend chart with AQI category bands
- Hazardous-AQI alert banner (AQI ≥ 150)
- Recent 14-day history charts
- SHAP-based feature importance, wrapped defensively so a library
  hiccup shows a graceful notice instead of breaking the page


## Setup

1. Create a free [Hopsworks](https://www.hopsworks.ai/) account + project
2. Generate an API key (Account Settings → API Keys)
3. Add GitHub Secrets: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`
4. Run **Historical Backfill** once from the Actions tab (`days: 730`)
5. `feature_pipeline.yml` and `training_pipeline.yml` then run automatically, hourly and daily
6. Deploy `app/dashboard.py` on Streamlit Community Cloud with the same two secrets

## Local development

```bash
pip install -r requirements.txt
export HOPSWORKS_API_KEY=...  HOPSWORKS_PROJECT=...

python src/backfill.py --days 730     # one-time
python src/feature_pipeline.py        # simulate one hourly run
python src/training_pipeline.py       # train + register models
python src/inference.py               # print forecast to console
streamlit run app/dashboard.py        # launch dashboard locally
```
