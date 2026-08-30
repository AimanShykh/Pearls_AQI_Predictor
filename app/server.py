"""
FastAPI server — serves the AQI forecast + a simple HTML dashboard.
Deployed on Render, where we control the Python version directly,
avoiding the version conflicts that broke Streamlit Cloud.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_next_3_days, load_latest_features

app = FastAPI(title="AQI Forecast API — Hyderabad, Sindh")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/forecast")
def get_forecast():
    """Returns the current AQI + next-3-day forecast as JSON."""
    result = predict_next_3_days()
    return JSONResponse({
        "city": result["city"],
        "as_of": str(result["as_of"]),
        "current_aqi": result["current_aqi"],
        "current_category": result["current_category"],
        "forecasts": [
            {
                "horizon_hours": f["horizon_hours"],
                "target_time": str(f["target_time"]),
                "predicted_aqi": f["predicted_aqi"],
                "category": f["category"],
                "color": f["color"],
                "alert": f["alert"],
            }
            for f in result["forecasts"]
        ],
    })


@app.get("/api/history")
def get_history():
    """Returns the last 14 days of AQI for the trend chart."""
    df = load_latest_features().tail(24 * 14)
    return JSONResponse({
        "history": [
            {"timestamp": str(row["timestamp"]), "aqi": row["aqi"]}
            for _, row in df.iterrows()
        ]
    })


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AQI Forecast — Hyderabad, Sindh</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .subtitle { color: #666; margin-bottom: 24px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; text-align: center; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .label { color: #666; font-size: 0.85rem; margin-top: 4px; }
  .alert { background: #fdecea; border: 1px solid #f5c2c0; color: #8a1c12; padding: 14px; border-radius: 8px; margin-bottom: 20px; }
  canvas { max-height: 380px; }
  .footer { color: #999; font-size: 0.8rem; margin-top: 30px; }
</style>
</head>
<body>
  <h1>🌫️ AQI Forecast — Hyderabad, Sindh, Pakistan</h1>
  <div class="subtitle">Live from Open-Meteo → Hopsworks → FastAPI on Render.</div>
  <div id="alertBox"></div>
  <div class="cards" id="cards"><p>Loading forecast...</p></div>
  <canvas id="forecastChart"></canvas>
  <div class="footer" id="footer"></div>

<script>
  fetch("/api/forecast")
    .then(r => r.json())
    .then(data => {
      document.getElementById("cards").innerHTML = `
        <div class="card"><div class="value">${Math.round(data.current_aqi)}</div><div class="label">Now — ${data.current_category}</div></div>
        ${data.forecasts.map(f => `
          <div class="card"><div class="value">${Math.round(f.predicted_aqi)}</div>
          <div class="label">+${f.horizon_hours}h — ${f.category}</div></div>
        `).join("")}
      `;
      if (data.forecasts.some(f => f.alert)) {
        document.getElementById("alertBox").innerHTML =
          `<div class="alert">🚨 Hazardous AQI levels predicted in the next 3 days.</div>`;
      }
      const labels = [data.as_of, ...data.forecasts.map(f => f.target_time)];
      const values = [data.current_aqi, ...data.forecasts.map(f => f.predicted_aqi)];
      new Chart(document.getElementById("forecastChart"), {
        type: "line",
        data: { labels, datasets: [{ label: "AQI forecast", data: values, borderColor: "#2c7a7b", tension: 0.2 }] },
        options: { plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: "AQI" } } } }
      });
      document.getElementById("footer").innerText = "Last updated: " + data.as_of;
    })
    .catch(() => { document.getElementById("cards").innerHTML = "<p>Couldn't load forecast — has training run at least once?</p>"; });
</script>
</body>
</html>
"""
