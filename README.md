# AirGuard — Can We Predict the Air You're Breathing?

A machine learning project that forecasts PM2.5 air pollution across the Los Angeles Basin — before it becomes a health emergency.

---

## The Problem

Every year, millions of people in Southern California wake up not knowing whether the air outside is safe to breathe. PM2.5 — fine particulate matter smaller than 2.5 micrometers — is one of the most dangerous air pollutants. It gets deep into your lungs, into your bloodstream, and contributes to heart disease, asthma, and premature death.

The problem isn't just wildfires. It's the slow, everyday accumulation from freeways, ports, factories, and weather patterns that create invisible danger zones across Los Angeles, Orange, Riverside, and San Bernardino counties — home to over 16 million people.

AirGuard asks a simple question: **given everything we know right now, what will the air look like in the next few hours?**

---

## What This Project Does

AirGuard pulls together data from EPA monitoring stations, weather sensors, and traffic infrastructure, then trains machine learning models to predict PM2.5 levels across the South Coast Air Basin.

The goal isn't just accuracy — it's understanding *why* the air gets bad. Is it wind patterns? Proximity to the 405? A wildfire three counties over? AirGuard breaks this down so the answer isn't just a number, but a story.

---

## The Data

Everything comes from publicly available sources:

- **EPA AQS** — hourly readings from PM2.5, NO2, ozone, SO2, and CO monitoring stations
- **Weather stations** — temperature, wind speed, humidity, pressure
- **US Census Bureau** — population data to verify that monitoring stations actually cover where people live
- **OpenStreetMap** — real freeway geometry to calculate how close each location is to major roads
- **Hardcoded infrastructure** — distances to ports (LA/Long Beach), airports (LAX, ONT, SNA), and key freeways (I-5, I-10, I-405, and more)

---

## How It Works

The pipeline runs in five stages:

**1. Load & Clean**
Raw EPA CSV files are messy — inconsistent column names, mixed date formats, different encodings. AirGuard normalizes all of it into a clean, unified dataset.

**2. Spatial Matching**
Not every PM2.5 sensor has a weather station next door. AirGuard uses spatial math (BallTree nearest-neighbor search) to match each air quality site to its closest weather and pollutant stations.

**3. Feature Engineering**
The model learns from:
- Past PM2.5 readings (lags and rolling averages)
- Weather conditions (wind, humidity, temperature)
- Other pollutants as proxy signals
- Time patterns (hour of day, day of week, season)
- Distance to freeways, ports, and airports

**4. Model Training**
Three models are trained and compared:

| Model | What it represents |
|---|---|
| Linear Regression | The simplest possible baseline |
| Random Forest | Handles non-linear patterns well |
| XGBoost | State-of-the-art gradient boosting |

Each model is tested across five feature combinations — from "PM2.5 history only" all the way up to "everything we've got." This ablation study answers: *does adding more data actually help, or does the algorithm matter more?*

**5. Explainability**
The best model is opened up with SHAP values — a technique that shows exactly which features drove each prediction. No black boxes.

---

## Wildfire Mode

Wildfires change everything. A model trained on normal days will completely fail during a fire event. AirGuard detects wildfire regimes and evaluates the model separately for fire vs. non-fire conditions — showing exactly where performance degrades and by how much.

---

## Validation

Two types of cross-validation confirm the results aren't just fitting to noise:

- **Spatial holdout** — hold out entire monitoring sites the model has never seen
- **Temporal walk-forward** — train on the past, predict the future, month by month

Both are stricter than random splits, which is the point. Real-world performance matters more than leaderboard numbers.

---

## The Maps

AirGuard generates publication-quality choropleth maps showing:
- Average observed PM2.5 across the basin
- Model predictions overlaid on real geography
- Where predictions are most and least accurate
- Wildfire hotspots
- Seasonal patterns (summer vs. winter pollution profiles)

---

## Setup

This project runs on **Google Colab** with data stored in **Google Drive**.

**1. Mount your Drive in Colab:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**2. Install dependencies (first time only):**
```
!pip install -q geopandas shapely osmnx contextily xgboost shap scipy
```

**3. Organize your data in Drive:**
```
MyDrive/AIRGUARD/
├── PM 2.5/       ← EPA PM2.5 hourly CSVs
├── weather/      ← Meteorology CSVs
├── NO2/
├── O3/
├── SO2/
├── CO/
└── outputs/      ← Created automatically
```

**4. Run the notebook top to bottom.** That's it.

All figures and prediction CSVs are saved automatically to `AIRGUARD/outputs/`.

---

## Coverage Area

**South Coast Air Basin, California**
- Los Angeles County (9.8M people)
- Orange County (3.2M people)
- Riverside County (2.5M people)
- San Bernardino County (2.2M people)

Total: ~17.7 million people. One of the most polluted airsheds in the United States.

---

## Why It Matters

Air quality forecasting is mostly locked inside government agencies and private companies. AirGuard is an open, reproducible attempt to build something that works — and explain it clearly enough that a policy maker, a researcher, or a concerned parent could understand what the model is saying and why.

The air doesn't care about county lines. Neither does this project.
