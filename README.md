# ml
# Landslide Risk Prediction — ML Module

AI-based landslide risk prediction system for the **Paglajhora–Tindharia Corridor, Darjeeling Himalayas**, built for Smart India Hackathon (SIH) 2026.

This repository contains the machine learning component of the project: data preprocessing, model training, evaluation, and deployment as an API that powers real-time landslide risk alerts.

## Overview

The model predicts landslide risk at locations within the study area, classifying risk into four levels — **Low / Medium / High / Critical** — using environmental and geospatial features. It is trained on historical data and deployed to work with live sensor and API feeds, with an additional tilt-sensor safety layer for real-time emergency escalation.

### Monitored Zones

The dashboard monitors four zones:

- **Sohra (Meghalaya)**
- **Mawsynram (Meghalaya)**
- **Aizawl (Mizoram)**
- **One NH (National Highway) corridor point**

### Training vs. Inference — One Model, Four Zones

The model is trained on a **single dataset** — whichever available dataset has the most complete features (Sohra-area data preferred if available, otherwise any Indian/Himalayan landslide dataset with a similar feature set). This is purely a training-data choice, not a decision about which zones are monitored.

At **prediction time**, the same trained model is used for all four zones. Each zone's own live sensor readings and static features (rainfall, soil moisture, slope, NDVI, landslide density) are fed into the model independently to generate that zone's risk prediction. So:

- **One trained model** (learned from one dataset's patterns)
- **Four independent predictions** (one per zone, each using that zone's own input data)

## Model

| | |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Inputs** | Rainfall, Soil Moisture, Slope, NDVI, Landslide Density |
| **Output** | Risk probability + Risk level (Low / Medium / High / Critical) |

### Training Phase
The model is trained on historical rainfall, historical soil moisture, slope, NDVI, and historical landslide inventory data. The trained model is saved/serialized for deployment.

### Deployment Phase
The deployed model consumes **live rainfall** and **live soil moisture** values to predict current landslide risk in real time.

### Tilt Sensor (Safety Layer)
Historical tilt data is generally unavailable, so the tilt sensor is **not** used to train the model. Instead, it continuously monitors slope movement post-deployment. If the AI predicts high risk **and** the tilt sensor detects rapid change, the system automatically upgrades the warning level and triggers an emergency alert.

## Data Sources

| Feature | Source |
|---|---|
| Rainfall | ERA5 / IMD |
| Soil Moisture | Historical data (training) → live sensor (deployment) |
| Slope | Calculated from SRTM DEM |
| NDVI | Sentinel-2 imagery |
| Landslide Density | Historical landslide inventory (ISRO / GSI) |

## Workflow

```
Historical Data (single best-feature dataset,
Sohra-area preferred, else similar Himalayan dataset)
      │
      ▼
Train Random Forest  →  Single Trained Model
      │
      ▼
Deploy Model
      │
      ├──▶ Sohra:      Live Rainfall + Soil Moisture + Static Features ──▶ Risk Prediction
      ├──▶ Mawsynram:  Live Rainfall + Soil Moisture + Static Features ──▶ Risk Prediction
      ├──▶ Aizawl:     Live Rainfall + Soil Moisture + Static Features ──▶ Risk Prediction
      └──▶ NH Corridor:Live Rainfall + Soil Moisture + Static Features ──▶ Risk Prediction
                              │
                              ▼
                      Tilt Sensor Check (per zone)
                              │
                              ▼
                      Dashboard & Alerts (4 zones)
```

## Tech Stack

- **Language/Libraries:** Python, Scikit-learn, Pandas, NumPy
- **Geospatial:** QGIS, Sentinel-2, SRTM DEM, ERA5 / IMD
- **Serving:** FastAPI
- **Hardware/Sensors:** ESP32, Soil Moisture Sensor, Tilt Sensor

## Setup

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python src/training/train.py
```

**Run the prediction API:**
```bash
uvicorn src.api.main:app --reload
```

*(Update these commands once actual script/file names are finalized.)*

## Team / Project

Part of the **SIH 2026** submission — AI-Based Landslide Risk Prediction for the Paglajhora–Tindharia Corridor, Darjeeling Himalayas.
