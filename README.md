# landsliderisk-mlmodel

Random Forest risk-prediction model for **TerraGuard** — SIH26001, "AI-Based Early
Warning and Landslide Risk Monitoring System" (MDoNER, Disaster Management theme).
Team: Ecosentry.

## Status

Trained and working on a documented **proxy dataset** (see below). This is an
honest, explicitly-flagged limitation, not an oversight — no official NER-specific
landslide dataset with matching features was available in our build window. The
model, pipeline, and integration interface are complete and ready to swap in real
data once collected.

## Why proxy data, and where the numbers come from

`generate_dataset.py` builds a synthetic training set whose feature ranges and
risk relationships are grounded in real, published landslide susceptibility
research — including a study specifically covering **NH31A and East Sikkim
Himalaya settlements** (geographically adjacent to our NER target region), plus
broader Himalayan/Indian landslide literature. Specific cited ranges (e.g.
"landslides most prevalent on 35–50° slopes and at 2000–2500mm / 3000–3300mm
rainfall") are documented inline in that file with sources. This is not
random noise — it's a documented, inspectable, physically-motivated labeling
rule that can be scrutinized and replaced with real labels later.

**This is a proxy for demo purposes. It is designed to be retrained on real
field data post-deployment — see "Recalibration path" below.**

## Features

| Feature | Type | Source (live deployment) |
|---|---|---|
| `soil_moisture` | Live sensor | ESP32 + capacitive soil moisture sensor |
| `rainfall_mm` | Live | Rainfall API / IMD data |
| `slope` | Static, per-zone | SRTM DEM |
| `ndvi` | Static, per-zone | Sentinel-2 imagery |
| `landslide_density` | Static, per-zone | ISRO Bhuvan / GSI historical inventory |

**Tilt and vibration are deliberately NOT model inputs.** No historical tilt
dataset exists to train against. Instead, tilt is used as a **live escalation
check** (`check_tilt_escalation()` in `predict.py`): if the model already
predicts Medium/High risk from sensor + geospatial features, and tilt spikes
sharply above its baseline, the system escalates the warning one level. This
keeps the trained model honest about what it actually learned, while still
using tilt data where it's genuinely useful — as a live safety signal.

## Model choice: Random Forest, not deep learning

Deliberate choice, not a default:
- **Interpretable** — `feature_importance.png` shows exactly which factors drove
  a prediction. For a life-safety system, judges and field authorities need to
  trust *why* an alert fired, not just that it did.
- **Works on small/proxy tabular data** — deep learning needs far more labeled
  examples than we have access to.
- **No GPU dependency** — realistic for constrained NER field deployment.

## Files

- `generate_dataset.py` — builds the documented proxy dataset (run standalone to
  regenerate `proxy_landslide_dataset.csv`)
- `train.py` — trains the Random Forest, writes `model.pkl`, `metrics.txt`,
  `feature_importance.png`
- `predict.py` — **the integration point.** Clean `predict_risk(...)` and
  `check_tilt_escalation(...)` functions matching the signature agreed with the
  backend team (`app/ml_client.py`). Run directly (`python predict.py`) for a
  quick sanity check against hand-picked obvious scenarios.
- `model.pkl` — the trained classifier (committed so backend can integrate
  immediately without retraining)
- `metrics.txt` — honest evaluation results, explicitly labeled as proxy-data
  performance, not real-world accuracy
- `feature_importance.png` — for the pitch/demo, to show this isn't a black box

## Evaluation (proxy test set)

See `metrics.txt` for full output. Headline: **83.1% recall on High-risk class**
— we optimized for catching real high-risk cases (false negatives are far worse
than false alarms for a disaster-warning system) over raw accuracy.

**Report this honestly if asked**: these numbers describe how well the model
learned the proxy data's patterns, not validated real-world NER accuracy.

## Integration

```python
from predict import predict_risk, check_tilt_escalation, escalate_level

score, level = predict_risk(
    soil_moisture=59.0, rainfall_mm=2100, slope=38.0, ndvi=0.55,
    landslide_density=2.3,
)

if level in ("Medium", "High") and check_tilt_escalation(tilt_angle=5.1, tilt_baseline=0.8):
    level = escalate_level(level)
```

## Recalibration path (post-deployment)

1. Collect real sensor + rainfall readings from deployed nodes alongside actual
   ground-truth outcomes (confirmed slope events / non-events) over a monitoring
   period.
2. Replace `generate_dataset.py`'s synthetic generation with the real collected
   dataset (same column schema, so `train.py` needs no changes).
3. Retrain: `python train.py` — produces an updated `model.pkl` as a drop-in
   replacement, no changes needed on the backend integration side.

## Setup

```bash
pip install -r requirements.txt
python generate_dataset.py   # builds proxy_landslide_dataset.csv
python train.py              # trains model.pkl, metrics.txt, feature_importance.png
python predict.py            # sanity-check predictions on example scenarios
```
