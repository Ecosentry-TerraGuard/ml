"""
predict.py — Clean prediction interface for the TerraGuard backend to import.

This is the file Person F (backend) swaps into app/ml_client.py in place of the
current mock. Function signatures match what was agreed with backend: live sensor
features and static per-zone geospatial features are passed separately, and tilt
is NOT a model input (no historical tilt data exists to train on) -- it's a
live safety-escalation check applied AFTER the model's prediction.

Usage from backend:
    from predict import predict_risk, check_tilt_escalation

    score, level = predict_risk(
        soil_moisture=59.0, rainfall_mm=2100, slope=38.0, ndvi=0.55,
        landslide_density=2.3,
    )
    if check_tilt_escalation(tilt_angle=5.1, tilt_baseline=0.8) and level in ("Medium", "High"):
        level = _escalate(level)
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

RiskLevel = Literal["Low", "Medium", "High"]

_MODEL_PATH = Path(__file__).parent / "model.pkl"
_model = None  # lazy-loaded


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_risk(
    soil_moisture: float,
    rainfall_mm: float,
    slope: float,
    ndvi: float,
    landslide_density: float,
) -> tuple[float, RiskLevel]:
    """
    Returns (risk_score 0-100, risk_level) using live sensor + rainfall data
    combined with static per-zone geospatial features (slope, NDVI, landslide
    density -- computed once per zone from DEM/satellite data, not live sensors).

    Tilt is deliberately NOT an input here -- see check_tilt_escalation() below.
    """
    model = _get_model()
    row = pd.DataFrame([{
        "soil_moisture": soil_moisture,
        "rainfall_mm": rainfall_mm,
        "slope": slope,
        "ndvi": ndvi,
        "landslide_density": landslide_density,
    }])

    level = model.predict(row)[0]
    # Use class probabilities to derive a smooth 0-100 score for the dashboard,
    # rather than just the hard class label.
    proba = model.predict_proba(row)[0]
    class_order = list(model.classes_)
    weights = {"Low": 15, "Medium": 50, "High": 90}
    score = sum(proba[class_order.index(c)] * weights[c] for c in class_order)

    return round(float(score), 1), level


def check_tilt_escalation(tilt_angle: float, tilt_baseline: float, threshold_delta: float = 3.0) -> bool:
    """
    Returns True if tilt has moved sharply enough above its baseline to warrant
    escalating an already-elevated risk level, regardless of what the ML model
    predicted from sensor + geospatial features alone.

    Historical tilt data doesn't exist for training (no labeled dataset of
    tilt-angle-over-time for past landslide events), so tilt is used as a live
    safety override rather than a trained model feature.
    """
    return (tilt_angle - tilt_baseline) >= threshold_delta


def escalate_level(level: RiskLevel) -> RiskLevel:
    """One-step escalation: Low->Medium->High. High stays High."""
    order = ["Low", "Medium", "High"]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]


if __name__ == "__main__":
    # Quick sanity check with a few manually-constructed "obvious" scenarios.
    scenarios = [
        dict(soil_moisture=15, rainfall_mm=300, slope=10, ndvi=0.8, landslide_density=0.2),
        dict(soil_moisture=70, rainfall_mm=3200, slope=42, ndvi=0.2, landslide_density=6.5),
        dict(soil_moisture=45, rainfall_mm=1500, slope=25, ndvi=0.5, landslide_density=2.0),
    ]
    labels = ["Expected LOW", "Expected HIGH", "Expected MEDIUM-ish"]
    for label, s in zip(labels, scenarios):
        score, level = predict_risk(**s)
        print(f"{label}: score={score}, level={level}  (inputs={s})")
