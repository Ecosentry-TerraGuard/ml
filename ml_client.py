# backend/app/ml_client.py
from predict import predict_risk as _real_predict_risk, check_tilt_escalation
from app.models import RiskLevel

_LEVEL_MAP = {"Low": RiskLevel.LOW, "Medium": RiskLevel.MEDIUM, "High": RiskLevel.HIGH}

def predict_risk(soil_moisture, rainfall_mm, slope, ndvi, landslide_density):
    score_0_100, level_str = _real_predict_risk(
        soil_moisture=soil_moisture, rainfall_mm=rainfall_mm,
        slope=slope, ndvi=ndvi, landslide_density=landslide_density,
    )
    return round(score_0_100 / 100, 3), _LEVEL_MAP[level_str]
