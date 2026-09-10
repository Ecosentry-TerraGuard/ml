"""
generate_dataset.py — Proxy training dataset for TerraGuard's landslide risk model.

WHY SYNTHETIC / PROXY DATA:
No official NER-specific landslide dataset with matching features (rainfall, soil
moisture, slope, NDVI, landslide density) was available in the time we had. Rather
than train on mismatched or overly generic data, we generate a synthetic dataset
whose feature ranges and risk relationships are grounded in real, published landslide
susceptibility research — including a study specifically on NH31A and East Sikkim
Himalaya settlements (Sikkim is adjacent to / bordering our target NER states), plus
broader Himalayan/Indian landslide literature. Sources and the specific numbers used
are cited inline below.

This is explicitly a PROXY dataset for demo purposes. The model trained on it is
designed to be retrained on real field data post-deployment -- see README.md.

CITED RANGES:
- Landslides most prevalent on slopes 35-50 degrees, at rainfall 2000-2500mm and
  3000-3300mm annually.
  Source: "Geospatial intelligence for landslide susceptibility and risk analysis:
  Insights from NH31A and east Sikkim Himalaya settlements" (ScienceDirect, 2024)
- Rainfall, slope, NDVI, and landslide density (via inventory-based factors like
  drainage/road density) are consistently the top features across multiple landslide
  susceptibility ML studies.
  Source: "Opening the Black-Box: A Systematic Review on Explainable AI in Remote
  Sensing" (arXiv:2402.13791)
- Higher NDVI (more vegetation) generally REDUCES landslide risk by root-binding soil,
  though the Nujiang Prefecture CF-SVM study notes this relationship can invert on
  very steep slopes where vegetation load itself adds stress. We model the dominant
  (protective) relationship, noting the inversion as a documented limitation.
  Source: "Evaluation of Landslide Susceptibility Based on CF-SVM in Nujiang
  Prefecture" (PMC9654940)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N_SAMPLES = 4000


def generate_dataset(n=N_SAMPLES) -> pd.DataFrame:
    # --- Feature generation ---
    # Rainfall (mm, monsoon-period cumulative): NER/Himalayan foothills range widely;
    # Cherrapunji/Mawsynram-adjacent areas see extreme values, so we skew right.
    rainfall_mm = RNG.gamma(shape=3.0, scale=600, size=n)  # roughly 200-4000mm range
    rainfall_mm = np.clip(rainfall_mm, 50, 5000)

    # Soil moisture (%): correlated loosely with rainfall + noise
    soil_moisture = np.clip(
        20 + 0.012 * rainfall_mm + RNG.normal(0, 8, n), 5, 100
    )

    # Slope (degrees): 0-70, landslide literature flags 35-50 as highest risk band
    slope = np.clip(RNG.normal(32, 14, n), 2, 70)

    # NDVI (-1 to 1): most vegetated hill terrain sits 0.2-0.8
    ndvi = np.clip(RNG.normal(0.5, 0.2, n), -0.2, 0.95)

    # Landslide density (historical events per sq km, proxy scale 0-10)
    landslide_density = np.clip(RNG.gamma(shape=1.5, scale=1.3, size=n), 0, 10)

    df = pd.DataFrame({
        "rainfall_mm": rainfall_mm,
        "soil_moisture": soil_moisture,
        "slope": slope,
        "ndvi": ndvi,
        "landslide_density": landslide_density,
    })

    # --- Physically-motivated risk score (0-100) ---
    # Weights reflect literature consensus: rainfall and slope dominate, soil moisture
    # and landslide density contribute strongly, NDVI is protective (subtracted).
    # This is our documented, inspectable ground-truth generation rule -- not a
    # black box -- so it can be scrutinized and replaced with real labels later.
    risk_score = (
        0.32 * _norm(df["rainfall_mm"], 0, 5000)
        + 0.24 * _norm(df["slope"], 0, 70)
        + 0.20 * _norm(df["soil_moisture"], 0, 100)
        + 0.14 * _norm(df["landslide_density"], 0, 10)
        - 0.10 * _norm(df["ndvi"], -0.2, 0.95)  # more vegetation -> lower risk
    ) * 100

    # Slope band 35-50 degrees gets an extra bump (per Sikkim Himalaya study)
    slope_band_bonus = np.where((df["slope"] >= 35) & (df["slope"] <= 50), 8, 0)
    # Rainfall bands 2000-2500 / 3000-3300mm get an extra bump (per same study)
    rain_band_bonus = np.where(
        ((df["rainfall_mm"] >= 2000) & (df["rainfall_mm"] <= 2500))
        | ((df["rainfall_mm"] >= 3000) & (df["rainfall_mm"] <= 3300)),
        6, 0,
    )

    risk_score = risk_score + slope_band_bonus + rain_band_bonus
    risk_score = risk_score + RNG.normal(0, 5, n)  # measurement/model noise
    risk_score = np.clip(risk_score, 0, 100)

    df["risk_score"] = risk_score
    # Quantile-based bins (rather than fixed 0-33-66-100 cutoffs) so the training
    # set has a usable, roughly balanced spread of Low/Medium/High examples for a
    # classifier -- with our generation formula, fixed cutoffs skewed heavily Low.
    df["risk_level"] = pd.qcut(
        risk_score,
        q=[0, 0.45, 0.80, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return df


def _norm(series, lo, hi):
    return (series - lo) / (hi - lo)


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("proxy_landslide_dataset.csv", index=False)
    print(f"Generated {len(data)} rows -> proxy_landslide_dataset.csv")
    print(data["risk_level"].value_counts())
    print(data.describe())
