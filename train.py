"""
train.py — Trains the TerraGuard landslide risk Random Forest classifier.

Run: python train.py
Produces: model.pkl (the trained classifier), metrics.txt (evaluation summary),
          feature_importance.png (for the pitch/demo -- explainability matters
          for a life-safety system).
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from generate_dataset import generate_dataset

FEATURES = ["soil_moisture", "rainfall_mm", "slope", "ndvi", "landslide_density"]
TARGET = "risk_level"


def main():
    df = generate_dataset()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest chosen over deep learning deliberately:
    # - Interpretable (feature_importances_ below) -- critical for a life-safety
    #   system where judges/authorities need to trust *why* a risk level fired.
    # - Works well on small/proxy tabular datasets; deep learning needs far more
    #   labeled data than we have access to.
    # - No GPU dependency -- realistic for constrained NER field deployment.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
    # Recall matters more than raw accuracy here: missing a real High-risk event
    # (false negative) is far worse than a false alarm for a disaster-warning system.
    recall_high = recall_score(y_test, y_pred, labels=["High"], average="macro")

    with open("metrics.txt", "w") as f:
        f.write("TerraGuard Landslide Risk Model -- Evaluation (on PROXY test data)\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            "NOTE: trained and evaluated on synthetic proxy data (see README.md / "
            "generate_dataset.py) -- these numbers describe how well the model "
            "learned the proxy data's patterns, NOT real-world NER accuracy. "
            "Report honestly as 'proxy validation performance' if asked.\n\n"
        )
        f.write(report)
        f.write("\nConfusion matrix (rows=actual, cols=predicted) [Low, Medium, High]:\n")
        f.write(str(cm) + "\n")
        f.write(f"\nHigh-risk recall (most important metric for this use case): {recall_high:.3f}\n")

    print(report)
    print("High-risk recall:", recall_high)

    # Feature importance chart -- use this in the pitch/demo to show the model
    # is not a black box.
    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 4))
    importances.plot(kind="barh", color="#2E5F8A")
    plt.title("Feature Importance -- TerraGuard Risk Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("\nSaved feature_importance.png")

    joblib.dump(clf, "model.pkl")
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
