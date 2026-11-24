from pathlib import Path
import pandas as pd
import json

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "ml_features.csv"
OUTPUT_DIR = ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_epss_baseline():

    print("=" * 60)
    print("EPSS BASELINE EVALUATION")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    # Temporal test data
    test = df[df["year"] >= 2024].copy()

    y_true = test["exploited"]
    scores = test["epss"]

    # Default threshold
    predictions = (scores >= 0.5).astype(int)

    metrics = {
        "Model": "EPSS Baseline",
        "Accuracy": accuracy_score(y_true, predictions),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        )
    }

    print("\nTest records:", len(test))

    print("\nEPSS Baseline Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, predictions)

    print(cm)

    # Save results
    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "epss_baseline_metrics.csv",
        index=False
    )

    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ).to_csv(
        OUTPUT_DIR / "epss_baseline_confusion_matrix.csv"
    )

    test["epss_prediction"] = scores
    test["epss_predicted_label"] = predictions

    test[
        ["cve", "year", "epss", "exploited",
         "epss_prediction", "epss_predicted_label"]
    ].to_csv(
        OUTPUT_DIR / "epss_baseline_predictions.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "epss_baseline_metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nFiles saved successfully in results/")
    print("EPSS baseline evaluation completed!")


if __name__ == "__main__":
    evaluate_epss_baseline()