import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "ml_features.csv"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


FEATURES = [
    "year",
    "month",
    "epss",
    "percentile",
    "epss_logit",
    "cvss_score",
    "cvss_version_num",
    "cvss_missing"
]


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(y_test, probabilities),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    print("\n" + "=" * 60)
    print(f"MODEL RESULTS: {name}")
    print("=" * 60)

    for metric, value in metrics.items():

        if metric != "Model":

            print(f"{metric}: {value:.6f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(y_test, predictions)

    print(cm)

    return metrics, predictions, probabilities, cm


def train_model():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Dataset records:", len(df))

    # --------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df["exploited"] = pd.to_numeric(
        df["exploited"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["year", "exploited"]
    )

    df = df[df["exploited"].isin([0, 1])]

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    X = df[FEATURES].copy()

    y = df["exploited"].astype(int)

    # --------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------

    train_mask = df["year"] <= 2023

    test_mask = df["year"] >= 2024

    X_train = X.loc[train_mask]

    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]

    y_test = y.loc[test_mask]

    print("\nTemporal Split")
    print("-" * 40)

    print("Training records:", len(X_train))

    print("Testing records:", len(X_test))

    print("\nTraining years:")
    print(df.loc[train_mask, "year"].value_counts().sort_index())

    print("\nTesting years:")
    print(df.loc[test_mask, "year"].value_counts().sort_index())

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    if len(X_train) == 0 or len(X_test) == 0:

        raise ValueError(
            "Temporal split produced an empty train or test dataset."
        )

    if y_train.nunique() < 2:

        raise ValueError(
            "Training dataset contains only one target class."
        )

    if y_test.nunique() < 2:

        raise ValueError(
            "Testing dataset contains only one target class."
        )

    # --------------------------------------------------
    # MODEL 1: LOGISTIC REGRESSION
    # --------------------------------------------------

    logistic_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )
    ])

    # --------------------------------------------------
    # MODEL 2: HISTOGRAM GRADIENT BOOSTING
    # --------------------------------------------------

    boosting_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_leaf_nodes=31,
                random_state=42
            )
        )
    ])

    models = {
        "Logistic Regression": logistic_model,
        "HistGradientBoosting": boosting_model
    }

    results = []

    model_objects = {}

    # --------------------------------------------------
    # TRAIN AND EVALUATE
    # --------------------------------------------------

    for name, model in models.items():

        print("\n")
        print("=" * 60)
        print("TRAINING:", name)
        print("=" * 60)

        model.fit(X_train, y_train)

        metrics, predictions, probabilities, cm = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(metrics)

        model_objects[name] = model

        # Save confusion matrix
        cm_file = RESULTS_DIR / (
            name.lower().replace(" ", "_")
            + "_confusion_matrix.csv"
        )

        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        ).to_csv(cm_file)

        # Save predictions
        prediction_file = RESULTS_DIR / (
            name.lower().replace(" ", "_")
            + "_predictions.csv"
        )

        prediction_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted": predictions,
            "probability": probabilities
        })

        prediction_df.to_csv(
            prediction_file,
            index=False
        )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(comparison.to_string(index=False))

    comparison_file = RESULTS_DIR / "model_comparison.csv"

    comparison.to_csv(
        comparison_file,
        index=False
    )

    # --------------------------------------------------
    # SELECT BEST MODEL BASED ON PR-AUC
    # --------------------------------------------------

    best_model_name = comparison.loc[
        comparison["PR_AUC"].idxmax(),
        "Model"
    ]

    best_model = model_objects[best_model_name]

    print("\nBest model based on PR-AUC:")
    print(best_model_name)

    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    best_model_file = MODEL_DIR / "exploit_model.pkl"

    joblib.dump(
        best_model,
        best_model_file
    )

    print("\nBest model saved:")
    print(best_model_file)

    # --------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------

    metrics_json = RESULTS_DIR / "model_metrics.json"

    with open(
        metrics_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "best_model": best_model_name,
                "results": results
            },
            file,
            indent=4
        )

    print("\nMetrics saved:")
    print(metrics_json)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    train_model()