import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "data/ml_features.csv"
MODEL_FILE = "models/exploit_model.pkl"
METRICS_FILE = "results/model_metrics.json"
PREDICTIONS_FILE = "results/model_predictions.csv"

FEATURES = [
    "year",
    "month",
    "epss",
    "percentile",
    "epss_logit",
    "cvss_score",
    "cvss_normalized",
    "cvss_version_num"
]


def evaluate(model, X, y):

    probability = model.predict_proba(X)[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    metrics = {
        "roc_auc": float(
            roc_auc_score(y, probability)
        ),
        "pr_auc": float(
            average_precision_score(y, probability)
        ),
        "precision": float(
            precision_score(
                y,
                prediction,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y,
                prediction,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y,
                prediction,
                zero_division=0
            )
        )
    }

    cm = confusion_matrix(y, prediction)

    metrics["confusion_matrix"] = cm.tolist()

    return metrics, probability, prediction


def create_priority_score(df):

    df = df.copy()

    df["priority_score"] = (
        0.60 *
        df["predicted_exploit_probability"]
        +
        0.25 *
        df["epss"]
        +
        0.15 *
        (
            df["cvss_score"] / 10.0
        )
    ) * 100

    df["priority"] = pd.cut(
        df["priority_score"],
        bins=[
            -1,
            25,
            50,
            75,
            101
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    )

    return df


def train_model():

    print("\nLoading feature dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    # ------------------------------------------------
    # TEMPORAL TRAIN / TEST SPLIT
    # ------------------------------------------------

    train_mask = df["year"] <= 2023
    test_mask = df["year"] >= 2024

    X_train = df.loc[
        train_mask,
        FEATURES
    ]

    y_train = df.loc[
        train_mask,
        "exploited"
    ]

    X_test = df.loc[
        test_mask,
        FEATURES
    ]

    y_test = df.loc[
        test_mask,
        "exploited"
    ]

    print(
        f"Training records: {len(X_train):,}"
    )

    print(
        f"Testing records:  {len(X_test):,}"
    )

    print(
        f"Training exploited rate: "
        f"{y_train.mean() * 100:.3f}%"
    )

    print(
        f"Testing exploited rate: "
        f"{y_test.mean() * 100:.3f}%"
    )

    # ------------------------------------------------
    # LOGISTIC REGRESSION BASELINE
    # ------------------------------------------------

    logistic = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            )
        )
    ])

    # ------------------------------------------------
    # GRADIENT BOOSTING MODEL
    # ------------------------------------------------

    gradient = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ])

    # ------------------------------------------------
    # HANDLE CLASS IMBALANCE
    # ------------------------------------------------

    positive = int(
        y_train.sum()
    )

    negative = int(
        len(y_train) - positive
    )

    positive_weight = (
        negative /
        max(positive, 1)
    )

    sample_weight = np.where(
        y_train.to_numpy() == 1,
        positive_weight,
        1.0
    )

    print(
        f"\nPositive training samples: "
        f"{positive:,}"
    )

    print(
        f"Negative training samples: "
        f"{negative:,}"
    )

    # ------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic.fit(
        X_train,
        y_train
    )

    print(
        "Training Gradient Boosting..."
    )

    gradient.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weight
    )

    # ------------------------------------------------
    # EVALUATION
    # ------------------------------------------------

    logistic_metrics, _, _ = evaluate(
        logistic,
        X_test,
        y_test
    )

    gradient_metrics, probabilities, predictions = evaluate(
        gradient,
        X_test,
        y_test
    )

    print("\n==============================")
    print("MODEL RESULTS")
    print("==============================")

    print("\nLogistic Regression")

    for key, value in logistic_metrics.items():

        if key != "confusion_matrix":
            print(
                f"{key}: {value:.4f}"
            )

    print(
        "\nGradient Boosting"
    )

    for key, value in gradient_metrics.items():

        if key != "confusion_matrix":
            print(
                f"{key}: {value:.4f}"
            )

    # ------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    joblib.dump(
        gradient,
        MODEL_FILE
    )

    # ------------------------------------------------
    # CREATE TEST PREDICTIONS
    # ------------------------------------------------

    test_output = df.loc[
        test_mask,
        [
            "cve",
            "snapshot_date",
            "epss",
            "percentile",
            "cvss_score",
            "exploited"
        ]
    ].copy()

    test_output[
        "predicted_exploit_probability"
    ] = probabilities

    test_output = create_priority_score(
        test_output
    )

    test_output = test_output.sort_values(
        "priority_score",
        ascending=False
    )

    test_output.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------

    metrics = {

        "train_years": "2021-2023",

        "test_years": "2024-2025",

        "train_records":
            int(len(X_train)),

        "test_records":
            int(len(X_test)),

        "positive_train":
            int(y_train.sum()),

        "positive_test":
            int(y_test.sum()),

        "logistic_baseline":
            logistic_metrics,

        "gradient_boosting":
            gradient_metrics,

        "selected_model":
            "HistGradientBoostingClassifier",

        "features":
            FEATURES,

        "priority_formula":
            "60% ML probability + "
            "25% EPSS + "
            "15% normalized CVSS"
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    # ------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "CYBER-11 MODEL TRAINING COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"\nROC-AUC: "
        f"{gradient_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{gradient_metrics['pr_auc']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{gradient_metrics['f1_score']:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print(
        f"\nModel saved to: "
        f"{MODEL_FILE}"
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    train_model()