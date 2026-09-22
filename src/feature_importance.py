import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
import joblib


INPUT_FILE = "data/ml_features.csv"
MODEL_FILE = "models/exploit_model.pkl"
OUTPUT_FILE = "results/feature_importance.csv"
PLOT_FILE = "results/feature_importance.png"

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


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    model = joblib.load(MODEL_FILE)

    X = df[FEATURES]
    y = df["exploited"]

    # Use a sample for faster permutation analysis
    sample_size = min(10000, len(df))

    sample = df.sample(
        sample_size,
        random_state=42
    )

    X_sample = sample[FEATURES]
    y_sample = sample["exploited"]

    print(
        f"Calculating feature importance "
        f"using {sample_size} records..."
    )

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean":
            result.importances_mean,
        "importance_std":
            result.importances_std
    })

    importance = importance.sort_values(
        "importance_mean",
        ascending=False
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    importance.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------
    # PLOT
    # -----------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importance["feature"],
        importance["importance_mean"]
    )

    plt.xlabel(
        "Permutation Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "CYBER-11 Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=300
    )

    plt.close()

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )

    print(
        f"Saved: {PLOT_FILE}"
    )


if __name__ == "__main__":
    main()