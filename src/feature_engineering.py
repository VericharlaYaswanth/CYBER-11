import numpy as np
import pandas as pd

INPUT_FILE = "data/ml_dataset.csv"
OUTPUT_FILE = "data/ml_features.csv"


def create_features():

    print("Loading ML dataset...")

    df = pd.read_csv(INPUT_FILE)

    # -----------------------------------------
    # DATE FEATURES
    # -----------------------------------------

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    df["year"] = df["snapshot_date"].dt.year
    df["month"] = df["snapshot_date"].dt.month

    # -----------------------------------------
    # EPSS FEATURES
    # -----------------------------------------

    df["epss"] = pd.to_numeric(
        df["epss"],
        errors="coerce"
    )

    df["percentile"] = pd.to_numeric(
        df["percentile"],
        errors="coerce"
    )

    # Avoid log(0) problems
    epss_safe = df["epss"].clip(
        lower=0.000001,
        upper=0.999999
    )

    # Logit transformation of EPSS
    df["epss_logit"] = np.log(
        epss_safe /
        (1 - epss_safe)
    )

    # -----------------------------------------
    # CVSS FEATURES
    # -----------------------------------------

    df["cvss_score"] = pd.to_numeric(
        df["cvss_score"],
        errors="coerce"
    )

    df["cvss_normalized"] = (
        df["cvss_score"] / 10.0
    )

    df["cvss_version_num"] = pd.to_numeric(
        df["cvss_version"],
        errors="coerce"
    )

    # -----------------------------------------
    # CVSS SEVERITY
    # -----------------------------------------

    df["cvss_severity"] = pd.cut(
        df["cvss_score"],
        bins=[
            -1,
            3.9,
            6.9,
            8.9,
            10
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    )

    # -----------------------------------------
    # SELECT FEATURES
    # -----------------------------------------

    features = df[
        [
            "cve",
            "snapshot_date",
            "year",
            "month",
            "epss",
            "percentile",
            "epss_logit",
            "cvss_score",
            "cvss_normalized",
            "cvss_version_num",
            "cvss_severity",
            "exploited"
        ]
    ].copy()

    # -----------------------------------------
    # REMOVE INVALID ROWS
    # -----------------------------------------

    features = features.dropna(
        subset=[
            "epss",
            "percentile",
            "cvss_score"
        ]
    )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature dataset created.")

    print(
        f"Records: {len(features)}"
    )

    print("\nFeatures:")
    print(
        features.columns.tolist()
    )

    print("\nTarget distribution:")
    print(
        features["exploited"]
        .value_counts()
    )


if __name__ == "__main__":
    create_features()