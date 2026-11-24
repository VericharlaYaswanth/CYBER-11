import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "ml_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "ml_features.csv"


def create_features():

    print("Loading ML dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded records: {len(df):,}")

    # --------------------------------------------------
    # 1. DATE FEATURES
    # --------------------------------------------------

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce"
    )

    df["year"] = df["snapshot_date"].dt.year
    df["month"] = df["snapshot_date"].dt.month

    # --------------------------------------------------
    # 2. EPSS FEATURES
    # --------------------------------------------------

    df["epss"] = pd.to_numeric(
        df["epss"],
        errors="coerce"
    )

    df["percentile"] = pd.to_numeric(
        df["percentile"],
        errors="coerce"
    )

    # Logit transformation of EPSS
    epss_clipped = df["epss"].clip(
        lower=0.000001,
        upper=0.999999
    )

    df["epss_logit"] = np.log(
        epss_clipped / (1 - epss_clipped)
    )

    # --------------------------------------------------
    # 3. CVSS FEATURES
    # --------------------------------------------------

    df["cvss_score"] = pd.to_numeric(
        df["cvss_score"],
        errors="coerce"
    )

    df["cvss_version"] = pd.to_numeric(
        df["cvss_version"],
        errors="coerce"
    )

    # CVSS missing indicator
    df["cvss_missing"] = (
        df["cvss_score"].isna()
    ).astype(int)

    # CVSS severity
    df["cvss_severity"] = pd.cut(
        df["cvss_score"],
        bins=[
            -0.01,
            0,
            3.9,
            6.9,
            8.9,
            10
        ],
        labels=[
            "None",
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    )

    # --------------------------------------------------
    # 4. CVSS VERSION FIX
    # --------------------------------------------------

    version_map = {
        2.0: 2.0,
        3.0: 3.0,
        3.1: 3.1,
        4.0: 4.0
    }

    df["cvss_version_num"] = (
        df["cvss_version"]
        .map(version_map)
    )

    # --------------------------------------------------
    # 5. TARGET VALIDATION
    # --------------------------------------------------

    df["exploited"] = pd.to_numeric(
        df["exploited"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 6. REMOVE INVALID TARGET RECORDS
    # --------------------------------------------------

    df = df.dropna(
        subset=[
            "exploited",
            "snapshot_date"
        ]
    )

    df = df[
        df["exploited"].isin([0, 1])
    ]

    # --------------------------------------------------
    # 7. SELECT FEATURES
    # --------------------------------------------------

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
            "cvss_severity",
            "cvss_version",
            "cvss_version_num",
            "cvss_missing",
            "exploited"
        ]
    ].copy()

    # --------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # 9. SUMMARY
    # --------------------------------------------------

    print("\nFeature dataset created successfully.")

    print(f"Records: {len(features):,}")

    print("\nFeatures:")
    print(features.columns.tolist())

    print("\nMissing values:")
    print(features.isnull().sum())

    print("\nTarget distribution:")
    print(features["exploited"].value_counts())

    print("\nTarget percentage:")
    print(
        features["exploited"]
        .value_counts(normalize=True)
        .mul(100)
        .round(4)
    )

    print("\nCVSS version distribution:")
    print(
        features["cvss_version_num"]
        .value_counts(dropna=False)
    )

    print("\nCVSS severity distribution:")
    print(
        features["cvss_severity"]
        .value_counts(dropna=False)
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_features()