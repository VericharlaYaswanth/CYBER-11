from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "ml_features.csv"
MODEL_FILE = ROOT / "models" / "exploit_model.pkl"
OUTPUT_FILE = ROOT / "results" / "prioritized_vulnerabilities.csv"

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


def classify_priority(score):

    if score >= 75:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 25:
        return "Medium"
    else:
        return "Low"


def main():

    print("=" * 60)
    print("COMPOSITE VULNERABILITY PRIORITIZATION")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    # Check required columns
    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    if "cve" not in df.columns:
        raise ValueError("CVE column is missing")

    # Load trained model
    model = joblib.load(MODEL_FILE)

    print("Dataset loaded:", len(df), "records")

    # Generate ML probability
    probabilities = model.predict_proba(
        df[FEATURES]
    )[:, 1]

    df["ml_probability"] = probabilities

    # Normalize scoring components to 0-100
    df["ml_score"] = df["ml_probability"] * 100

    df["epss_score"] = (
        df["epss"].fillna(0).clip(0, 1) * 100
    )

    df["cvss_normalized"] = (
        df["cvss_score"].fillna(0).clip(0, 10) * 10
    )

    # Composite score
    df["composite_score"] = (
        0.60 * df["ml_score"]
        + 0.25 * df["epss_score"]
        + 0.15 * df["cvss_normalized"]
    )

    # Priority classification
    df["priority"] = df["composite_score"].apply(
        classify_priority
    )

    # Keep latest snapshot for each CVE
    if "snapshot_date" in df.columns:

        df["snapshot_date"] = pd.to_datetime(
            df["snapshot_date"],
            errors="coerce"
        )

        df = df.sort_values("snapshot_date")

        df = df.drop_duplicates(
            subset="cve",
            keep="last"
        )

    # Highest priority first
    df = df.sort_values(
        "composite_score",
        ascending=False
    )

    # Output columns
    output_columns = [
        "cve",
        "snapshot_date",
        "epss",
        "percentile",
        "cvss_score",
        "ml_probability",
        "ml_score",
        "epss_score",
        "cvss_normalized",
        "composite_score",
        "priority"
    ]

    # Include exploitation label if available
    if "exploited" in df.columns:
        output_columns.append("exploited")

    result = df[output_columns]

    # Save results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Display summary
    print("\nUnique CVEs ranked:", len(result))

    print("\nPriority Distribution:")
    print(
        result["priority"].value_counts()
    )

    print("\nTop 10 Vulnerabilities:")
    print(
        result[
            [
                "cve",
                "composite_score",
                "priority",
                "ml_probability",
                "epss",
                "cvss_score"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nAverage Composite Score:")
    print(
        round(result["composite_score"].mean(), 2)
    )

    print("\nOutput saved:")
    print(OUTPUT_FILE)

    print("\nComposite scoring completed successfully!")


if __name__ == "__main__":
    main()
    