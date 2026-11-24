import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "EPSS": ROOT / "data" / "epss_combined.csv",
    "CVSS": ROOT / "data" / "cvss" / "cvss_data.csv",
    "Vulnerability Dataset": ROOT / "data" / "vulnerability_dataset.csv",
    "ML Dataset": ROOT / "data" / "ml_dataset.csv",
    "ML Features": ROOT / "data" / "ml_features.csv",
}

REPORT_DIR = ROOT / "results"


def check_file(name, path):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    if not path.exists():
        print("ERROR: File does not exist.")
        return

    df = pd.read_csv(path)

    print(f"File: {path.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    # Missing values
    print("\nMissing values:")
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values.")
    else:
        print(missing[missing > 0])

    # Duplicate rows
    print("\nDuplicate rows:", df.duplicated().sum())

    # CVE validation
    if "cve" in df.columns:

        invalid_cves = ~df["cve"].astype(str).str.match(
            r"^CVE-\d{4}-\d+$",
            na=False
        )

        print("Invalid CVE IDs:", invalid_cves.sum())

    # CVE and date duplicates
    if {"cve", "snapshot_date"}.issubset(df.columns):

        duplicate_pairs = df.duplicated(
            subset=["cve", "snapshot_date"]
        ).sum()

        print("Duplicate CVE/date combinations:", duplicate_pairs)

    # EPSS validation
    if "epss" in df.columns:

        print("\nEPSS range:")
        print(df["epss"].describe())

        invalid_epss = (
            (df["epss"] < 0) |
            (df["epss"] > 1)
        )

        print("Invalid EPSS values:", invalid_epss.sum())

    # EPSS percentile validation
    if "percentile" in df.columns:

        invalid_percentile = (
            (df["percentile"] < 0) |
            (df["percentile"] > 1)
        )

        print("Invalid percentile values:", invalid_percentile.sum())

    # CVSS validation
    if "cvss_score" in df.columns:

        print("\nCVSS range:")
        print(df["cvss_score"].describe())

        invalid_cvss = (
            (df["cvss_score"] < 0) |
            (df["cvss_score"] > 10)
        )

        print("Invalid CVSS values:", invalid_cvss.sum())

    # CVSS version
    if "cvss_version" in df.columns:

        print("\nCVSS versions:")
        print(df["cvss_version"].value_counts(dropna=False))

    # Date validation
    if "snapshot_date" in df.columns:

        dates = pd.to_datetime(
            df["snapshot_date"],
            errors="coerce"
        )

        print("\nDate range:")
        print("Earliest:", dates.min())
        print("Latest:", dates.max())

        print("Invalid dates:", dates.isna().sum())

        print("\nRecords by year:")
        print(dates.dt.year.value_counts().sort_index())

    # ML label validation
    if "exploited" in df.columns:

        print("\nTarget distribution:")
        print(df["exploited"].value_counts())

        invalid_labels = ~df["exploited"].isin([0, 1])

        print("Invalid labels:", invalid_labels.sum())

        print("\nClass percentage:")
        print(
            df["exploited"]
            .value_counts(normalize=True)
            .mul(100)
            .round(4)
        )

    # Feature validation
    if "cvss_version_num" in df.columns:

        print("\nCVSS numeric version feature:")

        print(df["cvss_version_num"].value_counts(dropna=False))

        print(
            "Missing cvss_version_num:",
            df["cvss_version_num"].isna().sum()
        )

    # Report
    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / (
        name.lower().replace(" ", "_") + "_validation.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:

        file.write(f"VALIDATION REPORT: {name}\n")
        file.write("=" * 70 + "\n")

        file.write(f"Rows: {len(df)}\n")
        file.write(f"Columns: {len(df.columns)}\n")

        file.write("\nColumn names:\n")
        file.write(str(df.columns.tolist()))

        file.write("\n\nMissing values:\n")
        file.write(str(df.isnull().sum()))

        file.write("\n\nDuplicate rows:\n")
        file.write(str(df.duplicated().sum()))

    print(f"\nValidation report saved: {report_path}")


def main():

    print("=" * 70)
    print("CYBERSECURITY VULNERABILITY DATA VALIDATION")
    print("=" * 70)

    for name, path in FILES.items():

        check_file(name, path)

    print("\n")
    print("=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    