import pandas as pd


EPSS_FILE = "data/epss_combined.csv"
CVSS_FILE = "data/cvss/cvss_data.csv"


def merge_datasets():

    print("Loading EPSS data...")
    epss = pd.read_csv(EPSS_FILE)

    print("Loading CVSS data...")
    cvss = pd.read_csv(CVSS_FILE)

    print(f"EPSS records: {len(epss)}")
    print(f"CVSS records: {len(cvss)}")

    # Merge using CVE ID
    merged = epss.merge(
        cvss,
        on="cve",
        how="left"
    )

    # Remove records without CVSS
    merged = merged.dropna(subset=["cvss_score"])

    # Remove duplicate CVE/date combinations
    merged = merged.drop_duplicates(
        subset=["cve", "snapshot_date"]
    )

    output = "data/vulnerability_dataset.csv"

    merged.to_csv(output, index=False)

    print("\nMerged dataset created.")
    print(f"Records: {len(merged)}")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    merge_datasets()