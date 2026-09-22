import pandas as pd

VULN_FILE = "data/vulnerability_dataset.csv"
KEV_FILE = "data/kev.csv"
OUTPUT_FILE = "data/ml_dataset.csv"


def prepare_ml_dataset():

    print("Loading vulnerability data...")
    vuln = pd.read_csv(VULN_FILE)

    print("Loading KEV data...")
    kev = pd.read_csv(KEV_FILE)

    # Convert dates
    vuln["snapshot_date"] = pd.to_datetime(
        vuln["snapshot_date"]
    )

    kev["dateAdded"] = pd.to_datetime(
        kev["dateAdded"]
    )

    # Keep required KEV information
    kev = kev[["cveID", "dateAdded"]].drop_duplicates()

    # Merge CVEs
    merged = vuln.merge(
        kev,
        left_on="cve",
        right_on="cveID",
        how="left"
    )

    # A CVE is considered exploited only if
    # it was added to KEV on or before the snapshot date
    merged["exploited"] = (
        merged["dateAdded"].notna()
        & (
            merged["dateAdded"]
            <= merged["snapshot_date"]
        )
    ).astype(int)

    # Remove temporary column
    merged = merged.drop(columns=["cveID", "dateAdded"])

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nTime-aware ML dataset created.")
    print(f"Records: {len(merged)}")

    print("\nExploitation label:")
    print(merged["exploited"].value_counts())

    print("\nExploitation percentage:")
    print(merged["exploited"].mean() * 100)

    print("\nExploited by snapshot:")
    print(
        merged.groupby("snapshot_date")["exploited"]
        .sum()
    )


if __name__ == "__main__":
    prepare_ml_dataset()