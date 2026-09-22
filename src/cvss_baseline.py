import pandas as pd

INPUT_FILE = "data/vulnerability_dataset.csv"
OUTPUT_FILE = "results/cvss_baseline.csv"


def create_baseline():

    df = pd.read_csv(INPUT_FILE)

    # Rank vulnerabilities using CVSS only
    df["cvss_priority_score"] = df["cvss_score"]

    # Create priority category
    def classify(score):
        if score >= 9.0:
            return "Critical"
        elif score >= 7.0:
            return "High"
        elif score >= 4.0:
            return "Medium"
        else:
            return "Low"

    df["cvss_priority"] = df["cvss_score"].apply(classify)

    # Highest CVSS first
    df = df.sort_values(
        by="cvss_priority_score",
        ascending=False
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("CVSS baseline created.")
    print(f"Records: {len(df)}")
    print("\nPriority distribution:")
    print(df["cvss_priority"].value_counts())

    print("\nTop 10 vulnerabilities:")
    print(
        df[
            ["cve", "cvss_score", "epss", "cvss_priority"]
        ].head(10)
    )


if __name__ == "__main__":
    create_baseline()