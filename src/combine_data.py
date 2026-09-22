import pandas as pd
import glob
import os


def combine_epss_data():

    files = glob.glob("data/epss/epss_*.csv")

    print(f"Found {len(files)} files")

    dataframes = []

    for file in files:

        print(f"Reading: {file}")

        # Read only the required EPSS columns
        df = pd.read_csv(
            file,
            usecols=["cve", "epss", "percentile"]
        )

        filename = os.path.basename(file)
        date = filename.replace("epss_", "").replace(".csv", "")

        df["snapshot_date"] = date

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Remove invalid CVE records
    combined_df = combined_df[
        combined_df["cve"].astype(str).str.startswith("CVE-")
    ]

    # Remove missing values
    combined_df = combined_df.dropna(
        subset=["cve", "epss", "percentile"]
    )

    # Remove duplicate CVE/date combinations
    combined_df = combined_df.drop_duplicates(
        subset=["cve", "snapshot_date"]
    )

    output_file = "data/epss_combined.csv"

    combined_df.to_csv(
        output_file,
        index=False
    )

    print("\nCombined dataset created successfully.")
    print(f"Total records: {len(combined_df)}")
    print(f"Columns: {combined_df.columns.tolist()}")


if __name__ == "__main__":
    combine_epss_data()