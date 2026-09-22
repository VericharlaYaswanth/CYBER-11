import pandas as pd
import os

BASE_URL = "https://raw.githubusercontent.com/empiricalsec/epss_scores/main"

DATES = [
    "2021-12-31",
    "2022-12-31",
    "2023-12-31",
    "2024-12-31",
    "2025-12-31"
]

def download_epss_data():

    output_dir = "data/epss"
    os.makedirs(output_dir, exist_ok=True)

    for date in DATES:

        year = date[:4]
        url = f"{BASE_URL}/{year}/epss_scores-{date}.csv.gz"
        output_file = f"{output_dir}/epss_{date}.csv"

        print(f"Downloading {date}...")

        try:
            df = pd.read_csv(
                url,
                compression="gzip",
                comment="#"
            )

            df = df[["cve", "epss", "percentile"]]

            df.to_csv(output_file, index=False)

            print(f"Saved: {output_file}")
            print(f"Records: {len(df)}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    download_epss_data()