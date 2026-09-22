import pandas as pd

URL = "https://raw.githubusercontent.com/cisagov/kev-data/main/known_exploited_vulnerabilities.csv"

OUTPUT = "data/kev.csv"


def download_kev():

    print("Downloading CISA KEV data...")

    df = pd.read_csv(URL)

    print(f"Records: {len(df)}")
    print("Columns:")
    print(df.columns.tolist())

    df.to_csv(OUTPUT, index=False)

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    download_kev()