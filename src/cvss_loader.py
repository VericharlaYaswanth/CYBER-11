import requests
import gzip
import json
import os
import pandas as pd
import time

BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0"

YEARS = [2021, 2022, 2023, 2024, 2025]


def download_file(url, retries=5):

    for attempt in range(1, retries + 1):

        try:
            print(f"Attempt {attempt}...")

            response = requests.get(
                url,
                timeout=120
            )

            response.raise_for_status()

            return response.content

        except Exception as e:

            print(f"Download failed: {e}")

            if attempt < retries:
                print("Retrying...")
                time.sleep(5)

    return None


def download_cvss_data():

    os.makedirs("data/cvss", exist_ok=True)

    all_records = []

    for year in YEARS:

        url = f"{BASE_URL}/nvdcve-2.0-{year}.json.gz"

        print(f"\nDownloading NVD {year}...")

        data = download_file(url)

        if data is None:
            print(f"Skipping {year}")
            continue

        nvd_data = json.loads(
            gzip.decompress(data)
        )

        for item in nvd_data["vulnerabilities"]:

            cve = item["cve"]
            cve_id = cve["id"]

            metrics = cve.get("metrics", {})

            metric = None

            if "cvssMetricV31" in metrics:
                metric = metrics["cvssMetricV31"][0]

            elif "cvssMetricV30" in metrics:
                metric = metrics["cvssMetricV30"][0]

            elif "cvssMetricV2" in metrics:
                metric = metrics["cvssMetricV2"][0]

            if metric is None:
                continue

            cvss_data = metric["cvssData"]

            all_records.append({
                "cve": cve_id,
                "cvss_score": cvss_data.get("baseScore"),
                "cvss_version": cvss_data.get("version"),
                "attack_vector": cvss_data.get("attackVector"),
                "attack_complexity": cvss_data.get("attackComplexity"),
                "privileges_required": cvss_data.get("privilegesRequired"),
                "user_interaction": cvss_data.get("userInteraction"),
                "scope": cvss_data.get("scope"),
                "confidentiality": cvss_data.get("confidentialityImpact"),
                "integrity": cvss_data.get("integrityImpact"),
                "availability": cvss_data.get("availabilityImpact")
            })

        print(f"Processed {year}")

    df = pd.DataFrame(all_records)

    output = "data/cvss/cvss_data.csv"

    df.to_csv(output, index=False)

    print("\nCVSS dataset updated.")
    print(f"Records: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    download_cvss_data()