from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "prioritized_vulnerabilities.csv"
OUTPUT_FILE = ROOT / "results" / "vulnerability_explanations.csv"

def generate_explanation(row):
    reasons = []

    ml_probability = row.get("ml_probability", 0)
    epss = row.get("epss", 0)
    cvss = row.get("cvss_score", 0)
    priority = row.get("priority", "Unknown")

    # ML explanation
    if ml_probability >= 0.70:
        reasons.append("Very high predicted exploitation likelihood")
    elif ml_probability >= 0.40:
        reasons.append("High predicted exploitation likelihood")
    elif ml_probability >= 0.20:
        reasons.append("Moderate predicted exploitation likelihood")
    else:
        reasons.append("Low predicted exploitation likelihood")

    # EPSS explanation
    if epss >= 0.70:
        reasons.append("Very high EPSS score")
    elif epss >= 0.40:
        reasons.append("High EPSS score")
    elif epss >= 0.10:
        reasons.append("Moderate EPSS score")
    else:
        reasons.append("Low EPSS score")

    # CVSS explanation
    if cvss >= 9.0:
        reasons.append("Critical CVSS severity")
    elif cvss >= 7.0:
        reasons.append("High CVSS severity")
    elif cvss >= 4.0:
        reasons.append("Medium CVSS severity")
    elif cvss > 0:
        reasons.append("Low CVSS severity")
    else:
        reasons.append("CVSS score unavailable")

    # Final priority explanation
    if priority == "Critical":
        reasons.append("Classified as Critical priority")
    elif priority == "High":
        reasons.append("Classified as High priority")
    elif priority == "Medium":
        reasons.append("Classified as Medium priority")
    else:
        reasons.append("Classified as Low priority")

    return "; ".join(reasons)

def main():
    if not INPUT_FILE.exists():
        print("ERROR: Prioritized vulnerability dataset not found.")
        print(f"Expected file: {INPUT_FILE}")
        print("Run composite_scoring.py first.")
        return

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "cve",
        "ml_probability",
        "epss",
        "cvss_score",
        "composite_score",
        "priority",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing required columns:", missing_columns)
        return

    df["explanation"] = df.apply(generate_explanation, axis=1)

    # Additional human-readable interpretation
    df["risk_summary"] = df.apply(
        lambda row: (
            f"{row['cve']} has a composite risk score of "
            f"{row['composite_score']:.2f}/100 and is categorized as "
            f"{row['priority']} priority. "
            f"{row['explanation']}."
        ),
        axis=1,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"Total vulnerabilities explained: {len(df)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nSample explanations:")
    print(
        df[["cve", "priority", "composite_score", "risk_summary"]]
        .head(10)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()
    