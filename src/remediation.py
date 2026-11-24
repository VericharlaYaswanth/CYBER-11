from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "vulnerability_explanations.csv"
OUTPUT_FILE = ROOT / "results" / "remediation_recommendations.csv"


def generate_recommendation(row):
    priority = str(row.get("priority", "Low")).strip()
    cvss = float(row.get("cvss_score", 0) or 0)
    epss = float(row.get("epss", 0) or 0)
    ml_probability = float(row.get("ml_probability", 0) or 0)

    exploited = row.get("exploited", 0)

    try:
        exploited = int(exploited)
    except (ValueError, TypeError):
        exploited = 0

    # Known exploitation
    if exploited == 1:
        return (
            "URGENT: Confirm affected assets immediately, apply vendor patches "
            "or mitigations, isolate vulnerable systems if necessary, and "
            "perform security monitoring for potential compromise."
        )

    # Critical priority
    if priority == "Critical":
        return (
            "Immediate remediation recommended. Apply the vendor security patch, "
            "verify affected assets, restrict exposure, and prioritize deployment."
        )

    # High priority
    if priority == "High":
        return (
            "Prioritize remediation. Apply available patches promptly, review "
            "network exposure, and implement temporary security controls if "
            "patching is delayed."
        )

    # Medium priority
    if priority == "Medium":
        return (
            "Schedule remediation within the organization's patching cycle. "
            "Review affected systems and apply vendor updates or mitigations."
        )

    # Low priority
    return (
        "Monitor vulnerability status, review applicability to deployed assets, "
        "and apply patches during routine maintenance."
    )


def generate_action(row):
    priority = str(row.get("priority", "Low"))

    if priority == "Critical":
        return "Immediate"
    elif priority == "High":
        return "Urgent"
    elif priority == "Medium":
        return "Scheduled"
    else:
        return "Routine"


def main():
    if not INPUT_FILE.exists():
        print("ERROR: Explanation dataset not found.")
        print(f"Expected file: {INPUT_FILE}")
        print("Run explainability.py first.")
        return

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "cve",
        "priority",
        "cvss_score",
        "epss",
        "ml_probability",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing columns:", missing_columns)
        return

    df["remediation_action"] = df.apply(
        generate_recommendation, axis=1
    )

    df["remediation_urgency"] = df.apply(
        generate_action, axis=1
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("REMEDIATION RECOMMENDATIONS GENERATED")
    print("=" * 60)

    print(f"Total vulnerabilities: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nUrgency distribution:")
    print(df["remediation_urgency"].value_counts())

    print("\nSample recommendations:")
    print(
        df[
            [
                "cve",
                "priority",
                "remediation_urgency",
                "remediation_action",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()