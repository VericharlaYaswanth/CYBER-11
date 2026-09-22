import json
import os

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CYBER-11 | Vulnerability Prioritization",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA = os.path.join(
    BASE,
    "data",
    "ml_features.csv"
)

MODEL = os.path.join(
    BASE,
    "models",
    "exploit_model.pkl"
)

METRICS = os.path.join(
    BASE,
    "results",
    "model_metrics.json"
)

IMPORTANCE = os.path.join(
    BASE,
    "results",
    "feature_importance.csv"
)


FEATURES = [
    "year",
    "month",
    "epss",
    "percentile",
    "epss_logit",
    "cvss_score",
    "cvss_normalized",
    "cvss_version_num"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL)


# ============================================================
# CALCULATE SCORES
# ============================================================

def build_scores(df):

    model = load_model()

    df = df.copy()

    df["predicted_exploit_probability"] = (
        model.predict_proba(
            df[FEATURES]
        )[:, 1]
    )

    df["priority_score"] = (

        0.60 *
        df["predicted_exploit_probability"]

        +

        0.25 *
        df["epss"]

        +

        0.15 *
        (
            df["cvss_score"] / 10
        )

    ) * 100

    df["priority"] = pd.cut(

        df["priority_score"],

        bins=[
            -1,
            25,
            50,
            75,
            101
        ],

        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ]
    )

    return df


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(DATA):

    st.error(
        "ml_features.csv is missing."
    )

    st.stop()


if not os.path.exists(MODEL):

    st.error(
        "exploit_model.pkl is missing."
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

df = build_scores(
    load_data()
)

latest_date = df[
    "snapshot_date"
].max()

latest = df[
    df["snapshot_date"] == latest_date
].copy()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ CYBER-11"
)

st.subheader(
    "Vulnerability Prioritization "
    "with Exploit-Likelihood Prediction"
)

st.caption(
    "EPSS + CVSS + Machine Learning "
    "| Temporal evaluation on 2024–2025"
)

st.markdown(
    f"**Latest snapshot:** "
    f"{latest_date.date()}"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Vulnerability Filters"
)

search_cve = st.sidebar.text_input(
    "Search CVE",
    placeholder="Example: CVE-2024-1234"
)

min_epss = st.sidebar.slider(
    "Minimum EPSS",
    0.0,
    1.0,
    0.0,
    0.01
)

min_cvss = st.sidebar.slider(
    "Minimum CVSS",
    0.0,
    10.0,
    0.0,
    0.5
)

priority_filter = st.sidebar.multiselect(

    "Priority",

    [
        "Critical",
        "High",
        "Medium",
        "Low"
    ],

    default=[
        "Critical",
        "High",
        "Medium",
        "Low"
    ]
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = latest[
    (latest["epss"] >= min_epss)
    &
    (latest["cvss_score"] >= min_cvss)
    &
    (
        latest["priority"]
        .astype(str)
        .isin(priority_filter)
    )
].copy()


if search_cve:

    search_results = latest[
        latest["cve"]
        .astype(str)
        .str.contains(
            search_cve,
            case=False,
            na=False
        )
    ].copy()

    filtered = search_results


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    "## 📊 Vulnerability Overview"
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Vulnerabilities",
    f"{len(filtered):,}"
)

c2.metric(
    "Critical",
    f"{(filtered['priority'] == 'Critical').sum():,}"
)

c3.metric(
    "High",
    f"{(filtered['priority'] == 'High').sum():,}"
)

c4.metric(
    "Avg EPSS",
    (
        f"{filtered['epss'].mean():.3f}"
        if len(filtered)
        else "0"
    )
)

c5.metric(
    "Avg CVSS",
    (
        f"{filtered['cvss_score'].mean():.2f}"
        if len(filtered)
        else "0"
    )
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dashboard",
        "🔍 Vulnerability Search",
        "🤖 Model Analysis",
        "ℹ️ About"
    ]
)


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    left, right = st.columns(2)

    # --------------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### Priority Distribution"
        )

        priority_counts = (
            filtered["priority"]
            .value_counts()
            .reindex(
                [
                    "Critical",
                    "High",
                    "Medium",
                    "Low"
                ]
            )
            .fillna(0)
        )

        st.bar_chart(
            priority_counts
        )


    # --------------------------------------------------------
    # EPSS VS CVSS
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### EPSS vs CVSS"
        )

        if len(filtered):

            chart = filtered[
                [
                    "epss",
                    "cvss_score"
                ]
            ].rename(

                columns={
                    "epss": "EPSS",
                    "cvss_score": "CVSS"
                }

            )

            st.scatter_chart(
                chart,
                x="EPSS",
                y="CVSS"
            )


    # --------------------------------------------------------
    # TOP VULNERABILITIES
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Top Vulnerabilities "
        "Requiring Attention"
    )

    show_cols = [
        "cve",
        "epss",
        "percentile",
        "cvss_score",
        "predicted_exploit_probability",
        "priority_score",
        "priority"
    ]

    top = (
        filtered
        .sort_values(
            "priority_score",
            ascending=False
        )
        .head(50)
        .copy()
    )

    top[
        "predicted_exploit_probability"
    ] = (
        top[
            "predicted_exploit_probability"
        ] * 100
    ).round(2)

    top[
        "priority_score"
    ] = top[
        "priority_score"
    ].round(2)

    st.dataframe(
        top[show_cols],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Vulnerabilities",
        data=csv_data,
        file_name="cyber11_filtered_vulnerabilities.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 2 — VULNERABILITY SEARCH
# ============================================================

with tab2:

    st.markdown(
        "## 🔍 Vulnerability Details"
    )

    if search_cve and len(filtered):

        selected_cve = st.selectbox(
            "Select CVE",
            filtered["cve"].tolist()
        )

        vulnerability = filtered[
            filtered["cve"] == selected_cve
        ].iloc[0]

        st.markdown(
            f"### {selected_cve}"
        )

        a, b, c = st.columns(3)

        a.metric(
            "EPSS",
            f"{vulnerability['epss']:.4f}"
        )

        b.metric(
            "CVSS",
            f"{vulnerability['cvss_score']:.1f}"
        )

        c.metric(
            "Priority Score",
            f"{vulnerability['priority_score']:.2f}"
        )

        st.markdown(
            "### Exploit Likelihood"
        )

        st.progress(
            float(
                vulnerability[
                    "predicted_exploit_probability"
                ]
            )
        )

        st.write(
            f"Predicted exploit probability: "
            f"**{vulnerability['predicted_exploit_probability'] * 100:.2f}%**"
        )

        st.write(
            f"Priority classification: "
            f"**{vulnerability['priority']}**"
        )

        st.markdown(
            "### Risk Components"
        )

        details = pd.DataFrame({

            "Component": [
                "ML Exploit Likelihood",
                "EPSS",
                "CVSS"
            ],

            "Value": [
                vulnerability[
                    "predicted_exploit_probability"
                ] * 100,

                vulnerability[
                    "epss"
                ] * 100,

                vulnerability[
                    "cvss_score"
                ] * 10
            ]
        })

        st.bar_chart(
            details.set_index(
                "Component"
            )
        )

    elif search_cve:

        st.warning(
            "No matching CVE found."
        )

    else:

        st.info(
            "Enter a CVE in the sidebar "
            "to view detailed vulnerability information."
        )


# ============================================================
# TAB 3 — MODEL ANALYSIS
# ============================================================

with tab3:

    st.markdown(
        "## 🤖 Machine Learning Analysis"
    )

    if os.path.exists(METRICS):

        with open(
            METRICS,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(file)

        model_metrics = metrics[
            "gradient_boosting"
        ]

        a, b, c, d = st.columns(4)

        a.metric(
            "ROC-AUC",
            f"{model_metrics['roc_auc']:.4f}"
        )

        b.metric(
            "PR-AUC",
            f"{model_metrics['pr_auc']:.4f}"
        )

        c.metric(
            "Precision",
            f"{model_metrics['precision']:.4f}"
        )

        d.metric(
            "Recall",
            f"{model_metrics['recall']:.4f}"
        )

        st.markdown(
            "### Model Configuration"
        )

        st.write(
            "Training period: **2021–2023**"
        )

        st.write(
            "Testing period: **2024–2025**"
        )

        st.write(
            f"Training records: "
            f"**{metrics['train_records']:,}**"
        )

        st.write(
            f"Testing records: "
            f"**{metrics['test_records']:,}**"
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🔬 Feature Importance"
    )

    if os.path.exists(IMPORTANCE):

        importance = pd.read_csv(
            IMPORTANCE
        )

        importance = importance.sort_values(
            "importance_mean",
            ascending=False
        )

        st.bar_chart(
            importance.set_index(
                "feature"
            )[
                "importance_mean"
            ]
        )

        st.dataframe(
            importance,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Feature importance file not found. "
            "Run feature_importance.py first."
        )


    # --------------------------------------------------------
    # PRIORITY FORMULA
    # --------------------------------------------------------

    st.markdown(
        "### Priority Score Formula"
    )

    st.code(
        """
Priority Score =
    60% × ML Exploit Probability
  + 25% × EPSS
  + 15% × Normalized CVSS
        """
    )


# ============================================================
# TAB 4 — ABOUT
# ============================================================

with tab4:

    st.markdown(
        "## 🛡️ About CYBER-11"
    )

    st.write(
        """
CYBER-11 is a vulnerability prioritization system
that combines vulnerability severity and exploit
likelihood to help security teams identify
vulnerabilities that require attention.
"""
    )

    st.markdown(
        "### Data Sources"
    )

    st.write(
        "- EPSS vulnerability exploitation probability"
    )

    st.write(
        "- CVSS vulnerability severity"
    )

    st.write(
        "- CISA KEV exploitation observations"
    )

    st.markdown(
        "### System Pipeline"
    )

    st.code(
        """
EPSS + CVSS + Historical Data
              ↓
       Feature Engineering
              ↓
     Machine Learning Model
              ↓
   Exploit-Likelihood Prediction
              ↓
       Priority Score
              ↓
      Risk Classification
              ↓
         Dashboard
        """
    )

    st.markdown(
        "### Priority Levels"
    )

    st.write(
        "**Critical:** Priority score > 75"
    )

    st.write(
        "**High:** Priority score 50–75"
    )

    st.write(
        "**Medium:** Priority score 25–50"
    )

    st.write(
        "**Low:** Priority score ≤ 25"
    )