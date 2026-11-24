import sys
import os
import json
import hashlib
import secrets

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# AUTHENTICATION IMPORT
# ============================================================

from auth import login, logout, get_role


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CYBER-11 | Vulnerability Prioritization",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# AUTHENTICATION
# ============================================================

if not login():
    st.stop()

current_role = get_role()

st.sidebar.markdown("---")

st.sidebar.write(
    f"👤 User: {st.session_state.get('username', 'Unknown')}"
)

st.sidebar.write(
    f"🔐 Role: {current_role}"
)

if st.sidebar.button("Logout"):

    logout()
    st.rerun()


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = PROJECT_ROOT

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

EXPLANATIONS = os.path.join(
    BASE,
    "results",
    "vulnerability_explanations.csv"
)

RECOMMENDATIONS = os.path.join(
    BASE,
    "results",
    "remediation_recommendations.csv"
)

PRIORITIZED = os.path.join(
    BASE,
    "results",
    "prioritized_vulnerabilities.csv"
)

AUTH_FILE = os.path.join(
    BASE,
    "auth_config.json"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "year",
    "month",
    "epss",
    "percentile",
    "epss_logit",
    "cvss_score",
    "cvss_version_num",
    "cvss_missing"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_number(value, default=0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA)

    if "snapshot_date" in df.columns:

        df["snapshot_date"] = pd.to_datetime(
            df["snapshot_date"],
            errors="coerce"
        )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL)


# ============================================================
# LOAD OPTIONAL FILES
# ============================================================

@st.cache_data
def load_optional_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception:

        return pd.DataFrame()


# ============================================================
# BUILD PRIORITY SCORES
# ============================================================

def build_scores(df):

    df = df.copy()

    model = load_model()

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    model_input = df[FEATURES].copy()

    model_input = model_input.fillna(0)

    # --------------------------------------------------------
    # ML EXPLOIT PROBABILITY
    # --------------------------------------------------------

    df["predicted_exploit_probability"] = (
        model.predict_proba(model_input)[:, 1]
    )

    # --------------------------------------------------------
    # ML SCORE
    # --------------------------------------------------------

    df["ml_score"] = (
        df["predicted_exploit_probability"] * 100
    )

    # --------------------------------------------------------
    # EPSS SCORE
    # --------------------------------------------------------

    df["epss_score"] = (

        pd.to_numeric(
            df["epss"],
            errors="coerce"
        )

        .fillna(0)

        .clip(0, 1)

        * 100
    )

    # --------------------------------------------------------
    # CVSS NORMALIZATION
    # --------------------------------------------------------

    df["cvss_normalized"] = (

        pd.to_numeric(
            df["cvss_score"],
            errors="coerce"
        )

        .fillna(0)

        .clip(0, 10)

        * 10
    )

    # --------------------------------------------------------
    # COMPOSITE PRIORITY SCORE
    # --------------------------------------------------------

    df["priority_score"] = (

        0.60 * df["ml_score"]

        +

        0.25 * df["epss_score"]

        +

        0.15 * df["cvss_normalized"]

    )

    df["priority_score"] = (
        df["priority_score"].clip(0, 100)
    )

    # --------------------------------------------------------
    # PRIORITY CLASSIFICATION
    # --------------------------------------------------------

    df["priority"] = pd.cut(

        df["priority_score"],

        bins=[
            -float("inf"),
            25,
            50,
            75,
            float("inf")
        ],

        labels=[
            "Low",
            "Medium",
            "High",
            "Critical"
        ],

        right=False

    )

    return df


# ============================================================
# MERGE EXPLANATIONS
# ============================================================

def merge_explanations(df):

    df = df.copy()

    explanation_df = load_optional_csv(
        EXPLANATIONS
    )

    if (
        not explanation_df.empty
        and "cve" in explanation_df.columns
    ):

        explanation_columns = [
            "cve",
            "explanation",
            "risk_summary"
        ]

        available_columns = [
            column
            for column in explanation_columns
            if column in explanation_df.columns
        ]

        explanation_df = explanation_df[
            available_columns
        ].copy()

        explanation_df["cve"] = (
            explanation_df["cve"].astype(str)
        )

        explanation_df = explanation_df.drop_duplicates(
            subset=["cve"],
            keep="first"
        )

        df["cve"] = df["cve"].astype(str)

        df = df.merge(
            explanation_df,
            on="cve",
            how="left"
        )

    return df


# ============================================================
# MERGE REMEDIATION
# ============================================================

def merge_remediation(df):

    df = df.copy()

    remediation_df = load_optional_csv(
        RECOMMENDATIONS
    )

    if (
        not remediation_df.empty
        and "cve" in remediation_df.columns
    ):

        remediation_columns = [
            "cve",
            "remediation_action",
            "remediation_urgency"
        ]

        available_columns = [
            column
            for column in remediation_columns
            if column in remediation_df.columns
        ]

        remediation_df = remediation_df[
            available_columns
        ].copy()

        remediation_df["cve"] = (
            remediation_df["cve"].astype(str)
        )

        remediation_df = remediation_df.drop_duplicates(
            subset=["cve"],
            keep="first"
        )

        df["cve"] = df["cve"].astype(str)

        df = df.merge(
            remediation_df,
            on="cve",
            how="left"
        )

    return df


# ============================================================
# EXPLOITATION STATUS
# ============================================================

def add_exploitation_status(df):

    df = df.copy()

    if "exploited" in df.columns:

        df["known_exploited"] = (

            pd.to_numeric(
                df["exploited"],
                errors="coerce"
            )

            .fillna(0)

            .astype(int)

            == 1
        )

    else:

        df["known_exploited"] = False

    return df


# ============================================================
# REQUIRED FILE CHECKS
# ============================================================

if not os.path.exists(DATA):

    st.error(
        "Dataset file ml_features.csv is missing."
    )

    st.stop()


if not os.path.exists(MODEL):

    st.error(
        "Trained model exploit_model.pkl is missing."
    )

    st.stop()


# ============================================================
# LOAD AND PROCESS DATA
# ============================================================

try:

    df = load_data()

    df = build_scores(df)

    df = merge_explanations(df)

    df = merge_remediation(df)

    df = add_exploitation_status(df)

except Exception as error:

    st.error(
        f"Error loading dashboard data: {error}"
    )

    st.stop()


# ============================================================
# VALIDATE REQUIRED DATA COLUMNS
# ============================================================

required_columns = [
    "cve",
    "epss",
    "cvss_score",
    "priority",
    "priority_score",
    "predicted_exploit_probability"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Required dataset columns are missing: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# LATEST SNAPSHOT
# ============================================================

if "snapshot_date" in df.columns:

    valid_dates = df["snapshot_date"].dropna()

    if not valid_dates.empty:

        latest_date = valid_dates.max()

        latest = df[
            df["snapshot_date"] == latest_date
        ].copy()

    else:

        latest_date = "Unavailable"

        latest = df.copy()

else:

    latest_date = "Unavailable"

    latest = df.copy()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ CYBER-11")

st.subheader(
    "Vulnerability Prioritization "
    "with Exploit-Likelihood Prediction"
)

st.caption(
    "EPSS + CVSS + Machine Learning "
    "| Temporal Evaluation on 2024–2025"
)

st.markdown(
    f"**Latest snapshot:** {latest_date}"
)


# ============================================================
# SIDEBAR FILTERS
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
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01
)

min_cvss = st.sidebar.slider(
    "Minimum CVSS",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5
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

filtered = latest.copy()

# EPSS filter
filtered = filtered[
    filtered["epss"].fillna(0) >= min_epss
].copy()

# CVSS filter
filtered = filtered[
    filtered["cvss_score"].fillna(0) >= min_cvss
].copy()

# Priority filter
filtered = filtered[
    filtered["priority"]
    .astype(str)
    .isin(priority_filter)
].copy()

# CVE search
if search_cve:

    filtered = filtered[
        filtered["cve"]
        .astype(str)
        .str.contains(
            search_cve,
            case=False,
            na=False
        )
    ].copy()


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
        if not filtered.empty
        else "0"
    )
)

c5.metric(
    "Avg CVSS",
    (
        f"{filtered['cvss_score'].mean():.2f}"
        if not filtered.empty
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

        if not filtered.empty:

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

        else:

            st.info(
                "No data available for the selected filters."
            )

    # --------------------------------------------------------
    # TOP VULNERABILITIES
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Top Vulnerabilities Requiring Attention"
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

    if "known_exploited" in filtered.columns:

        show_cols.append(
            "known_exploited"
        )

    available_show_cols = [
        column
        for column in show_cols
        if column in filtered.columns
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

    if not top.empty:

        top["predicted_exploit_probability"] = (
            top["predicted_exploit_probability"] * 100
        ).round(2)

        top["priority_score"] = (
            top["priority_score"].round(2)
        )

        st.dataframe(
            top[available_show_cols],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No vulnerabilities match the selected filters."
        )

    # --------------------------------------------------------
    # CSV DOWNLOAD — RBAC
    # --------------------------------------------------------

    if current_role in ["Admin", "Analyst"]:

        csv_data = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Filtered Vulnerabilities",
            data=csv_data,
            file_name="cyber11_filtered_vulnerabilities.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Viewer role: Download functionality is disabled."
        )


# ============================================================
# TAB 2 — VULNERABILITY SEARCH
# ============================================================

with tab2:

    st.markdown(
        "## 🔍 Vulnerability Details"
    )

    if search_cve and not filtered.empty:

        cve_list = (
            filtered["cve"]
            .astype(str)
            .unique()
            .tolist()
        )

        selected_cve = st.selectbox(
            "Select CVE",
            cve_list
        )

        vulnerability = filtered[
            filtered["cve"].astype(str) == selected_cve
        ].iloc[0]

        st.markdown(
            f"### {selected_cve}"
        )

        a, b, c = st.columns(3)

        a.metric(
            "EPSS",
            f"{safe_number(vulnerability['epss']):.4f}"
        )

        b.metric(
            "CVSS",
            f"{safe_number(vulnerability['cvss_score']):.1f}"
        )

        c.metric(
            "Priority Score",
            f"{safe_number(vulnerability['priority_score']):.2f}"
        )

        # ----------------------------------------------------
        # EXPLOIT LIKELIHOOD
        # ----------------------------------------------------

        st.markdown(
            "### Exploit Likelihood"
        )

        probability = safe_number(
            vulnerability["predicted_exploit_probability"]
        )

        probability = max(
            0.0,
            min(1.0, probability)
        )

        st.progress(
            probability
        )

        st.write(
            f"Predicted exploit probability: "
            f"**{probability * 100:.2f}%**"
        )

        st.write(
            f"Priority classification: "
            f"**{vulnerability['priority']}**"
        )

        # ----------------------------------------------------
        # EXPLOITATION STATUS
        # ----------------------------------------------------

        st.markdown(
            "### Known Exploitation Status"
        )

        if vulnerability.get(
            "known_exploited",
            False
        ):

            st.warning(
                "⚠️ This vulnerability is identified as exploited."
            )

        else:

            st.info(
                "No known exploited status is available "
                "in the current dataset."
            )

        # ----------------------------------------------------
        # RISK COMPONENTS
        # ----------------------------------------------------

        st.markdown(
            "### Risk Components"
        )

        details = pd.DataFrame(
            {
                "Component": [
                    "ML Exploit Likelihood",
                    "EPSS",
                    "CVSS"
                ],

                "Value": [
                    probability * 100,

                    safe_number(
                        vulnerability["epss"]
                    ) * 100,

                    safe_number(
                        vulnerability["cvss_score"]
                    ) * 10
                ]
            }
        )

        st.bar_chart(
            details.set_index("Component")
        )

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Risk Explanation"
        )

        if (
            "risk_summary" in vulnerability.index
            and pd.notna(vulnerability["risk_summary"])
        ):

            st.write(
                vulnerability["risk_summary"]
            )

        elif (
            "explanation" in vulnerability.index
            and pd.notna(vulnerability["explanation"])
        ):

            st.write(
                vulnerability["explanation"]
            )

        else:

            st.write(
                f"This vulnerability received a priority score of "
                f"{safe_number(vulnerability['priority_score']):.2f}/100. "
                f"The score combines machine-learning exploit likelihood, "
                f"EPSS, and CVSS severity."
            )

        # ----------------------------------------------------
        # REMEDIATION
        # ----------------------------------------------------

        st.markdown(
            "### 🛠️ Recommended Remediation"
        )

        if (
            "remediation_action" in vulnerability.index
            and pd.notna(vulnerability["remediation_action"])
        ):

            st.write(
                vulnerability["remediation_action"]
            )

        else:

            st.info(
                "Remediation recommendation is unavailable. "
                "Run remediation.py to generate recommendations."
            )

        # ----------------------------------------------------
        # REMEDIATION URGENCY
        # ----------------------------------------------------

        if (
            "remediation_urgency" in vulnerability.index
            and pd.notna(vulnerability["remediation_urgency"])
        ):

            st.write(
                "**Remediation Urgency:** "
                f"{vulnerability['remediation_urgency']}"
            )

    elif search_cve:

        st.warning(
            "No matching CVE found."
        )

    else:

        st.info(
            "Enter a CVE in the sidebar to view "
            "detailed vulnerability information."
        )


# ============================================================
# TAB 3 — MODEL ANALYSIS
# ============================================================

with tab3:

    st.markdown(
        "## 🤖 Machine Learning Analysis"
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    if os.path.exists(METRICS):

        try:

            with open(
                METRICS,
                "r",
                encoding="utf-8"
            ) as file:

                metrics = json.load(file)

            best_model_name = metrics.get(
                "best_model",
                "Logistic Regression"
            )

            results = metrics.get(
                "results",
                []
            )

            model_metrics = next(
                (
                    result
                    for result in results
                    if result.get("Model") == best_model_name
                ),
                None
            )

            if model_metrics is not None:

                st.write(
                    f"**Best Model:** {best_model_name}"
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "ROC-AUC",
                    f"{model_metrics.get('ROC_AUC', 0):.4f}"
                )

                b.metric(
                    "PR-AUC",
                    f"{model_metrics.get('PR_AUC', 0):.4f}"
                )

                c.metric(
                    "Precision",
                    f"{model_metrics.get('Precision', 0):.4f}"
                )

                d.metric(
                    "Recall",
                    f"{model_metrics.get('Recall', 0):.4f}"
                )

                e, f, g = st.columns(3)

                e.metric(
                    "Accuracy",
                    f"{model_metrics.get('Accuracy', 0):.4f}"
                )

                f.metric(
                    "F1 Score",
                    f"{model_metrics.get('F1', 0):.4f}"
                )

                g.metric(
                    "Selected Model",
                    best_model_name
                )

            else:

                st.warning(
                    "Best model metrics could not be found "
                    "in model_metrics.json."
                )

        except Exception as error:

            st.error(
                f"Error reading model metrics: {error}"
            )

    else:

        st.warning(
            "Model metrics file not found."
        )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.markdown(
        "### Model Configuration"
    )

    st.write(
        "Training period: **2021–2023**"
    )

    st.write(
        "Testing period: **2024–2025**"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 🔬 Feature Importance"
    )

    if os.path.exists(IMPORTANCE):

        try:

            importance = pd.read_csv(
                IMPORTANCE
            )

            if (
                "importance_mean" in importance.columns
                and "feature" in importance.columns
            ):

                importance = importance.sort_values(
                    "importance_mean",
                    ascending=False
                )

                st.bar_chart(
                    importance.set_index("feature")[
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
                    "Feature importance file has an unexpected format."
                )

        except Exception as error:

            st.error(
                f"Error loading feature importance: {error}"
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
       Composite Priority Score
              ↓
       Risk Classification
              ↓
      Explainability Analysis
              ↓
   Remediation Recommendations
              ↓
          Dashboard
        """
    )

    st.markdown(
        "### Priority Levels"
    )

    st.write(
        "**Critical:** Priority score ≥ 75"
    )

    st.write(
        "**High:** Priority score 50–74.99"
    )

    st.write(
        "**Medium:** Priority score 25–49.99"
    )

    st.write(
        "**Low:** Priority score < 25"
    )

    st.markdown(
        "### Composite Scoring Weights"
    )

    st.write(
        "- Machine Learning Exploit Probability: **60%**"
    )

    st.write(
        "- EPSS: **25%**"
    )

    st.write(
        "- Normalized CVSS: **15%**"
    )


# ============================================================
# ADMIN PANEL
# ============================================================

if current_role == "Admin":

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "## ⚙️ Admin Panel"
    )

    with st.sidebar.expander(
        "👥 User Management",
        expanded=False
    ):

        # ----------------------------------------------------
        # LOAD AUTH CONFIG
        # ----------------------------------------------------

        try:

            with open(
                AUTH_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                auth_config = json.load(file)

        except Exception:

            auth_config = {
                "users": {}
            }

        users = auth_config.get(
            "users",
            {}
        )

        # ----------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------

        st.markdown(
            "### Create User"
        )

        new_username = st.text_input(
            "Username",
            key="admin_new_username"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="admin_new_password"
        )

        new_role = st.selectbox(
            "Role",
            [
                "Admin",
                "Analyst",
                "Viewer"
            ],
            key="admin_new_role"
        )

        if st.button(
            "Create User",
            key="admin_create_user"
        ):

            username = new_username.strip()

            if not username:

                st.error(
                    "Username cannot be empty."
                )

            elif username in users:

                st.error(
                    "Username already exists."
                )

            elif len(new_password) < 8:

                st.error(
                    "Password must contain at least 8 characters."
                )

            else:

                salt = secrets.token_hex(16)

                password_hash = hashlib.pbkdf2_hmac(
                    "sha256",
                    new_password.encode("utf-8"),
                    salt.encode("utf-8"),
                    200000
                ).hex()

                users[username] = {
                    "password_hash": f"{salt}:{password_hash}",
                    "role": new_role
                }

                auth_config["users"] = users

                try:

                    with open(
                        AUTH_FILE,
                        "w",
                        encoding="utf-8"
                    ) as file:

                        json.dump(
                            auth_config,
                            file,
                            indent=4
                        )

                    st.success(
                        f"User '{username}' created successfully."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Could not save user: {error}"
                    )

        # ----------------------------------------------------
        # EXISTING USERS
        # ----------------------------------------------------

        st.markdown(
            "### Existing Users"
        )

        if users:

            user_display = pd.DataFrame(
                [
                    {
                        "Username": username,
                        "Role": details.get(
                            "role",
                            "Unknown"
                        )
                    }

                    for username, details in users.items()
                ]
            )

            st.dataframe(
                user_display,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No users found."
            )