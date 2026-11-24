
import os
import json
import hashlib
import hmac
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_FILE = os.path.join(BASE_DIR, "auth_config.json")

def load_users():
    if not os.path.exists(AUTH_FILE):
        return {}

    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as file:
            config = json.load(file)

        return config.get("users", {})

    except (json.JSONDecodeError, OSError):
        return {}

def verify_password(password, stored_hash):
    try:
        salt, expected_hash = stored_hash.split(":", 1)

        calculated_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            200000
        ).hex()

        return hmac.compare_digest(
            calculated_hash,
            expected_hash
        )

    except (ValueError, AttributeError):
        return False

def login():
    if st.session_state.get("authenticated", False):
        return True

    # Modern cybersecurity login design
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, #17365d 0%, transparent 35%),
            radial-gradient(circle at 85% 80%, #123b55 0%, transparent 35%),
            linear-gradient(135deg, #071426, #0b1930 55%, #10243d);
        color: #ffffff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 5rem;
        max-width: 850px;
    }

    .login-brand {
        text-align: center;
        margin-bottom: 28px;
    }

    .login-icon {
        display: inline-flex;
        width: 76px;
        height: 76px;
        align-items: center;
        justify-content: center;
        border-radius: 22px;
        background: linear-gradient(135deg, #087cf0, #00c6d7);
        font-size: 38px;
        box-shadow: 0 10px 35px rgba(0, 150, 255, 0.25);
        margin-bottom: 14px;
    }

    .login-brand h1 {
        color: #ffffff;
        font-size: 34px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }

    .login-brand p {
        color: #9db5d0;
        font-size: 14px;
        margin-top: 8px;
    }

    .login-panel {
        background: rgba(18, 37, 62, 0.92);
        border: 1px solid rgba(123, 171, 220, 0.22);
        border-radius: 20px;
        padding: 25px 28px;
        margin-bottom: 15px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .login-panel h3 {
        color: #ffffff;
        text-align: center;
        margin: 0 0 8px 0;
        font-size: 23px;
    }

    .login-panel p {
        color: #a9bfd8;
        text-align: center;
        font-size: 13px;
    }

    div[data-testid="stTextInput"] label {
        color: #dce9f8 !important;
    }

    div[data-testid="stTextInput"] input {
        background: #10233b !important;
        color: white !important;
        border: 1px solid #365675 !important;
        border-radius: 9px !important;
        min-height: 43px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #20a8ff !important;
        box-shadow: 0 0 0 1px #20a8ff !important;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #087cf0, #00a9d6);
        color: white;
        border: none;
        border-radius: 9px;
        min-height: 45px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0068d7, #008ebc);
        border: none;
        color: white;
    }

    .login-footer {
        text-align: center;
        color: #829ab5;
        font-size: 12px;
        margin-top: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-brand">
        <div class="login-icon">🛡️</div>
        <h1>CYBER-11</h1>
        <p>INTELLIGENT VULNERABILITY PRIORITIZATION SYSTEM</p>
    </div>

    <div class="login-panel">
        <h3>Welcome Back</h3>
        <p>Sign in to access your security dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])

    with center:
        with st.form("cyber11_login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter your username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            submitted = st.form_submit_button(
                "🔐  Secure Login",
                use_container_width=True
            )

        if submitted:
            users = load_users()

            if not users:
                st.error("No user accounts found. Check auth_config.json.")
            elif username.strip() not in users:
                st.error("Username not found.")
            else:
                user = users[username.strip()]
                stored_hash = user.get("password_hash", "")

                if verify_password(password, stored_hash):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    st.session_state.role = user.get("role", "Viewer")
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")

    st.markdown("""
    <div class="login-footer">
        🛡️ Secure Access &nbsp; | &nbsp;
        Role-Based Authentication &nbsp; | &nbsp;
        CYBER-11
    </div>
    """, unsafe_allow_html=True)

    return False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()

def get_role():
    return st.session_state.get("role", "Viewer")