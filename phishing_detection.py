"""
AI-Powered Phishing Detection System
-------------------------------------
A Streamlit app that classifies email bodies as Phishing / Safe using a
text (bag-of-words) model combined with heuristic URL-based features.

Bug fixes in this pass:
  1. CRASH BUG: st.text_area() was created with BOTH `key="email_input"`
     AND `value=st.session_state.email_input`. Streamlit forbids setting a
     widget's value both ways once something (here, the sidebar "example"
     buttons, and the initial default) has written to that session_state
     key before the widget is instantiated — it raises a
     StreamlitAPIException the moment you click an example button. Fixed
     by dropping `value=` and letting `key=` alone own the state.
  2. LOGIC BUG: history/confidence always showed P(phishing), even when
     the predicted label was "Safe" — so a Safe email with e.g. a 10%
     phishing score displayed "10% confidence", which reads backwards.
     Now shows confidence in the *predicted* label (proba if Phishing,
     1-proba if Safe), with the raw phishing score shown separately.
  3. Removed duplicate score display (was printed as text AND inside the
     risk meter).
  4. Fragile, version-specific CSS selectors (.css-1d391kg, .stMetricValue,
     etc.) replaced with stable data-testid selectors for the dark theme.
  5. Unused `io` import removed.

Fixes carried over from the previous pass:
  - `model` NameError/scope bug -> now lives in st.session_state.
  - URL features engineered but never fed into the model -> now combined
    with text features via a ColumnTransformer.
  - Removed live `requests.get()` on every URL in an email (SSRF-style
    risk + slow/unreliable) -> replaced with static heuristics.
  - MultinomialNB -> LogisticRegression to handle mixed text + scaled
    numeric features properly.
  - Added a real train/test split with reported accuracy, single-class
    dataset guard, quiet/cached NLTK download.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords

MODEL_PATH = "phishing_model.pkl"
SAMPLE_DATA_PATH = "emails.csv"

NUMERIC_FEATURE_COLS = [
    "url_count",
    "avg_domain_length",
    "avg_path_length",
    "https_ratio",
    "has_ip_address",
    "has_at_symbol",
    "suspicious_word_count",
]

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update",
    "confirm", "bank", "suspend", "urgent", "password",
]


# --------------------------------------------------------------------------
# NLTK setup (only download once, quietly)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


STOPWORDS = ensure_stopwords()


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def clean_email_body(email_body: str) -> str:
    """Lowercase, strip non-alphabetic chars, remove stopwords."""
    if not isinstance(email_body, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", email_body)
    words = [w.lower() for w in text.split() if w.lower() not in STOPWORDS]
    return " ".join(words)


def extract_url_features(email_body: str) -> dict:
    """
    Static, no-network heuristics describing any URLs found in the email.
    Deliberately avoids fetching the URLs (slow + SSRF risk in a security tool).
    """
    if not isinstance(email_body, str):
        email_body = ""

    urls = re.findall(r"(https?://[^\s]+)", email_body)

    if not urls:
        return {
            "url_count": 0,
            "avg_domain_length": 0,
            "avg_path_length": 0,
            "https_ratio": 0,
            "has_ip_address": 0,
            "has_at_symbol": 0,
            "suspicious_word_count": 0,
        }

    domain_lengths, path_lengths = [], []
    https_count = 0
    has_ip = 0
    has_at = 0
    suspicious_count = 0
    ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

    for url in urls:
        parsed = urlparse(url)
        domain_lengths.append(len(parsed.netloc))
        path_lengths.append(len(parsed.path))
        if parsed.scheme == "https":
            https_count += 1
        host = parsed.netloc.split(":")[0].split("@")[-1]
        if ip_pattern.match(host):
            has_ip = 1
        if "@" in parsed.netloc:
            has_at = 1
        lowered = url.lower()
        suspicious_count += sum(1 for w in SUSPICIOUS_WORDS if w in lowered)

    return {
        "url_count": len(urls),
        "avg_domain_length": float(np.mean(domain_lengths)),
        "avg_path_length": float(np.mean(path_lengths)),
        "https_ratio": https_count / len(urls),
        "has_ip_address": has_ip,
        "has_at_symbol": has_at,
        "suspicious_word_count": suspicious_count,
    }


def build_feature_frame(raw_bodies: pd.Series) -> pd.DataFrame:
    """Turn a series of raw email strings into the model's input DataFrame."""
    cleaned = raw_bodies.apply(clean_email_body)
    url_feats = raw_bodies.apply(extract_url_features).apply(pd.Series)
    frame = pd.DataFrame({"cleaned_body": cleaned})
    frame = pd.concat([frame.reset_index(drop=True), url_feats.reset_index(drop=True)], axis=1)
    return frame[["cleaned_body"] + NUMERIC_FEATURE_COLS]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", CountVectorizer(), "cleaned_body"),
            ("num", MinMaxScaler(), NUMERIC_FEATURE_COLS),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_model(data: pd.DataFrame):
    """Train, evaluate on a held-out split, and persist the model."""
    data = data.dropna(subset=["email_body", "label"]).copy()
    y = data["label"].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "Training data needs at least two classes (0 = safe, 1 = phishing)."
        )

    X = build_feature_frame(data["email_body"])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test) if len(X_test) else None

    joblib.dump(model, MODEL_PATH)
    return model, accuracy, len(data)


def predict_phishing(model, email_body: str):
    """Returns (label, phishing_probability). phishing_probability is
    always P(class == 1), regardless of the predicted label."""
    X = build_feature_frame(pd.Series([email_body]))

    prediction = model.predict(X)[0]
    phishing_probability = None

    try:
        proba_all = model.predict_proba(X)
        classes = list(model.classes_)
        if 1 in classes:
            phishing_probability = float(proba_all[0][classes.index(1)])
        else:
            # Model never saw class 1 during training; fall back to the
            # probability of the predicted class so we still show *something*.
            phishing_probability = float(np.max(proba_all[0]))
    except Exception:
        phishing_probability = None

    label = "Phishing" if int(prediction) == 1 else "Safe"
    return label, phishing_probability


def load_model_from_disk():
    """Load the trained model if it exists."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def risk_meter(phishing_probability: float):
    """Renders a bar for P(phishing). Always uses the raw phishing score,
    regardless of which label was predicted, so it reads unambiguously as
    'how phishy is this' rather than 'confidence in the verdict'."""
    if phishing_probability is None:
        return

    pct = max(0.0, min(1.0, phishing_probability))

    if pct < 0.4:
        color, risk_word = "#22c55e", "Low"
    elif pct < 0.7:
        color, risk_word = "#f59e0b", "Medium"
    else:
        color, risk_word = "#ef4444", "High"

    st.markdown(
        f"""
        <div style="margin-top:8px; color:#e2e8f0;">
            <div style="display:flex;justify-content:space-between;font-weight:600;">
                <span>Phishing risk</span>
                <span><b>{risk_word}</b> · {pct:.0%}</span>
            </div>
            <div style="background:#334155;border-radius:999px;height:12px;margin-top:8px;">
                <div style="width:{pct*100:.0f}%;background:{color};height:100%;border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        if os.path.exists("logo.JPEG"):
            st.image("logo.JPEG", width=90)
        else:
            st.markdown(
                '<div style="font-size:46px; text-align:center;">🛡️</div>',
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown(
            '<div style="font-size:30px; font-weight:700; color:#fff;">'
            "AI-Powered Phishing Detection</div>"
            '<div style="color:#cbd5e1; font-size:15px; margin-top:8px;">'
            "Paste an email body and get an instant risk assessment, "
            "backed by a text + URL-heuristics model.</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Phishing Detection", page_icon="🛡️", layout="wide")

    # Dark theme — uses stable data-testid selectors instead of Streamlit's
    # auto-generated, version-specific `.css-xxxxx` class names.
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0f172a !important;
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stVerticalBlock"] {
            background-color: transparent !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #111827 !important;
            color: #e2e8f0 !important;
        }
        .card {
            background: #111827; padding: 22px; border-radius: 18px;
            box-shadow: 0 16px 40px rgba(15,23,42,.35);
            border: 1px solid rgba(148,163,184,.12);
        }
        .stButton button {
            background-color: #2563eb !important;
            color: #f8fafc !important;
            border: 1px solid rgba(96,165,250,.4) !important;
        }
        .stButton button:hover {
            background-color: #1d4ed8 !important;
        }
        .stTextArea textarea, .stTextInput input {
            background: #0b1120 !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
        }
        .stMarkdown p, .stMarkdown div, .stMarkdown span,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
        label, .stCaption, [data-testid="stCaptionContainer"] {
            color: #e2e8f0 !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #e2e8f0 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: #0b1120 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "model" not in st.session_state:
        st.session_state.model = load_model_from_disk()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_accuracy" not in st.session_state:
        st.session_state.last_accuracy = None
    if "email_input" not in st.session_state:
        st.session_state.email_input = ""

    render_header()
    st.markdown("---")

    # ---- Sidebar ---------------------------------------------------
    with st.sidebar:
        st.header("Model status")
        if st.session_state.model is not None:
            st.success("Model loaded ✅")
            if hasattr(st.session_state.model, "classes_"):
                st.caption(f"Classes: {list(st.session_state.model.classes_)}")
            if st.session_state.last_accuracy is not None:
                st.metric("Last test accuracy", f"{st.session_state.last_accuracy:.1%}")
        else:
            st.warning("No trained model yet. Go to the **Train Model** tab.")

        st.markdown("---")
        st.subheader("Try an example")
        example1 = "Your account has been suspended. Click https://secure-login.example.com to restore access."
        example2 = "Invoice attached. Please login at http://192.168.5.20/pay to view details."
        example3 = "Hey, attached is the agenda for Monday's team sync. See you there!"

        # Setting session_state[key] here, BEFORE the text_area with that
        # same key is instantiated below, is the correct/only way to
        # programmatically change a keyed widget's value in Streamlit.
        if st.button("⚠️ Suspicious example 1", use_container_width=True):
            st.session_state.email_input = example1
        if st.button("⚠️ Suspicious example 2", use_container_width=True):
            st.session_state.email_input = example2
        if st.button("✅ Benign example", use_container_width=True):
            st.session_state.email_input = example3

        if st.session_state.history:
            st.markdown("---")
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = []

    # ---- Tabs --------------------------------------------------------
    tab_detect, tab_train, tab_about = st.tabs(["🔍 Detect", "📊 Train Model", "ℹ️ About"])

    # ---- Detect tab ----------------------------------------------------
    with tab_detect:
        left, right = st.columns([3, 2])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Check an email")
            # IMPORTANT: no `value=` here. `key="email_input"` alone binds
            # this widget to st.session_state.email_input. Passing `value`
            # too raises a StreamlitAPIException once that key has been set
            # elsewhere (the example buttons) before this line runs.
            email_input = st.text_area(
                "Email body",
                height=240,
                key="email_input",
                label_visibility="collapsed",
                placeholder="Paste the email body here…",
            )
            check_clicked = st.button("🔍 Check Phishing", type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

            if check_clicked:
                if not email_input.strip():
                    st.info("Please enter or select an email body to analyze.")
                elif st.session_state.model is None:
                    st.error("No trained model found. Train one in the **Train Model** tab first.")
                else:
                    with st.spinner("Analyzing..."):
                        label, phishing_probability = predict_phishing(st.session_state.model, email_input)

                    if label == "Phishing":
                        st.error(f"Prediction: **{label}**")
                    else:
                        st.success(f"Prediction: **{label}**")

                    # Confidence in the *predicted label* (not always P(phishing)):
                    # a "Safe" call with a 10% phishing score is 90% confident.
                    if phishing_probability is not None:
                        label_confidence = (
                            phishing_probability if label == "Phishing" else 1 - phishing_probability
                        )
                        st.caption(f"Confidence in this verdict: {label_confidence:.1%}")
                        risk_meter(phishing_probability)
                    else:
                        label_confidence = None
                        st.caption("Confidence score unavailable.")

                    st.session_state.history.insert(
                        0,
                        {
                            "preview": (email_input[:60] + "…") if len(email_input) > 60 else email_input,
                            "label": label,
                            "confidence": f"{label_confidence:.1%}" if label_confidence is not None else "n/a",
                        },
                    )

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Recent checks")
            if st.session_state.history:
                st.dataframe(
                    pd.DataFrame(st.session_state.history),
                    hide_index=True,
                    use_container_width=True,
                    height=280,
                )
            else:
                st.caption("Your recent predictions will show up here.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Train tab -----------------------------------------------------
    with tab_train:
        col1, col2 = st.columns([3, 2])

        uploaded_file = None
        data = None

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Upload training data")
            st.caption("CSV with columns `email_body` and `label` (1 = phishing, 0 = safe).")
            uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except Exception:
                    st.error("Couldn't read that file as a CSV.")
                    data = None

                if data is not None:
                    if not {"email_body", "label"}.issubset(data.columns):
                        st.error("CSV must contain `email_body` and `label` columns.")
                    else:
                        st.write(f"Loaded **{len(data)}** rows.")
                        st.dataframe(data.head(5), use_container_width=True)

                        if st.button("🚀 Train model", type="primary"):
                            try:
                                with st.spinner("Training model..."):
                                    model, accuracy, n = train_model(data)
                                st.session_state.model = model
                                st.session_state.last_accuracy = accuracy
                                st.success(f"Model trained on {n} rows and saved to `{MODEL_PATH}`.")
                                if accuracy is not None:
                                    st.metric("Held-out test accuracy", f"{accuracy:.1%}")
                            except ValueError as e:
                                st.error(str(e))
                            except Exception as e:
                                st.error(f"Training failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Quick actions")
            if st.button("Retrain with sample dataset (`emails.csv`)"):
                if os.path.exists(SAMPLE_DATA_PATH):
                    try:
                        sample_data = pd.read_csv(SAMPLE_DATA_PATH)
                        with st.spinner("Training on sample data..."):
                            model, accuracy, n = train_model(sample_data)
                        st.session_state.model = model
                        st.session_state.last_accuracy = accuracy
                        st.success(f"Retrained on {n} rows from the sample dataset.")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                else:
                    st.error(f"No sample `{SAMPLE_DATA_PATH}` found in the project folder.")

            if st.session_state.model is not None and os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    st.download_button(
                        "⬇️ Download trained model (.pkl)",
                        data=f,
                        file_name="phishing_model.pkl",
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Dataset tips")
            st.write("- Include the full email text for best results.")
            st.write("- Labels: `1` = phishing, `0` = safe.")
            st.write("- Aim for a reasonably balanced dataset.")
            if data is not None and "label" in data.columns:
                st.markdown("**Class balance**")
                st.bar_chart(data["label"].value_counts())
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- About tab -------------------------------------------------------
    with tab_about:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How it works")
        st.write(
            "This app combines two signal sources:\n\n"
            "1. **Text content** — the cleaned email body is vectorized with a "
            "bag-of-words `CountVectorizer`.\n"
            "2. **URL heuristics** — any links in the email are inspected "
            "*without being fetched* for signals like IP-literal hosts, "
            "`@`-tricks, suspicious keywords (`login`, `verify`, `secure`, …), "
            "HTTPS ratio, and link count.\n\n"
            "Both feature sets are combined and fed into a `LogisticRegression` "
            "classifier trained via a `scikit-learn` pipeline."
        )
        st.info(
            "Note: URLs are analyzed structurally only — the app never makes "
            "outbound requests to links found in an email, to avoid the risk "
            "and unreliability of fetching attacker-controlled URLs server-side."
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()