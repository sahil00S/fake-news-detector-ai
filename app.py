import joblib
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/fake_news_calibrated_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_calibrated.joblib"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Fake News Detector",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .hero {
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.8;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .result-confidence {
        font-size: 1rem;
        opacity: 0.8;
    }

    /* Information cards */
    .info-card {
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
    }

    .info-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.65;
        font-size: 0.85rem;
        padding: 1.5rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:

    model, vectorizer = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🇮🇳 Fake News Detector")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        """
        This application uses machine learning
        to classify Indian news statements as
        **TRUE**, **FAKE**, or **UNCERTAIN**.
        """
    )

    st.markdown("### Machine Learning")

    st.write(
        """
        **Feature Extraction:** TF-IDF

        **Classifier:** Linear SVM

        **Calibration:** Sigmoid probability calibration

        **Dataset:** IFND
        """
    )

    st.markdown("### Model Performance")

    st.metric(
        "Test Accuracy",
        "96.46%"
    )

    st.metric(
        "Macro F1",
        "0.96"
    )

    st.markdown("---")

    st.caption(
        "Academic project • AICTE Internship"
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">🇮🇳 Indian Fake News Detector</div>
    <div class="hero-subtitle">Machine-learning powered analysis of Indian news statements using TF-IDF and a calibrated Linear SVM.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ This is a machine-learning prediction tool, "
    "not a definitive fact-checking service. "
    "Always verify important claims using reliable sources."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📰 Analyze a News Statement")

st.write(
    "Paste a news headline, statement, or short article below."
)

statement = st.text_area(
    "News statement",
    height=220,
    placeholder=(
        "Example: The government announced a new policy "
        "to improve railway passenger safety..."
    ),
    label_visibility="collapsed",
)


analyze = st.button(
    "🔍 Analyze News",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not statement.strip():

        st.warning(
            "Please enter a news statement before analyzing."
        )

    else:

        with st.spinner("Analyzing the statement..."):

            text_vector = vectorizer.transform(
                [statement]
            )

            prediction = model.predict(
                text_vector
            )[0]

            probabilities = model.predict_proba(
                text_vector
            )[0]

            probability_map = dict(
                zip(
                    model.classes_,
                    probabilities
                )
            )

            fake_probability = (
                probability_map["FAKE"] * 100
            )

            true_probability = (
                probability_map["TRUE"] * 100
            )

            confidence = max(
                fake_probability,
                true_probability
            )


        # --------------------------------------------------------
        # RESULT CLASSIFICATION
        # --------------------------------------------------------

        if confidence < 60:

            result = "UNCERTAIN"
            emoji = "🟡"

        elif prediction == "FAKE":

            result = "LIKELY FAKE"
            emoji = "🔴"

        else:

            result = "LIKELY TRUE"
            emoji = "🟢"


        # --------------------------------------------------------
        # RESULT HEADER
        # --------------------------------------------------------

        st.markdown("---")

        st.subheader("📊 Analysis Result")

        st.markdown(
        f"""
<div class="result-card">
    <div class="result-title">{emoji} {result}</div>
    <div class="result-confidence">Model confidence: {confidence:.2f}%</div>
</div>
""",
        unsafe_allow_html=True,
    )

        # --------------------------------------------------------
        # PROBABILITY METRICS
        # --------------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🔴 FAKE probability",
                f"{fake_probability:.2f}%"
            )

        with col2:

            st.metric(
                "🟢 TRUE probability",
                f"{true_probability:.2f}%"
            )

        with col3:

            st.metric(
                "🎯 Model confidence",
                f"{confidence:.2f}%"
            )


        # --------------------------------------------------------
        # PROBABILITY VISUALIZATION
        # --------------------------------------------------------

        st.markdown("### Probability Distribution")

        st.write(
            f"FAKE — {fake_probability:.2f}%"
        )

        st.progress(
            min(int(fake_probability), 100)
        )

        st.write(
            f"TRUE — {true_probability:.2f}%"
        )

        st.progress(
            min(int(true_probability), 100)
        )


        # --------------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------------

        st.markdown("### 🧠 Interpretation")

        if result == "LIKELY FAKE":

            st.error(
                "The model detected patterns that are more "
                "similar to FAKE examples in the training data."
            )

        elif result == "LIKELY TRUE":

            st.success(
                "The model detected patterns that are more "
                "similar to TRUE examples in the training data."
            )

        else:

            st.warning(
                "The model is not sufficiently confident "
                "to classify this statement as likely TRUE "
                "or likely FAKE."
            )


        # --------------------------------------------------------
        # IMPORTANT NOTICE
        # --------------------------------------------------------

        st.info(
            "💡 Model predictions are based on linguistic patterns "
            "learned from the IFND dataset. A prediction does not "
            "establish whether a real-world claim is factually true."
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🔬 About This Project")

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">📚 Dataset</div>
    IFND — Indian Fake News Dataset<br><br>
    56,149 records after cleaning
</div>
""",
        unsafe_allow_html=True,
    )
with info2:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">⚙️ ML Pipeline</div>
    Text → TF-IDF → Linear SVM → Probability Calibration
</div>
""",
        unsafe_allow_html=True,
    )

with info3:

    st.markdown(
        """
<div class="info-card">
    <div class="info-title">🏆 Performance</div>
    96.46% test accuracy<br><br>
    0.96 macro F1-score
</div>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    🇮🇳 Indian Fake News Detector<br>
    TF-IDF + Calibrated Linear SVM<br>
    Built as an AICTE Internship Project
</div>
""",
    unsafe_allow_html=True,
)