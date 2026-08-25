import joblib
import streamlit as st


MODEL_PATH = "models/fake_news_calibrated_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_calibrated.joblib"


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Indian Fake News Detector",
    page_icon="🇮🇳",
    layout="centered",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🇮🇳 Indian Fake News Detector")

st.write(
    "Analyze Indian news statements using "
    "TF-IDF and a calibrated Linear SVM machine-learning model."
)

st.info(
    "This tool provides a model-based prediction and "
    "should not be treated as a definitive fact-check."
)


# --------------------------------------------------
# Input
# --------------------------------------------------

statement = st.text_area(
    "Paste a news statement",
    height=200,
    placeholder="Paste the news article or statement here..."
)


# --------------------------------------------------
# Analyze
# --------------------------------------------------

if st.button("🔍 Analyze News", type="primary"):

    if not statement.strip():

        st.warning(
            "Please enter a news statement first."
        )

    else:

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

        # --------------------------------------------------
        # Result classification
        # --------------------------------------------------

        if confidence < 60:

            result = "UNCERTAIN"

        elif prediction == "FAKE":

            result = "LIKELY FAKE"

        else:

            result = "LIKELY TRUE"


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.subheader("Analysis Result")

        if result == "LIKELY FAKE":

            st.error(
                f"🔴 {result}"
            )

        elif result == "LIKELY TRUE":

            st.success(
                f"🟢 {result}"
            )

        else:

            st.warning(
                f"🟡 {result}"
            )


        # --------------------------------------------------
        # Probabilities
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "FAKE probability",
                f"{fake_probability:.2f}%"
            )

        with col2:

            st.metric(
                "TRUE probability",
                f"{true_probability:.2f}%"
            )


        st.progress(
            int(confidence)
        )

        st.caption(
            f"Model confidence: {confidence:.2f}%"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Model: TF-IDF + Calibrated Linear SVM | "
    "Dataset: IFND"
)