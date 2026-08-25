import joblib


MODEL_PATH = "models/fake_news_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"


def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_news(text, model, vectorizer):
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]
    decision_score = model.decision_function(text_vector)[0]

    return prediction, decision_score


def main():
    print("=" * 60)
    print("INDIAN FAKE NEWS DETECTOR")
    print("=" * 60)

    model, vectorizer = load_model()

    print("\nModel loaded successfully.")

    while True:
        print("\nEnter a news statement.")
        print("Type 'exit' to stop.")

        text = input("\nNews: ").strip()

        if text.lower() == "exit":
            print("\nExiting...")
            break

        if not text:
            print("Please enter some text.")
            continue

        prediction, decision_score = predict_news(
            text,
            model,
            vectorizer,
        )

        print("\n" + "-" * 60)
        print(f"Prediction: {prediction}")
        print(f"Decision score: {decision_score:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()