import joblib


MODEL_PATH = "models/fake_news_calibrated_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_calibrated.joblib"


def main():
    print("=" * 60)
    print("INDIAN FAKE NEWS DETECTOR")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("\nCalibrated model loaded successfully.")
    print("\nEnter a news statement.")
    print("Type 'exit' to stop.\n")

    while True:
        statement = input("News: ").strip()

        if statement.lower() == "exit":
            print("\nExiting...")
            break

        if not statement:
            print("Please enter a news statement.\n")
            continue

        vector = vectorizer.transform([statement])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        classes = model.classes_

        probability_map = dict(
            zip(classes, probabilities)
        )

        predicted_probability = probability_map[prediction]

        print("\n" + "-" * 60)
        print(f"Prediction: {prediction}")
        print(
            f"Model confidence: "
            f"{predicted_probability * 100:.2f}%"
        )

        print(
            f"FAKE probability: "
            f"{probability_map['FAKE'] * 100:.2f}%"
        )

        print(
            f"TRUE probability: "
            f"{probability_map['TRUE'] * 100:.2f}%"
        )

        print("-" * 60)

        print("\nEnter a news statement.")
        print("Type 'exit' to stop.\n")


if __name__ == "__main__":
    main()