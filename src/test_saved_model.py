import joblib
import pandas as pd


DATA_PATH = "Data/IFND_clean.csv"
MODEL_PATH = "models/fake_news_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"


def main():
    print("=" * 60)
    print("TESTING SAVED FAKE NEWS MODEL")
    print("=" * 60)

    # Load model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Load real dataset
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    # Take examples from both classes
    fake_examples = df[df["Label"] == "FAKE"].sample(
        n=5,
        random_state=42
    )

    true_examples = df[df["Label"] == "TRUE"].sample(
        n=5,
        random_state=42
    )

    test_examples = pd.concat(
        [fake_examples, true_examples]
    )

    print("\nTesting 5 FAKE and 5 TRUE real articles...\n")

    correct = 0

    for index, row in test_examples.iterrows():
        statement = row["Statement"]
        actual = row["Label"]

        vector = vectorizer.transform([statement])
        prediction = model.predict(vector)[0]
        score = model.decision_function(vector)[0]

        if prediction == actual:
            correct += 1

        print("-" * 60)
        print(f"Actual:     {actual}")
        print(f"Prediction: {prediction}")
        print(f"Score:      {score:.4f}")
        print(f"Statement:  {statement[:250]}")

    print("\n" + "=" * 60)
    print(f"Correct: {correct}/10")
    print(f"Accuracy: {correct / 10 * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()