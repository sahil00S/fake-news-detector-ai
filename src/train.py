import os

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = "Data/IFND_clean.csv"
MODEL_DIR = "models"

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.joblib")


def main():
    print("=" * 60)
    print("TRAINING FINAL FAKE NEWS MODEL")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    X = df["Statement"]
    y = df["Label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Final model: Linear SVM
    model = LinearSVC(
        random_state=42,
    )

    print("\nTraining Linear SVM...")
    model.fit(X_train_tfidf, y_train)

    # Test predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("FINAL MODEL RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"\nVectorizer: {VECTORIZER_PATH}")
    print(f"Model:      {MODEL_PATH}")


if __name__ == "__main__":
    main()