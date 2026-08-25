import os

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


DATA_PATH = "Data/IFND_clean.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_news_calibrated_model.joblib"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer_calibrated.joblib"
)


def main():
    print("=" * 60)
    print("CALIBRATED FAKE NEWS MODEL")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    X = df["Statement"]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nTraining documents: {len(X_train)}")
    print(f"Testing documents: {len(X_test)}")
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    print("\nTraining Linear SVM...")

    base_svm = LinearSVC(
        random_state=42,
    )

    model = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=3,
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    y_pred = model.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print("CALIBRATED MODEL RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print("\n" + "=" * 60)
    print("CALIBRATED MODEL SAVED")
    print("=" * 60)

    print(
        f"\nModel: {MODEL_PATH}"
    )

    print(
        f"Vectorizer: {VECTORIZER_PATH}"
    )


if __name__ == "__main__":
    main()