import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


DATA_PATH = "Data/IFND_clean.csv"


def main():
    print("=" * 60)
    print("IFND MODEL COMPARISON")
    print("=" * 60)

    # Load data
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

    print(f"\nTraining documents: {len(X_train)}")
    print(f"Testing documents: {len(X_test)}")
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Training Logistic Regression...")
    print("-" * 60)

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    logistic_model.fit(X_train_tfidf, y_train)

    logistic_pred = logistic_model.predict(X_test_tfidf)

    logistic_accuracy = accuracy_score(y_test, logistic_pred)

    print(f"\nLogistic Regression Accuracy: {logistic_accuracy:.4f}")
    print(f"Percentage: {logistic_accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, logistic_pred))

    # --------------------------------------------------
    # Linear SVM
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("Training Linear SVM...")
    print("-" * 60)

    svm_model = LinearSVC(
        random_state=42,
    )

    svm_model.fit(X_train_tfidf, y_train)

    svm_pred = svm_model.predict(X_test_tfidf)

    svm_accuracy = accuracy_score(y_test, svm_pred)

    print(f"\nLinear SVM Accuracy: {svm_accuracy:.4f}")
    print(f"Percentage: {svm_accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, svm_pred))

    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(f"\nLogistic Regression: {logistic_accuracy * 100:.2f}%")
    print(f"Linear SVM:          {svm_accuracy * 100:.2f}%")

    if logistic_accuracy > svm_accuracy:
        print("\nWinner: Logistic Regression")
    elif svm_accuracy > logistic_accuracy:
        print("\nWinner: Linear SVM")
    else:
        print("\nResult: Tie")


if __name__ == "__main__":
    main()