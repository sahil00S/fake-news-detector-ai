import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "Data/IFND_clean.csv"


def main():
    print("=" * 60)
    print("IFND TRAIN / TEST SPLIT")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    print(f"\nTotal records: {len(df)}")

    X = df["Statement"]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    print("\nTraining percentages:")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nTesting percentages:")
    print((y_test.value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    main()