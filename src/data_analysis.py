import pandas as pd

DATA_PATH = "Data/IFND.csv"


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    return df


def main():
    df = load_data()

    print("=" * 60)
    print("IFND DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["Label"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate statements:")
    print(df["Statement"].duplicated().sum())


if __name__ == "__main__":
    main()