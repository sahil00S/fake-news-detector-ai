import pandas as pd

DATA_PATH = "Data/IFND.csv"
OUTPUT_PATH = "Data/IFND_clean.csv"


def load_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")


def clean_data(df):
    df = df.copy()

    # Remove rows with missing text or labels
    df = df.dropna(subset=["Statement", "Label"])

    # Clean whitespace from text
    df["Statement"] = df["Statement"].str.strip()

    # Normalize labels
    df["Label"] = df["Label"].str.upper().str.strip()

    # Keep only valid target classes
    df = df[df["Label"].isin(["TRUE", "FAKE"])]

    # Remove duplicate statements AFTER text cleaning
    df = df.drop_duplicates(subset=["Statement"])

    return df

def main():
    print("=" * 60)
    print("IFND DATA CLEANING")
    print("=" * 60)

    df = load_data()

    print(f"\nOriginal rows: {len(df)}")

    df_clean = clean_data(df)

    print(f"Cleaned rows: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")

    print("\nLabel distribution after cleaning:")
    print(df_clean["Label"].value_counts())

    print("\nRemaining missing values:")
    print(df_clean[["Statement", "Label"]].isnull().sum())

    print("\nRemaining duplicate statements:")
    print(df_clean["Statement"].duplicated().sum())

    df_clean.to_csv(OUTPUT_PATH, index=False)

    print(f"\nClean dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


