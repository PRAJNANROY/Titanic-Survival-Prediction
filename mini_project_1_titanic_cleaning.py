import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def main():
    # 1. Load the Kaggle Titanic dataset (ensure train.csv is in the same folder)
    df = pd.read_csv("train.csv")
    print("Initial shape:", df.shape)
    print("Missing values before cleaning:\n", df.isnull().sum())

    # 2. Clean missing data
    # Median imputation for Age (robust to skewness/outliers)
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Mode imputation for Embarked (categorical)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Drop high-cardinality / heavily missing text columns
    cols_to_drop = [c for c in ["Cabin", "Name", "Ticket"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Fill any remaining numeric columns with their median
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    print("\nMissing values after cleaning:\n", df.isnull().sum())

    # 3. Visualize Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=30, kde=True)
    plt.title("Titanic Passenger Age Distribution (Cleaned)")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("age_distribution.png", dpi=150)
    plt.close()

    # 4. Encode 'Sex' and 'Embarked' columns
    le_sex = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"])

    ohe_embarked = OneHotEncoder(sparse_output=False)
    embarked_encoded = ohe_embarked.fit_transform(df[["Embarked"]])
    embarked_df = pd.DataFrame(
        embarked_encoded,
        columns=ohe_embarked.get_feature_names_out(["Embarked"]),
        index=df.index,
    )

    df_cleaned = pd.concat([df.drop(columns=["Embarked"]), embarked_df], axis=1)

    # 5. Output cleaned dataset as a new CSV
    output_path = "titanic_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to '{output_path}'. Final shape: {df_cleaned.shape}")
    print(df_cleaned.head())


if __name__ == "__main__":
    main()
