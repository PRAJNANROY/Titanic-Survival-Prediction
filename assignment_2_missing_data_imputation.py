import numpy as np
import pandas as pd


def main():
    # Sample dataset with missing values
    df = pd.DataFrame({
        "age": [22.0, 38.0, np.nan, 35.0, 54.0, np.nan],
        "fare": [7.25, 71.28, 7.92, 53.10, np.nan, 8.05],
    })

    print("Missing values before imputation:")
    print(df.isnull().sum())

    # Mean imputation (suitable for symmetrical distributions)
    df["fare_mean_imputed"] = df["fare"].fillna(df["fare"].mean())

    # Median imputation (preferred when outliers or skewness exist)
    df["age_median_imputed"] = df["age"].fillna(df["age"].median())

    print("\nMissing values after imputation:")
    print(df.isnull().sum())

    print("\nResulting DataFrame:")
    print(df)


if __name__ == "__main__":
    main()
