import pandas as pd


def main():
    # Sample dataset (replace with pd.read_csv("your_dataset.csv") for external files)
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "Age": [24, 30, 22, 35, 28],
        "Department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
        "Salary": [72000, 58000, 65000, 81000, 61000],
    }
    df = pd.DataFrame(data)

    # 1. Structure, data types, and non-null counts
    print("=== Dataset Info ===")
    df.info()

    # 2. Summary statistics for numerical columns
    print("\n=== Numerical Summary ===")
    print(df.describe())

    # 3. Summary statistics for categorical/object columns
    print("\n=== Categorical Summary ===")
    print(df.describe(include="object"))


if __name__ == "__main__":
    main()
