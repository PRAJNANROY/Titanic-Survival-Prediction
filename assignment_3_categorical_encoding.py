import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def main():
    df = pd.DataFrame({
        "color": ["Red", "Blue", "Green", "Blue"],
        "size": ["Small", "Large", "Medium", "Small"],
    })

    # 1. LabelEncoder: converts categories to integer labels (0, 1, 2...)
    le = LabelEncoder()
    df["size_encoded"] = le.fit_transform(df["size"])

    # 2. OneHotEncoder: creates separate binary columns per category
    ohe = OneHotEncoder(sparse_output=False)
    ohe_array = ohe.fit_transform(df[["color"]])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(["color"]),
        index=df.index,
    )

    df_encoded = pd.concat([df.drop(columns=["color"]), ohe_df], axis=1)
    print("=== Encoded DataFrame ===")
    print(df_encoded)


if __name__ == "__main__":
    main()
