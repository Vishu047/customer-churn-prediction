from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw Telco churn data."""
    df_clean = df.copy()

    # 1. Drop customerID
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # 2. Fix TotalCharges
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # 3. Clean Target variable: map Yes/No to 1/0 (robust check)
    if "Churn" in df_clean.columns:
        if set(df_clean["Churn"].dropna().unique()).issubset({"Yes", "No"}):
            df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Saves the cleaned DataFrame to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned dataset saved successfully to: {path.resolve()}")

def prepare_data(data_path: str | Path = "data/processed/telco_cleaned.csv"):
    """
    End-to-end preprocessing pipeline:
    1. Loads the cleaned dataset.
    2. Performs stratified train/test split.
    3. Fits ColumnTransformer on X_train only.
    4. Transforms both X_train and X_test.
    
    Returns:
        X_train_processed, X_test_processed, y_train, y_test, preprocessor
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

    df = pd.read_csv(path)

    # 1. Stratified split
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Extract feature categories
    num_cols, cat_cols = get_feature_lists(X_train)

    # 3. Build & fit pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separates numerical and categorical column names."""
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    - Scales numerical columns with StandardScaler.
    - One-hot encodes categorical columns, dropping the first category to avoid multicollinearity.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def split_data(
    df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42
):
    """
    Splits DataFrame into train and test sets using stratification.
    Stratification ensures both sets have the exact same 73.5% / 26.5% churn ratio.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    processed_path = Path("data") / "processed" / "telco_cleaned.csv"
    df = pd.read_csv(processed_path)

    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    # 2. Identify numerical vs categorical
    num_cols, cat_cols = get_feature_lists(X_train)
    print(f"\nNumerical columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # 3. Fit preprocessor ONLY on X_train, then transform both
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # fit_transform on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # transform ONLY on test data (never fit on test!)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\nProcessed X_train matrix shape: {X_train_processed.shape}")
    print(f"Processed X_test matrix shape:  {X_test_processed.shape}")