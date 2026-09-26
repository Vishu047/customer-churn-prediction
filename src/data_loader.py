# from pathlib import Path
# import pandas as pd


# def load_raw_data(filepath: str | Path) -> pd.DataFrame:
#     """Loads the raw customer churn CSV into a Pandas DataFrame."""
#     path = Path(filepath)
#     if not path.exists():
#         raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
    
#     df = pd.read_csv(path)
#     return df


# if __name__ == "__main__":
#     # Define relative path using pathlib for cross-platform stability
#     raw_data_path = Path("data") / "raw" / "Telco-Customer-Churn.csv"
    
#     print("Loading raw data...")
#     df = load_raw_data(raw_data_path)
    
#     print("\n--- Dataset Dimensions ---")
#     print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
#     print("\n--- First 3 Rows ---")
#     print(df.head(3))
    
#     print("\n--- Column Data Types & Non-Null Counts ---")
#     df.info()

from pathlib import Path
import pandas as pd


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Loads the raw customer churn CSV into a Pandas DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
    return pd.read_csv(path)


def inspect_total_charges(df: pd.DataFrame) -> None:
    """Diagnoses whitespace and type issues in TotalCharges."""
    # Check for pure whitespace values
    whitespace_mask = df["TotalCharges"].str.strip() == ""
    whitespace_count = whitespace_mask.sum()
    print(f"Number of rows with empty whitespace in TotalCharges: {whitespace_count}")

    # Inspect a few rows where TotalCharges is blank
    blank_rows = df[whitespace_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]
    print("\nSample rows with blank TotalCharges:")
    print(blank_rows.head(5))


if __name__ == "__main__":
    raw_data_path = Path("data") / "raw" / "Telco-Customer-Churn.csv"
    df = load_raw_data(raw_data_path)


    print("\n--- Dataset Dimensions ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    inspect_total_charges(df)

    df.info()