from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_numerical_summary(df: pd.DataFrame) -> None:
    """Prints mean and median of numerical features grouped by churn status."""
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print("\n=== Numerical Features Grouped by Churn (0 = Stayed, 1 = Churned) ===")
    summary = df.groupby("Churn")[numerical_cols].agg(["mean", "median"]).round(2)
    print(summary)


def generate_categorical_churn_rates(df: pd.DataFrame, columns: list[str]) -> None:
    """Computes and prints the churn rate for key categorical features."""
    print("\n=== Churn Rates by Category ===")
    for col in columns:
        churn_rate = df.groupby(col)["Churn"].agg(["count", "mean"]).rename(
            columns={"mean": "churn_rate"}
        )
        churn_rate["churn_rate"] = (churn_rate["churn_rate"] * 100).round(1)
        print(f"\nFeature: {col}")
        print(churn_rate.sort_values(by="churn_rate", ascending=False))


def save_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates and saves visual charts to the reports/figures directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Tenure vs Churn Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        multiple="stack",
        palette={0: "#0a77ec", 1: "#d95f02"},
        bins=30,
    )
    plt.title("Customer Tenure Distribution by Churn Status")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(output_dir / "tenure_distribution.png", dpi=150)
    plt.close()

    # 2. Monthly Charges vs Churn
    plt.figure(figsize=(8, 4))
    sns.kdeplot(
        data=df[df["Churn"] == 0]["MonthlyCharges"],
        label="Stayed (0)",
        fill=True,
        color="#2b5c8f",
    )
    sns.kdeplot(
        data=df[df["Churn"] == 1]["MonthlyCharges"],
        label="Churned (1)",
        fill=True,
        color="#d95f02",
    )
    plt.title("Monthly Charges Density: Stayed vs Churned")
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_charges_density.png", dpi=150)
    plt.close()

    # 3. Contract Type vs Churn Rate
    plt.figure(figsize=(7, 4))
    contract_rates = (
        df.groupby("Contract")["Churn"].mean().reset_index()
    )
    contract_rates["Churn_Pct"] = contract_rates["Churn"] * 100
    sns.barplot(
        data=contract_rates,
        x="Contract",
        y="Churn_Pct",
        palette="viridis",
        hue="Contract",
        legend=False,
    )
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Percentage (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "contract_churn_rate.png", dpi=150)
    plt.close()

    print(f"\nCharts successfully generated in: {output_dir.resolve()}")


if __name__ == "__main__":
    processed_file = Path("data") / "processed" / "telco_cleaned.csv"
    figures_path = Path("reports") / "figures"

    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed file not found at {processed_file}. Run preprocessing.py first!"
        )

    df = pd.read_csv(processed_file)

    # 1. Numerical analysis
    generate_numerical_summary(df)

    # 2. Categorical analysis on highest-signal features
    key_categories = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    generate_categorical_churn_rates(df, key_categories)

    # 3. Save plots
    save_eda_plots(df, figures_path)