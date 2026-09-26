# src/models.py

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.preprocessing import prepare_data


def evaluate_predictions(y_true, y_pred, y_proba=None) -> dict:
    """Computes core classification metrics for class 1 (churn)."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["ROC-AUC"] = np.nan
    return metrics


def plot_all_confusion_matrices(
    cms: dict, output_dir: str | Path = "reports/figures"
) -> None:
    """Plots a 2x2 grid of confusion matrices for our core competing models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (model_name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Stayed (0)", "Churned (1)"],
            yticklabels=["Stayed (0)", "Churned (1)"],
        )
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    save_file = output_path / "model_comparison_confusion_matrices.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"\n[Artifact] Confusion matrix grid saved to: {save_file.resolve()}")


def run_model_experiments(data_path: str = "data/processed/telco_cleaned.csv"):
    # 1. Load leak-free train/test transformed matrices
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path)

    # 2. Define the candidate models
    models = {
        "Logistic Regression (Default)": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}
    confusion_matrices = {}

    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"Training: {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = evaluate_predictions(y_test, y_pred, y_proba)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    # 3. Present consolidated scorecard
    df_results = pd.DataFrame(results).T
    print("\n=== Comprehensive Model Benchmark ===")
    print(df_results.round(4))

    # 4. Generate comparison plots
    plot_all_confusion_matrices(confusion_matrices)

    return df_results


if __name__ == "__main__":
    run_model_experiments()