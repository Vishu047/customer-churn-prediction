# src/explainability.py

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.preprocessing import prepare_data


def get_feature_names(preprocessor) -> list[str]:
    """Extracts explicit transformed feature names from ColumnTransformer."""
    # 1. Numerical column names
    num_cols = preprocessor.transformers_[0][2]

    # 2. One-hot encoded categorical feature names
    cat_encoder = preprocessor.transformers_[1][1]
    cat_cols_encoded = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])

    return list(num_cols) + list(cat_cols_encoded)


def run_explainability_pipeline(
    model_path: str = "models/best_random_forest.joblib",
    data_path: str = "data/processed/telco_cleaned.csv",
    output_dir: str = "reports/figures",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data and fitted preprocessor
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path)
    feature_names = get_feature_names(preprocessor)

    # 2. Load the trained model artifact
    model = joblib.load(model_path)

    # 3. Built-in Gini Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features

    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Feature Importances (Gini Impurity Reduction)", fontsize=13, fontweight="bold")
    plt.barh(range(len(indices)), importances[indices][::-1], align="center", color="#1f77b4")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    
    gini_plot_file = output_path / "rf_gini_feature_importance.png"
    plt.savefig(gini_plot_file, dpi=300)
    plt.close()
    print(f"[Artifact] Gini Feature Importance saved to: {gini_plot_file.resolve()}")

    # 4. SHAP (TreeExplainer)
    print("\nComputing TreeSHAP values on test set...")
    explainer = shap.TreeExplainer(model)
    
    # SHAP values for class 1 (Churn)
    shap_values = explainer.shap_values(X_test)
    
    # Random Forest shap_values returns a list [class_0, class_1] or a 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        churn_shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        churn_shap_values = shap_values[:, :, 1]
    else:
        churn_shap_values = shap_values

    # SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        churn_shap_values,
        features=X_test,
        feature_names=feature_names,
        max_display=12,
        show=False,
    )
    plt.title("SHAP Beeswarm Summary Plot (Class 1 - Churn)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    shap_plot_file = output_path / "shap_summary_beeswarm.png"
    plt.savefig(shap_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifact] SHAP Beeswarm Plot saved to: {shap_plot_file.resolve()}")

    # 5. Export preprocessor alongside model for production inference
    prep_path = Path("models") / "preprocessor.joblib"
    joblib.dump(preprocessor, prep_path)
    print(f"[Artifact] Saved preprocessor pipeline to: {prep_path.resolve()}")


if __name__ == "__main__":
    run_explainability_pipeline()