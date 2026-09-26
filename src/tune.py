# src/tune.py

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.preprocessing import prepare_data


def tune_random_forest(data_path: str = "data/processed/telco_cleaned.csv"):
    # 1. Fetch splits (test set remains untouched during tuning)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path)

    # 2. Define parameter search space
    param_dist = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [5, 7, 9, 12],
        "min_samples_split": [5, 10, 15, 20],
        "min_samples_leaf": [2, 4, 8, 12],
        "class_weight": ["balanced", {0: 1, 1: 2.5}],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # 3. Stratified 5-Fold Cross-Validation targeting ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("\nStarting Hyperparameter Search (20 iterations x 5 folds = 100 fits)...")
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_
    print(f"\nBest Cross-Validation ROC-AUC: {search.best_score_:.4f}")
    print(f"Optimal Hyperparameters: {search.best_params_}")

    # 4. Final Evaluation on Untouched Test Set
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nFinal Untouched Test Set ROC-AUC: {test_roc_auc:.4f}")
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed (0)", "Churned (1)"]))

    # 5. Export Model Artifact
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_random_forest.joblib"
    joblib.dump(best_rf, model_path)
    print(f"[Artifact] Exported tuned model to: {model_path.resolve()}")

    return best_rf, preprocessor


if __name__ == "__main__":
    tune_random_forest()
    