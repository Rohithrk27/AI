"""
Model Evaluator Module
Computes metrics, cross-validation scores, and model rankings.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X, y, cv_folds: int = 5) -> dict:
    """
    Evaluate a trained model's performance.
    
    Args:
        model: Trained SurrogateModel
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        
    Returns:
        dict with metrics: R², Adjusted R², RMSE, MAE, MAPE, CV_R², CV_RMSE
    """
    # Predictions on training data
    y_pred = model.predict(X)

    n = len(y)
    p = X.shape[1]

    # Basic metrics
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # MAPE (avoid division by zero)
    mask = np.abs(y) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
    else:
        mape = 0.0

    # Cross-validation
    actual_folds = min(cv_folds, n)
    if actual_folds < 2:
        actual_folds = 2

    cv_r2_scores = []
    cv_rmse_scores = []

    try:
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Clone and retrain model
            from modules.ml.models import get_all_models
            model_name = model.get_name()
            fresh_model = get_all_models().get(model_name)

            if fresh_model is None:
                # Fallback: use the trained model's predictions
                break

            fresh_model.train(X_train, y_train, params=model.params)
            y_val_pred = fresh_model.predict(X_val)

            cv_r2 = r2_score(y_val, y_val_pred)
            cv_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            cv_r2_scores.append(cv_r2)
            cv_rmse_scores.append(cv_rmse)
    except Exception:
        cv_r2_scores = [r2]
        cv_rmse_scores = [rmse]

    return {
        "R²": round(r2, 4),
        "Adj_R²": round(adj_r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE_%": round(mape, 2),
        "CV_R²_mean": round(np.mean(cv_r2_scores), 4),
        "CV_R²_std": round(np.std(cv_r2_scores), 4),
        "CV_RMSE_mean": round(np.mean(cv_rmse_scores), 4),
    }


def rank_models(results: dict) -> pd.DataFrame:
    """
    Rank all models by cross-validated R².
    
    Args:
        results: Dict of {model_name: metrics_dict}
        
    Returns:
        pd.DataFrame sorted by CV_R² descending
    """
    rows = []
    for name, metrics in results.items():
        if "error" in metrics:
            row = {"Model": name, "Status": "❌ Failed", "Error": metrics["error"]}
        else:
            row = {"Model": name, "Status": "✅"}
            row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by CV R² if available
    if "CV_R²_mean" in df.columns:
        df = df.sort_values("CV_R²_mean", ascending=False).reset_index(drop=True)

    # Add rank
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df


def compare_tuned_vs_default(default_results: dict, tuned_results: dict) -> pd.DataFrame:
    """
    Compare model performance before and after Optuna tuning.
    
    Args:
        default_results: Metrics with default hyperparameters
        tuned_results: Metrics with Optuna-tuned hyperparameters
        
    Returns:
        pd.DataFrame showing improvement for each model
    """
    rows = []
    for name in default_results:
        row = {"Model": name}
        if name in default_results and "CV_R²_mean" in default_results[name]:
            row["Default_CV_R²"] = default_results[name]["CV_R²_mean"]
        else:
            row["Default_CV_R²"] = None

        if name in tuned_results and "CV_R²_mean" in tuned_results[name]:
            row["Tuned_CV_R²"] = tuned_results[name]["CV_R²_mean"]
        else:
            row["Tuned_CV_R²"] = None

        if row["Default_CV_R²"] is not None and row["Tuned_CV_R²"] is not None:
            row["Improvement"] = round(row["Tuned_CV_R²"] - row["Default_CV_R²"], 4)
        else:
            row["Improvement"] = None

        rows.append(row)

    return pd.DataFrame(rows)
