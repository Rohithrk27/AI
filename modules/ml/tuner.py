"""
Optuna Hyperparameter Tuner Module
Auto-tunes hyperparameters for all 7 ML models using Optuna.
"""

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def define_search_space(trial: optuna.Trial, model_name: str) -> dict:
    """
    Define Optuna search space for each model type.
    
    Args:
        trial: Optuna trial
        model_name: Name of the model
        
    Returns:
        dict: Hyperparameter configuration
    """
    if model_name == "Gaussian Process":
        return {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "matern32", "matern52", "rational_quadratic"]),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-1, log=True),
            "n_restarts": trial.suggest_int("n_restarts", 5, 20),
        }

    elif model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20) if trial.suggest_categorical("use_max_depth", [True, False]) else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }

    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    elif model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        }

    elif model_name == "Neural Network":
        return {
            "hidden_layers": trial.suggest_int("hidden_layers", 1, 4),
            "hidden_units": trial.suggest_int("hidden_units", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "epochs": trial.suggest_int("epochs", 50, 300),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }

    elif model_name == "SVR":
        return {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.001, 1.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    return {}


def _create_sklearn_model(model_name: str, params: dict):
    """Create a scikit-learn compatible model for cross-validation."""
    if model_name == "Gaussian Process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel, RationalQuadratic

        kernel_name = params.get("kernel", "rbf")
        if kernel_name == "matern32":
            kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
        elif kernel_name == "matern52":
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
        elif kernel_name == "rational_quadratic":
            kernel = ConstantKernel() * RationalQuadratic() + WhiteKernel()
        else:
            kernel = ConstantKernel() * RBF() + WhiteKernel()

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=params.get("alpha", 1e-6),
            n_restarts_optimizer=params.get("n_restarts", 10),
            random_state=42,
        )

    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", 1.0),
            random_state=42, n_jobs=-1,
        )

    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=42, verbosity=0,
        )

    elif model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            num_leaves=int(params.get("num_leaves", 31)),
            learning_rate=params.get("learning_rate", 0.1),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42, verbosity=-1,
        )

    elif model_name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=int(params.get("iterations", 200)),
            depth=int(params.get("depth", 6)),
            learning_rate=params.get("learning_rate", 0.1),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            random_state=42, verbose=0,
        )

    elif model_name == "SVR":
        from sklearn.svm import SVR
        return SVR(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 10.0),
            epsilon=params.get("epsilon", 0.1),
            gamma=params.get("gamma", "scale"),
        )

    return None


def tune_model(model_name: str, X, y, n_trials: int = 50, cv_folds: int = 5, timeout: int = 60) -> dict:
    """
    Tune hyperparameters for a single model using Optuna.
    
    Args:
        model_name: Name of the model to tune
        X: Feature matrix
        y: Target vector
        n_trials: Number of Optuna trials
        cv_folds: Cross-validation folds
        timeout: Max time in seconds
        
    Returns:
        dict: {best_params, best_cv_score, study_summary, optimization_history}
    """
    # For ANN, we use a different approach (no sklearn cross_val_score)
    if model_name == "Neural Network":
        return _tune_ann(X, y, n_trials, cv_folds, timeout)

    def objective(trial):
        params = define_search_space(trial, model_name)
        try:
            sklearn_model = _create_sklearn_model(model_name, params)
            if sklearn_model is None:
                return -1.0

            # Adjust CV folds if dataset is too small
            actual_folds = min(cv_folds, len(X))
            if actual_folds < 2:
                actual_folds = 2

            scores = cross_val_score(sklearn_model, X, y, cv=actual_folds, scoring="r2")
            return scores.mean()
        except Exception:
            return -1.0

    study = optuna.create_study(direction="maximize", study_name=model_name)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # Collect results
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "trial": t.number,
            "value": t.value if t.value is not None else -1.0,
            "params": t.params,
        })

    return {
        "best_params": study.best_params if study.best_trial else {},
        "best_cv_score": study.best_value if study.best_trial else -1.0,
        "study_summary": trials_data,
        "optimization_history": [t["value"] for t in trials_data],
    }


def _tune_ann(X, y, n_trials, cv_folds, timeout):
    """Tune ANN using manual cross-validation with PyTorch."""
    from modules.ml.models import ANNModel
    from sklearn.model_selection import KFold

    def objective(trial):
        params = define_search_space(trial, "Neural Network")
        try:
            actual_folds = min(cv_folds, len(X))
            if actual_folds < 2:
                actual_folds = 2

            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = ANNModel()
                model.train(X_train, y_train, params=params)
                y_pred = model.predict(X_val)

                # R² score
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                scores.append(r2)

            return np.mean(scores)
        except Exception:
            return -1.0

    study = optuna.create_study(direction="maximize", study_name="Neural Network")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "trial": t.number,
            "value": t.value if t.value is not None else -1.0,
            "params": t.params,
        })

    return {
        "best_params": study.best_params if study.best_trial else {},
        "best_cv_score": study.best_value if study.best_trial else -1.0,
        "study_summary": trials_data,
        "optimization_history": [t["value"] for t in trials_data],
    }


def tune_all_models(models: dict, X, y, n_trials: int = 50, cv_folds: int = 5,
                     timeout: int = 60, progress_callback=None) -> dict:
    """
    Tune hyperparameters for all provided models.
    
    Args:
        models: Dict of {model_name: SurrogateModel}
        X: Feature matrix
        y: Target vector
        n_trials: Trials per model
        cv_folds: Cross-validation folds
        timeout: Max time per model in seconds
        progress_callback: Optional callable(model_name, progress_fraction)
        
    Returns:
        dict: {model_name: tuning_results}
    """
    results = {}
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        try:
            result = tune_model(name, X, y, n_trials=n_trials, cv_folds=cv_folds, timeout=timeout)
            results[name] = result
        except Exception as e:
            results[name] = {
                "best_params": {},
                "best_cv_score": -1.0,
                "error": str(e),
            }

        if progress_callback:
            progress_callback(name, (i + 1) / total)

    return results
