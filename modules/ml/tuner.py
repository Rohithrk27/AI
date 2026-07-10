import optuna
import numpy as np
from sklearn.model_selection import cross_val_score

def define_search_space(trial, model_name: str, n_samples: int = None) -> dict:
    """Define the Optuna search space for Gaussian Process."""
    if model_name == "Gaussian Process":
        return {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "matern32", "matern52", "rational_quadratic"]),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-2, log=True),
            "n_restarts": trial.suggest_int("n_restarts", 2, 10),
        }
    return {}


def _create_sklearn_model(model_name: str, params: dict):
    """Create raw scikit-learn model with params for CV evaluation."""
    if model_name == "Gaussian Process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, ConstantKernel, WhiteKernel
        
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
    return None


def tune_model(model_name: str, X, y, n_trials: int = 50, cv_folds: int = 5, timeout: int = 60) -> dict:
    """Tune hyperparameters for a single model using Optuna."""
    def objective(trial):
        params = define_search_space(trial, model_name)
        try:
            sklearn_model = _create_sklearn_model(model_name, params)
            if sklearn_model is None:
                return -1.0

            actual_folds = min(cv_folds, len(X))
            if actual_folds < 2:
                actual_folds = 2

            scores = cross_val_score(sklearn_model, X, y, cv=actual_folds, scoring="r2")
            return scores.mean()
        except Exception:
            return -1.0

    study = optuna.create_study(direction="maximize", study_name=model_name)
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
    """Tune hyperparameters for all provided models."""
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
