"""
Model Selector Module
Automatically selects the best-performing model based on cross-validated metrics.
"""

import numpy as np


def auto_select_best(results: dict, trained_models: dict, metric: str = "CV_R²_mean") -> tuple:
    """
    Select the best model based on a specified metric.
    
    Args:
        results: Dict of {model_name: metrics_dict}
        trained_models: Dict of {model_name: trained SurrogateModel}
        metric: Metric to optimize (default: CV_R²_mean)
        
    Returns:
        tuple: (best_model_name, best_model_object)
    """
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, metrics in results.items():
        if "error" in metrics:
            continue

        score = metrics.get(metric, metrics.get("R²", -np.inf))

        if score > best_score:
            best_score = score
            best_name = name
            best_model = trained_models.get(name)

    # If no model was selected (all failed), try to pick any trained model
    if best_model is None and trained_models:
        best_name = next(iter(trained_models))
        best_model = trained_models[best_name]

    return best_name, best_model
