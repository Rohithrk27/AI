"""
Bayesian Optimization Engine
Generates candidates, evaluates acquisition functions, and recommends next experiments.
"""

import numpy as np
import pandas as pd
from modules.optimization.acquisition import get_acquisition_function


def generate_candidates(bounds, n_candidates=20000, random_state=42):
    """
    Generate candidate points using Latin Hypercube Sampling.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_candidates: Number of candidate points
        random_state: Random seed
        
    Returns:
        np.ndarray: (n_candidates, n_dimensions)
    """
    rng = np.random.RandomState(random_state)
    n_dims = len(bounds)

    # Latin Hypercube Sampling
    # Divide each dimension into n_candidates equal intervals
    # Sample one point per interval, then shuffle
    result = np.zeros((n_candidates, n_dims))

    for d in range(n_dims):
        low, high = bounds[d]
        # Create evenly spaced intervals
        intervals = np.linspace(low, high, n_candidates + 1)
        # Sample uniformly within each interval
        points = rng.uniform(intervals[:-1], intervals[1:])
        # Shuffle
        rng.shuffle(points)
        result[:, d] = points

    return result


def recommend_experiments(model, bounds, feature_names, y_best,
                           acq_func="EI", n_recommend=5, n_candidates=20000,
                           minimize=False):
    """
    Recommend next experiments using Bayesian Optimization.
    
    Args:
        model: Trained SurrogateModel
        bounds: List of (min, max) tuples
        feature_names: List of feature names
        y_best: Best observed value
        acq_func: "EI", "UCB", or "PI"
        n_recommend: Number of experiments to suggest
        n_candidates: Candidate pool size
        minimize: Whether to minimize (default: maximize)
        
    Returns:
        pd.DataFrame with recommended experiments
    """
    # Generate candidates
    candidates = generate_candidates(bounds, n_candidates)

    # Evaluate acquisition function
    acq_fn = get_acquisition_function(acq_func)

    if acq_func == "UCB":
        acq_values = acq_fn(candidates, model, minimize=minimize)
    else:
        acq_values = acq_fn(candidates, model, y_best, minimize=minimize)

    # Get predictions and uncertainty
    pred_mean, pred_std = model.predict_with_uncertainty(candidates)

    # Rank by acquisition value
    top_indices = np.argsort(acq_values)[::-1][:n_recommend]

    # Build results DataFrame
    results = []
    for rank, idx in enumerate(top_indices, 1):
        row = {"Rank": rank}
        for j, name in enumerate(feature_names):
            row[name] = round(candidates[idx, j], 4)
        row["Predicted_Response"] = round(pred_mean[idx], 4)
        row["Uncertainty (±)"] = round(pred_std[idx], 4)
        row[f"{acq_func}_Score"] = round(acq_values[idx], 6)
        results.append(row)

    return pd.DataFrame(results)


def optimization_step(model, X_data, y_data, new_X, new_y, bounds, feature_names,
                       acq_func="EI", n_recommend=5, minimize=False):
    """
    Perform one optimization step: add new data, retrain, recommend.
    
    Args:
        model: SurrogateModel to retrain
        X_data: Existing feature data
        y_data: Existing response data
        new_X: New experiment features
        new_y: New experiment responses
        bounds: Factor bounds
        feature_names: Feature names
        acq_func: Acquisition function
        n_recommend: Number of recommendations
        minimize: Whether to minimize
        
    Returns:
        dict: {updated_X, updated_y, recommendations, new_best}
    """
    # Combine data
    updated_X = np.vstack([X_data, new_X])
    updated_y = np.concatenate([y_data, new_y])

    # Retrain model
    model.train(updated_X, updated_y, params=model.params)

    # New best
    if minimize:
        new_best = updated_y.min()
    else:
        new_best = updated_y.max()

    # Get new recommendations
    recommendations = recommend_experiments(
        model, bounds, feature_names, new_best,
        acq_func=acq_func, n_recommend=n_recommend, minimize=minimize,
    )

    return {
        "updated_X": updated_X,
        "updated_y": updated_y,
        "recommendations": recommendations,
        "new_best": new_best,
    }
