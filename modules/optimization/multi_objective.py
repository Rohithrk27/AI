"""
Multi-Objective Optimization Module
Pareto front identification and multi-objective Bayesian Optimization.
"""

import numpy as np
import pandas as pd


def is_dominated(point, other_points, directions):
    """
    Check if a point is dominated by any other point.
    
    Args:
        point: Single objective vector
        other_points: Array of objective vectors
        directions: List of "maximize" or "minimize" for each objective
        
    Returns:
        bool: True if point is dominated
    """
    for other in other_points:
        if np.array_equal(point, other):
            continue

        dominated = True
        strictly_better = False

        for i, direction in enumerate(directions):
            if direction == "maximize":
                if other[i] < point[i]:
                    dominated = False
                    break
                if other[i] > point[i]:
                    strictly_better = True
            else:  # minimize
                if other[i] > point[i]:
                    dominated = False
                    break
                if other[i] < point[i]:
                    strictly_better = True

        if dominated and strictly_better:
            return True

    return False


def pareto_front(objectives: np.ndarray, directions: list = None) -> np.ndarray:
    """
    Identify the Pareto-optimal (non-dominated) solutions.
    
    Args:
        objectives: (n_points, n_objectives) array
        directions: List of "maximize" or "minimize" per objective.
                    Default: all "maximize"
                    
    Returns:
        np.ndarray: Boolean mask of Pareto-optimal points
    """
    n = len(objectives)

    if directions is None:
        directions = ["maximize"] * objectives.shape[1]

    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto_mask[i]:
            continue
        for j in range(n):
            if i == j or not pareto_mask[j]:
                continue
            if is_dominated(objectives[i], objectives[j:j+1], directions):
                pareto_mask[i] = False
                break

    return pareto_mask


def multi_objective_recommend(models, bounds, feature_names, objectives_config,
                               n_recommend=5, n_candidates=20000):
    """
    Multi-objective optimization using scalarization approach.
    
    Args:
        models: List of trained SurrogateModels (one per objective)
        bounds: List of (min, max) tuples
        feature_names: List of feature names
        objectives_config: List of dicts:
            [{"name": "Yield", "direction": "maximize", "weight": 1.0},
             {"name": "Cost", "direction": "minimize", "weight": 0.5}]
        n_recommend: Number of experiments to suggest
        n_candidates: Candidate pool size
        
    Returns:
        pd.DataFrame: Recommended experiments with predicted values for all objectives
    """
    from modules.optimization.bayesian_opt import generate_candidates

    candidates = generate_candidates(bounds, n_candidates)

    # Predict each objective
    all_predictions = []
    for model in models:
        pred_mean, _ = model.predict_with_uncertainty(candidates)
        all_predictions.append(pred_mean)

    predictions = np.column_stack(all_predictions)

    # Normalize predictions to [0, 1] for fair comparison
    pred_normalized = np.zeros_like(predictions)
    for i in range(predictions.shape[1]):
        col = predictions[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            pred_normalized[:, i] = (col - col_min) / (col_max - col_min)
        else:
            pred_normalized[:, i] = 0.5

    # Apply direction (flip for minimize so higher = better)
    for i, config in enumerate(objectives_config):
        if config.get("direction", "maximize") == "minimize":
            pred_normalized[:, i] = 1 - pred_normalized[:, i]

    # Weighted sum scalarization
    weights = np.array([config.get("weight", 1.0) for config in objectives_config])
    weights = weights / weights.sum()  # Normalize weights

    scalarized = pred_normalized @ weights

    # Get top candidates
    top_indices = np.argsort(scalarized)[::-1][:n_recommend * 3]  # Get more for Pareto filtering

    # Extract objectives for top candidates
    top_objectives = predictions[top_indices]

    # Find Pareto front among top candidates
    directions = [config.get("direction", "maximize") for config in objectives_config]
    pareto_mask = pareto_front(top_objectives, directions)

    # Select from Pareto front first, then fill with remaining
    pareto_indices = top_indices[pareto_mask]
    non_pareto_indices = top_indices[~pareto_mask]

    selected = list(pareto_indices[:n_recommend])
    if len(selected) < n_recommend:
        remaining = n_recommend - len(selected)
        selected.extend(non_pareto_indices[:remaining])

    # Build results
    results = []
    for rank, idx in enumerate(selected[:n_recommend], 1):
        row = {"Rank": rank}
        for j, name in enumerate(feature_names):
            row[name] = round(candidates[idx, j], 4)
        for k, config in enumerate(objectives_config):
            row[f"Predicted_{config['name']}"] = round(predictions[idx, k], 4)
        row["Pareto_Optimal"] = idx in pareto_indices
        row["Scalarized_Score"] = round(scalarized[idx], 4)
        results.append(row)

    return pd.DataFrame(results)


def plot_pareto(objectives: np.ndarray, pareto_mask: np.ndarray,
                obj_names: list = None):
    """
    Create a Pareto front visualization.
    
    Args:
        objectives: (n_points, n_objectives) array
        pareto_mask: Boolean mask of Pareto-optimal points
        obj_names: Names of objectives
        
    Returns:
        plotly.Figure
    """
    import plotly.graph_objects as go

    if obj_names is None:
        obj_names = [f"Objective {i+1}" for i in range(objectives.shape[1])]

    fig = go.Figure()

    # Non-Pareto points
    fig.add_trace(go.Scatter(
        x=objectives[~pareto_mask, 0],
        y=objectives[~pareto_mask, 1],
        mode="markers",
        marker=dict(size=8, color="#4a5568", opacity=0.5),
        name="Dominated",
    ))

    # Pareto points
    pareto_pts = objectives[pareto_mask]
    # Sort by first objective for line
    sort_idx = np.argsort(pareto_pts[:, 0])
    pareto_sorted = pareto_pts[sort_idx]

    fig.add_trace(go.Scatter(
        x=pareto_sorted[:, 0],
        y=pareto_sorted[:, 1],
        mode="markers+lines",
        marker=dict(size=12, color="#667eea", line=dict(width=2, color="white")),
        line=dict(color="#667eea", dash="dash"),
        name="Pareto Front",
    ))

    fig.update_layout(
        title="Pareto Front",
        xaxis_title=obj_names[0],
        yaxis_title=obj_names[1] if len(obj_names) > 1 else "Objective 2",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        font=dict(family="Inter", color="#e0e0e0"),
    )

    return fig
