"""
Visualization Module
All interactive charts for the OptiML platform using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────
# Common theme
# ──────────────────────────────────────────────
DARK_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.8)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
)

COLORS = ["#667eea", "#f093fb", "#4facfe", "#43e97b", "#f5af19", "#ff6b6b", "#38f9d7"]


def _apply_theme(fig):
    """Apply dark theme to a plotly figure."""
    fig.update_layout(**DARK_THEME)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


# ──────────────────────────────────────────────
# 3D Response Surface
# ──────────────────────────────────────────────
def plot_3d_surface(model, X, feature_names, factor1, factor2, resolution=50):
    """
    Interactive 3D response surface plot for two selected factors.
    Other factors held at their mean values.
    """
    f1_idx = feature_names.index(factor1)
    f2_idx = feature_names.index(factor2)

    f1_range = np.linspace(X[:, f1_idx].min(), X[:, f1_idx].max(), resolution)
    f2_range = np.linspace(X[:, f2_idx].min(), X[:, f2_idx].max(), resolution)
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

    # Build prediction grid (other factors at mean)
    grid_points = np.tile(X.mean(axis=0), (resolution * resolution, 1))
    grid_points[:, f1_idx] = f1_grid.ravel()
    grid_points[:, f2_idx] = f2_grid.ravel()

    z_pred = model.predict(grid_points).reshape(resolution, resolution)

    fig = go.Figure(data=[
        go.Surface(
            x=f1_grid, y=f2_grid, z=z_pred,
            colorscale=[[0, "#667eea"], [0.5, "#f093fb"], [1, "#ff6b6b"]],
            opacity=0.85,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
            ),
        ),
    ])

    # Add experimental points
    fig.add_trace(go.Scatter3d(
        x=X[:, f1_idx], y=X[:, f2_idx],
        z=model.predict(X),
        mode="markers",
        marker=dict(size=5, color="#43e97b", line=dict(width=1, color="white")),
        name="Experiments",
    ))

    fig.update_layout(
        title=f"Response Surface: {factor1} × {factor2}",
        scene=dict(
            xaxis_title=factor1,
            yaxis_title=factor2,
            zaxis_title="Response",
            bgcolor="rgba(14,17,23,0.8)",
        ),
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Contour Plot
# ──────────────────────────────────────────────
def plot_contour(model, X, feature_names, factor1, factor2, resolution=80):
    """
    2D contour plot for two selected factors with optimal point marked.
    """
    f1_idx = feature_names.index(factor1)
    f2_idx = feature_names.index(factor2)

    f1_range = np.linspace(X[:, f1_idx].min(), X[:, f1_idx].max(), resolution)
    f2_range = np.linspace(X[:, f2_idx].min(), X[:, f2_idx].max(), resolution)
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

    grid_points = np.tile(X.mean(axis=0), (resolution * resolution, 1))
    grid_points[:, f1_idx] = f1_grid.ravel()
    grid_points[:, f2_idx] = f2_grid.ravel()

    z_pred = model.predict(grid_points).reshape(resolution, resolution)

    fig = go.Figure(data=[
        go.Contour(
            x=f1_range, y=f2_range, z=z_pred,
            colorscale=[[0, "#667eea"], [0.5, "#f093fb"], [1, "#ff6b6b"]],
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            line=dict(smoothing=0.85),
        ),
    ])

    # Mark experimental points
    fig.add_trace(go.Scatter(
        x=X[:, f1_idx], y=X[:, f2_idx],
        mode="markers",
        marker=dict(size=8, color="#43e97b", line=dict(width=1, color="white")),
        name="Experiments",
    ))

    # Mark optimal point
    opt_idx = z_pred.argmax()
    opt_i, opt_j = np.unravel_index(opt_idx, z_pred.shape)
    fig.add_trace(go.Scatter(
        x=[f1_range[opt_j]], y=[f2_range[opt_i]],
        mode="markers",
        marker=dict(size=15, symbol="star", color="#f5af19", line=dict(width=2, color="white")),
        name=f"Predicted Optimum ({z_pred.max():.2f})",
    ))

    fig.update_layout(
        title=f"Contour Plot: {factor1} × {factor2}",
        xaxis_title=factor1,
        yaxis_title=factor2,
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Predicted vs Actual
# ──────────────────────────────────────────────
def plot_pred_vs_actual(y_actual, y_pred):
    """Scatter plot of predicted vs actual values with R² and 1:1 line."""
    from sklearn.metrics import r2_score

    r2 = r2_score(y_actual, y_pred)

    fig = go.Figure()

    # 1:1 reference line
    all_vals = np.concatenate([y_actual, y_pred])
    line_min, line_max = all_vals.min() * 0.95, all_vals.max() * 1.05
    fig.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max],
        mode="lines",
        line=dict(color="#4a5568", dash="dash", width=2),
        name="Perfect Prediction",
        showlegend=True,
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=y_actual, y=y_pred,
        mode="markers",
        marker=dict(size=10, color="#667eea", line=dict(width=1, color="white"), opacity=0.8),
        name=f"Predictions (R² = {r2:.4f})",
    ))

    fig.update_layout(
        title=f"Predicted vs Actual (R² = {r2:.4f})",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Residuals
# ──────────────────────────────────────────────
def plot_residuals(y_actual, y_pred):
    """Residuals plot with distribution."""
    residuals = y_actual - y_pred

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals vs Predicted", "Residual Distribution"))

    # Residuals vs predicted
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(size=8, color="#f093fb", opacity=0.8),
        name="Residuals",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#4a5568", row=1, col=1)

    # Distribution
    fig.add_trace(go.Histogram(
        y=residuals,
        marker_color="#667eea",
        opacity=0.7,
        name="Distribution",
    ), row=1, col=2)

    fig.update_layout(
        title="Residual Analysis",
        showlegend=False,
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Model Comparison
# ──────────────────────────────────────────────
def plot_model_comparison(results: dict):
    """Grouped bar chart comparing all models across metrics."""
    models = []
    r2_vals = []
    cv_r2_vals = []
    rmse_vals = []

    for name, metrics in results.items():
        if "error" in metrics:
            continue
        models.append(name)
        r2_vals.append(metrics.get("R²", 0))
        cv_r2_vals.append(metrics.get("CV_R²_mean", 0))
        rmse_vals.append(metrics.get("RMSE", 0))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Comparison", "RMSE Comparison"))

    fig.add_trace(go.Bar(
        x=models, y=r2_vals,
        name="Training R²",
        marker_color="#667eea",
        opacity=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=models, y=cv_r2_vals,
        name="CV R²",
        marker_color="#f093fb",
        opacity=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=models, y=rmse_vals,
        name="RMSE",
        marker_color="#4facfe",
        opacity=0.8,
    ), row=1, col=2)

    fig.update_layout(
        title="Model Benchmarking",
        barmode="group",
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Convergence Plot
# ──────────────────────────────────────────────
def plot_convergence(history: list):
    """Plot best value found vs BO iteration."""
    iterations = [h["iteration"] for h in history]
    best_values = [h["best_value"] for h in history]
    n_experiments = [h.get("n_experiments", 0) for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iterations, y=best_values,
        mode="lines+markers",
        line=dict(color="#667eea", width=3),
        marker=dict(size=10, color="#667eea", line=dict(width=2, color="white")),
        name="Best Value",
        fill="tozeroy",
        fillcolor="rgba(102,126,234,0.1)",
    ))

    fig.update_layout(
        title="Optimization Convergence",
        xaxis_title="Iteration",
        yaxis_title="Best Response Value",
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Main Effects
# ──────────────────────────────────────────────
def plot_main_effects(model, X, feature_names, resolution=50):
    """One-at-a-time factor effect curves."""
    n_features = len(feature_names)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=feature_names,
    )

    for i, name in enumerate(feature_names):
        row = i // cols + 1
        col = i % cols + 1

        # Vary this factor, keep others at mean
        x_range = np.linspace(X[:, i].min(), X[:, i].max(), resolution)
        grid = np.tile(X.mean(axis=0), (resolution, 1))
        grid[:, i] = x_range

        y_pred = model.predict(grid)

        fig.add_trace(go.Scatter(
            x=x_range, y=y_pred,
            mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=3),
            name=name,
            showlegend=False,
        ), row=row, col=col)

    fig.update_layout(
        title="Main Effects (One-at-a-Time)",
        height=300 * rows,
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Uncertainty Heatmap
# ──────────────────────────────────────────────
def plot_uncertainty(model, X, feature_names, factor1=None, factor2=None, resolution=60):
    """Heatmap showing model prediction uncertainty."""
    if factor1 is None:
        factor1 = feature_names[0]
    if factor2 is None:
        factor2 = feature_names[1] if len(feature_names) > 1 else feature_names[0]

    f1_idx = feature_names.index(factor1)
    f2_idx = feature_names.index(factor2)

    f1_range = np.linspace(X[:, f1_idx].min(), X[:, f1_idx].max(), resolution)
    f2_range = np.linspace(X[:, f2_idx].min(), X[:, f2_idx].max(), resolution)
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

    grid_points = np.tile(X.mean(axis=0), (resolution * resolution, 1))
    grid_points[:, f1_idx] = f1_grid.ravel()
    grid_points[:, f2_idx] = f2_grid.ravel()

    _, std = model.predict_with_uncertainty(grid_points)
    std_grid = std.reshape(resolution, resolution)

    fig = go.Figure(data=[
        go.Heatmap(
            x=f1_range, y=f2_range, z=std_grid,
            colorscale=[[0, "#0e1117"], [0.5, "#667eea"], [1, "#ff6b6b"]],
            colorbar=dict(title="Uncertainty (σ)"),
        ),
    ])

    # Mark experimental points
    fig.add_trace(go.Scatter(
        x=X[:, f1_idx], y=X[:, f2_idx],
        mode="markers",
        marker=dict(size=8, color="#43e97b", line=dict(width=1, color="white")),
        name="Experiments",
    ))

    fig.update_layout(
        title=f"Prediction Uncertainty: {factor1} × {factor2}",
        xaxis_title=factor1,
        yaxis_title=factor2,
        **DARK_THEME,
    )

    return fig


# ──────────────────────────────────────────────
# Acquisition Landscape
# ──────────────────────────────────────────────
def plot_acquisition_landscape(model, X, feature_names, acq_func="EI",
                                factor1=None, factor2=None, resolution=60):
    """Visualize the acquisition function landscape."""
    from modules.optimization.acquisition import get_acquisition_function

    if factor1 is None:
        factor1 = feature_names[0]
    if factor2 is None:
        factor2 = feature_names[1] if len(feature_names) > 1 else feature_names[0]

    f1_idx = feature_names.index(factor1)
    f2_idx = feature_names.index(factor2)

    f1_range = np.linspace(X[:, f1_idx].min(), X[:, f1_idx].max(), resolution)
    f2_range = np.linspace(X[:, f2_idx].min(), X[:, f2_idx].max(), resolution)
    f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

    grid_points = np.tile(X.mean(axis=0), (resolution * resolution, 1))
    grid_points[:, f1_idx] = f1_grid.ravel()
    grid_points[:, f2_idx] = f2_grid.ravel()

    acq_fn = get_acquisition_function(acq_func)
    y_best = model.predict(X).max()

    if acq_func == "UCB":
        acq_values = acq_fn(grid_points, model)
    else:
        acq_values = acq_fn(grid_points, model, y_best)

    acq_grid = acq_values.reshape(resolution, resolution)

    fig = go.Figure(data=[
        go.Contour(
            x=f1_range, y=f2_range, z=acq_grid,
            colorscale=[[0, "#0e1117"], [0.5, "#4facfe"], [1, "#43e97b"]],
            contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
        ),
    ])

    # Mark existing experiments
    fig.add_trace(go.Scatter(
        x=X[:, f1_idx], y=X[:, f2_idx],
        mode="markers",
        marker=dict(size=8, color="#ff6b6b", symbol="x", line=dict(width=1)),
        name="Completed Experiments",
    ))

    # Mark peak of acquisition
    peak_idx = acq_values.argmax()
    peak_i, peak_j = np.unravel_index(peak_idx, acq_grid.shape)
    fig.add_trace(go.Scatter(
        x=[f1_range[peak_j]], y=[f2_range[peak_i]],
        mode="markers",
        marker=dict(size=15, symbol="star", color="#f5af19", line=dict(width=2, color="white")),
        name="Next Best Experiment",
    ))

    acq_labels = {"EI": "Expected Improvement", "UCB": "Upper Confidence Bound", "PI": "Probability of Improvement"}

    fig.update_layout(
        title=f"{acq_labels.get(acq_func, acq_func)} Landscape",
        xaxis_title=factor1,
        yaxis_title=factor2,
        **DARK_THEME,
    )

    return fig
