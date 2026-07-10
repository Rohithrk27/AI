"""
SHAP Explainability Module
Computes SHAP values and generates feature importance visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")


def compute_shap_values(model, X, feature_names):
    """
    Compute SHAP values for a trained model.
    
    Auto-selects the appropriate SHAP explainer:
    - TreeExplainer for tree-based models (RF, XGBoost, LightGBM, CatBoost)
    - KernelExplainer for others (GP, ANN, SVR)
    
    Args:
        model: Trained SurrogateModel
        X: Feature matrix
        feature_names: List of feature names
        
    Returns:
        shap.Explanation: SHAP values object
    """
    model_name = model.get_name()
    inner_model = model.model

    # Tree-based models
    tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]

    if model_name in tree_models and inner_model is not None:
        try:
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X)

            return shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value,
                data=X,
                feature_names=feature_names,
            )
        except Exception:
            pass  # Fall through to KernelExplainer

    # KernelExplainer for all other models
    def predict_fn(x):
        return model.predict(x)

    # Use a subset for background (KernelExplainer is slow)
    n_background = min(50, len(X))
    background = shap.sample(X, n_background)

    explainer = shap.KernelExplainer(predict_fn, background)

    # Compute SHAP values (limit samples for speed)
    n_explain = min(100, len(X))
    X_explain = X[:n_explain]

    shap_values = explainer.shap_values(X_explain, nsamples=100)

    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_explain,
        feature_names=feature_names,
    )


def shap_summary_plot(shap_values, X, feature_names):
    """
    Generate SHAP beeswarm summary plot.
    
    Args:
        shap_values: shap.Explanation object
        X: Feature matrix
        feature_names: Feature names
        
    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)

    shap.summary_plot(
        shap_values.values if hasattr(shap_values, 'values') else shap_values,
        X if not hasattr(shap_values, 'data') else shap_values.data,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )

    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#2d3748")

    plt.tight_layout()
    return fig


def shap_dependence_plot(shap_values, feature, X, feature_names):
    """
    Generate SHAP dependence plot for a single feature.
    
    Args:
        shap_values: shap.Explanation object
        feature: Feature name or index
        X: Feature matrix
        feature_names: Feature names
        
    Returns:
        matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)

    # Get feature index
    if isinstance(feature, str):
        feat_idx = feature_names.index(feature) if feature in feature_names else 0
    else:
        feat_idx = feature

    vals = shap_values.values if hasattr(shap_values, 'values') else shap_values
    data = shap_values.data if hasattr(shap_values, 'data') else X

    shap.dependence_plot(
        feat_idx,
        vals,
        data,
        feature_names=feature_names,
        show=False,
        ax=ax,
    )

    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    ax.title.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#2d3748")

    plt.tight_layout()
    return fig


def feature_importance_table(shap_values, feature_names):
    """
    Generate a ranked feature importance table from SHAP values.
    
    Args:
        shap_values: shap.Explanation object
        feature_names: Feature names
        
    Returns:
        pd.DataFrame with columns: Rank, Feature, Mean_|SHAP|, Relative_Importance_%
    """
    vals = shap_values.values if hasattr(shap_values, 'values') else shap_values

    mean_abs_shap = np.abs(vals).mean(axis=0)
    total = mean_abs_shap.sum()

    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "Feature": name,
            "Mean_|SHAP|": round(mean_abs_shap[i], 4),
            "Relative_Importance_%": round(mean_abs_shap[i] / total * 100, 2) if total > 0 else 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Mean_|SHAP|", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df
