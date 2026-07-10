"""
Acquisition Functions Module
Implements Expected Improvement (EI), Upper Confidence Bound (UCB), and Probability of Improvement (PI).
"""

import numpy as np
from scipy.stats import norm


# Registry for easy lookup
ACQUISITION_FUNCTIONS = {
    "EI": "Expected Improvement",
    "UCB": "Upper Confidence Bound",
    "PI": "Probability of Improvement",
}


def expected_improvement(X, model, y_best, xi=0.01, minimize=False):
    """
    Expected Improvement acquisition function.
    
    EI(x) = (μ(x) − y_best) × Φ(Z) + σ(x) × φ(Z)
    where Z = (μ(x) − y_best − ξ) / σ(x)
    
    Args:
        X: Candidate points (n_candidates, n_features)
        model: Trained SurrogateModel with predict_with_uncertainty
        y_best: Best observed value so far
        xi: Exploration-exploitation trade-off parameter
        minimize: If True, minimizes instead of maximizes
        
    Returns:
        np.ndarray: EI values for each candidate
    """
    mu, sigma = model.predict_with_uncertainty(X)

    if minimize:
        mu = -mu
        y_best = -y_best

    sigma = np.maximum(sigma, 1e-8)  # Avoid division by zero

    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

    # Set EI to 0 where sigma is essentially 0
    ei[sigma < 1e-8] = 0.0

    return ei


def upper_confidence_bound(X, model, kappa=2.0, minimize=False):
    """
    Upper Confidence Bound acquisition function.
    
    UCB(x) = μ(x) + κ × σ(x)
    Higher κ = more exploration.
    
    Args:
        X: Candidate points
        model: Trained SurrogateModel
        kappa: Exploration parameter (default: 2.0)
        minimize: If True, uses LCB (Lower Confidence Bound)
        
    Returns:
        np.ndarray: UCB values
    """
    mu, sigma = model.predict_with_uncertainty(X)

    if minimize:
        return -(mu - kappa * sigma)  # LCB, negated for maximization
    return mu + kappa * sigma


def probability_of_improvement(X, model, y_best, xi=0.01, minimize=False):
    """
    Probability of Improvement acquisition function.
    
    PI(x) = Φ((μ(x) − y_best − ξ) / σ(x))
    
    Args:
        X: Candidate points
        model: Trained SurrogateModel
        y_best: Best observed value
        xi: Improvement threshold
        minimize: If True, minimizes
        
    Returns:
        np.ndarray: PI values
    """
    mu, sigma = model.predict_with_uncertainty(X)

    if minimize:
        mu = -mu
        y_best = -y_best

    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - y_best - xi) / sigma
    pi = norm.cdf(Z)

    return pi


def get_acquisition_function(name: str):
    """Return the acquisition function by name."""
    funcs = {
        "EI": expected_improvement,
        "UCB": upper_confidence_bound,
        "PI": probability_of_improvement,
    }
    return funcs.get(name, expected_improvement)
