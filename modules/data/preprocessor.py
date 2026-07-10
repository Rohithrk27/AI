"""
Data Preprocessor Module
Handles missing values, normalization, and categorical encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def handle_missing_values(df: pd.DataFrame, cols: list, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in specified columns.
    
    Args:
        df: DataFrame
        cols: Columns to process
        strategy: "mean", "median", "drop", or "interpolate"
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()

    if strategy == "drop":
        df = df.dropna(subset=cols).reset_index(drop=True)
    elif strategy == "mean":
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
    elif strategy == "interpolate":
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    return df


def normalize_features(X: np.ndarray, method: str = "standard", scaler=None):
    """
    Normalize feature matrix.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        method: "standard" (z-score) or "minmax" (0-1)
        scaler: Optional pre-fitted scaler for transform-only
        
    Returns:
        tuple: (X_scaled, scaler)
    """
    if scaler is not None:
        return scaler.transform(X), scaler

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def encode_categoricals(df: pd.DataFrame, cat_cols: list):
    """
    One-hot encode categorical columns.
    
    Args:
        df: DataFrame
        cat_cols: List of categorical column names
        
    Returns:
        tuple: (encoded_df, encoder, new_col_names)
    """
    if not cat_cols:
        return df, None, []

    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoded = encoder.fit_transform(df[cat_cols])

    # Generate column names
    new_col_names = []
    for i, col in enumerate(cat_cols):
        categories = encoder.categories_[i][1:]  # Skip first due to drop="first"
        for cat in categories:
            new_col_names.append(f"{col}_{cat}")

    encoded_df = pd.DataFrame(encoded, columns=new_col_names, index=df.index)

    # Drop original categorical columns and add encoded
    result = df.drop(columns=cat_cols)
    result = pd.concat([result, encoded_df], axis=1)

    return result, encoder, new_col_names


def preprocess_pipeline(df: pd.DataFrame, factor_cols: list, response_cols: list, config: dict) -> dict:
    """
    Full preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        factor_cols: Factor column names
        response_cols: Response column names
        config: {missing_strategy, normalization}
        
    Returns:
        dict: {X, y, scaler, encoder, feature_names, metadata}
    """
    all_cols = factor_cols + response_cols
    working_df = df[all_cols].copy()

    # Step 1: Handle missing values
    missing_strategy = config.get("missing_strategy", "mean")
    working_df = handle_missing_values(working_df, all_cols, strategy=missing_strategy)

    # Step 2: Identify categorical columns
    cat_cols = []
    num_factor_cols = []
    for col in factor_cols:
        if working_df[col].dtype == object or working_df[col].dtype.name == "category":
            cat_cols.append(col)
        else:
            num_factor_cols.append(col)

    # Step 3: Encode categoricals
    encoder = None
    encoded_col_names = []
    if cat_cols:
        working_df, encoder, encoded_col_names = encode_categoricals(working_df, cat_cols)

    # Feature names after encoding
    feature_names = num_factor_cols + encoded_col_names

    # Step 4: Extract X and y
    X = working_df[feature_names].values.astype(float)
    y = working_df[response_cols].values.astype(float)

    # Flatten y if single response
    if y.shape[1] == 1:
        y = y.ravel()

    # Step 5: Normalize features
    norm_method = config.get("normalization", "standard")
    X_scaled, scaler = normalize_features(X, method=norm_method)

    return {
        "X": X_scaled,
        "X_original": X,
        "y": y,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": feature_names,
        "factor_cols": factor_cols,
        "response_cols": response_cols,
        "n_samples": X_scaled.shape[0],
        "n_features": X_scaled.shape[1],
    }
