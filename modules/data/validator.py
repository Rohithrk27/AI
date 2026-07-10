"""
Data Validator Module
Detects variable types, validates datasets, and computes statistics.
"""

import pandas as pd
import numpy as np


def detect_variable_types(df: pd.DataFrame, factor_cols: list) -> dict:
    """
    Auto-detect whether each factor is continuous, categorical, or integer.
    
    Heuristics:
    - If dtype is object/string -> categorical
    - If all values are integers and unique_ratio < 0.3 -> integer
    - Otherwise -> continuous
    
    Args:
        df: DataFrame
        factor_cols: List of factor column names
        
    Returns:
        dict: {column_name: "continuous" | "categorical" | "integer"}
    """
    var_types = {}

    for col in factor_cols:
        if col not in df.columns:
            var_types[col] = "continuous"
            continue

        series = df[col].dropna()

        # String/object type
        if series.dtype == object or series.dtype.name == "category":
            var_types[col] = "categorical"
            continue

        # Check if all values are integers
        try:
            is_integer = np.all(series == series.astype(int))
        except (ValueError, TypeError):
            is_integer = False

        if is_integer:
            n_unique = series.nunique()
            unique_ratio = n_unique / len(series) if len(series) > 0 else 0

            # Few unique values relative to dataset size -> likely integer/level
            if unique_ratio < 0.3 or n_unique <= 10:
                var_types[col] = "integer"
            else:
                var_types[col] = "continuous"
        else:
            var_types[col] = "continuous"

    return var_types


def validate_dataset(df: pd.DataFrame, factor_cols: list, response_cols: list) -> dict:
    """
    Validate a dataset for use in the optimization pipeline.
    
    Args:
        df: DataFrame
        factor_cols: List of factor column names
        response_cols: List of response column names
        
    Returns:
        dict: {valid: bool, errors: list, warnings: list}
    """
    errors = []
    warnings = []

    # Check columns exist
    all_cols = factor_cols + response_cols
    for col in all_cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in dataset.")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check minimum sample size
    n_rows = len(df)
    if n_rows < 5:
        errors.append(f"Too few data points ({n_rows}). Need at least 5 experiments.")
    elif n_rows < 10:
        warnings.append(f"Small dataset ({n_rows} experiments). Results may be unreliable.")

    # Check response is numeric
    for col in response_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Response column '{col}' must be numeric.")

    # Check for constant columns
    for col in all_cols:
        if col in df.columns and df[col].nunique() <= 1:
            warnings.append(f"Column '{col}' has only one unique value.")

    # Check missing values
    missing = df[all_cols].isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            warnings.append(f"Column '{col}' has {count} missing values.")

    # Check for factors vs response overlap
    overlap = set(factor_cols) & set(response_cols)
    if overlap:
        errors.append(f"Columns {overlap} cannot be both factor and response.")

    # Check factor columns are numeric (except categoricals)
    for col in factor_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype != object:
                warnings.append(f"Factor '{col}' is not numeric. Will be treated as categorical.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all columns.
    
    Args:
        df: DataFrame
        
    Returns:
        pd.DataFrame with stats: Count, Missing, Min, Max, Mean, Std, Median
    """
    stats_data = []

    for col in df.columns:
        row = {"Column": col}
        row["Count"] = df[col].count()
        row["Missing"] = df[col].isnull().sum()
        row["Type"] = str(df[col].dtype)

        if pd.api.types.is_numeric_dtype(df[col]):
            row["Min"] = round(df[col].min(), 4)
            row["Max"] = round(df[col].max(), 4)
            row["Mean"] = round(df[col].mean(), 4)
            row["Std"] = round(df[col].std(), 4)
            row["Median"] = round(df[col].median(), 4)
        else:
            row["Min"] = "-"
            row["Max"] = "-"
            row["Mean"] = "-"
            row["Std"] = "-"
            row["Median"] = "-"
            row["Unique"] = df[col].nunique()

        stats_data.append(row)

    return pd.DataFrame(stats_data)
