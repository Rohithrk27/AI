"""
Data Loader Module
Handles CSV and Excel file imports.
"""

import pandas as pd
import io


def load_file(source):
    """
    Load data from a file source (path string or uploaded file object).
    
    Args:
        source: File path (str) or Streamlit UploadedFile object
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if isinstance(source, str):
        # File path
        if source.endswith(".csv"):
            return pd.read_csv(source)
        elif source.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        elif source.endswith(".txt"):
            # Try tab-delimited first, then comma
            try:
                df = pd.read_csv(source, sep="\t")
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
            return pd.read_csv(source)
        else:
            return pd.read_csv(source)
    else:
        # Streamlit UploadedFile
        name = getattr(source, "name", "").lower()
        if name.endswith(".csv"):
            return pd.read_csv(source)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(source)
        elif name.endswith(".txt"):
            content = source.read().decode("utf-8")
            source.seek(0)
            try:
                df = pd.read_csv(io.StringIO(content), sep="\t")
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
            return pd.read_csv(io.StringIO(content))
        else:
            return pd.read_csv(source)


def load_csv(file):
    """Load a CSV file."""
    return pd.read_csv(file)


def load_excel(file, sheet_name=None):
    """Load an Excel file."""
    if sheet_name:
        return pd.read_excel(file, sheet_name=sheet_name)
    return pd.read_excel(file)
