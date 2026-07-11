import pandas as pd
import numpy as np
from pyDOE3 import bbdesign, ccdesign

def scale_factor(coded_val, f_min, f_max):
    """
    Scale a coded value (e.g., -1, 0, 1) to its actual physical value 
    based on the minimum and maximum bounds.
    """
    f_mid = (f_min + f_max) / 2.0
    f_range = (f_max - f_min) / 2.0
    return f_mid + (coded_val * f_range)

def generate_doe(factors: list, design_type: str = "BBD") -> pd.DataFrame:
    """
    Generate a Design of Experiments (DOE) matrix.
    
    Args:
        factors: list of dicts, e.g., [{"name": "Temp", "min": 20, "max": 40}, ...]
        design_type: "BBD" (Box-Behnken) or "CCD" (Central Composite)
        
    Returns:
        pd.DataFrame containing the un-coded (actual) experimental matrix
    """
    n_factors = len(factors)
    
    if n_factors < 2:
        raise ValueError("At least 2 factors are required for a DOE.")
        
    if design_type == "BBD":
        if n_factors < 3:
            raise ValueError("Box-Behnken Design requires at least 3 factors.")
        # Generate coded matrix ([-1, 0, 1])
        coded_matrix = bbdesign(n_factors, center=3)
        
    elif design_type == "CCD":
        # Generate coded matrix with center points and axial points
        # Face-centered (alpha='f') means axial points are at -1 and +1
        # Inscribed (alpha='i') or Orthogonal (alpha='o') are options, but 'f' or standard is typical.
        # We'll use the default center=2 and standard alpha for now, but clip it to user bounds if requested.
        # Let's use 'r' (rotatable) or just default which is often alpha = sqrt(n).
        # But wait, if alpha > 1, it will exceed user's min/max. 
        # For simplicity and to strictly respect user bounds, Face-Centered ('f') is safer!
        coded_matrix = ccdesign(n_factors, center=(2, 2), alpha='f')
    
    else:
        raise ValueError(f"Unknown design type: {design_type}")
        
    # Scale coded matrix to actual values
    actual_matrix = np.zeros_like(coded_matrix, dtype=float)
    
    for i, factor in enumerate(factors):
        f_min = float(factor["min"])
        f_max = float(factor["max"])
        
        # Apply scaling to the entire column
        for row_idx in range(len(coded_matrix)):
            actual_matrix[row_idx, i] = scale_factor(coded_matrix[row_idx, i], f_min, f_max)
            
    # Create DataFrame
    df = pd.DataFrame(actual_matrix, columns=[f["name"] for f in factors])
    
    return df
