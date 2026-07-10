"""
RSM Software Parsers
Detects and parses native exports from Design-Expert, Minitab, and JMP.
"""

import pandas as pd
import numpy as np
import re
import io


def detect_source_format(content: str) -> str:
    """
    Auto-detect the source format of a data file.
    
    Args:
        content: Raw text content of the file
        
    Returns:
        str: "design_expert", "minitab", "jmp", or "generic"
    """
    content_lower = content.lower()

    # Design-Expert detection
    de_indicators = [
        "design-expert",
        "design expert",
        "factor 1",
        "response 1",
        "coded factor",
        "actual factor",
        "anova for",
        "model f-value",
        "std. dev.",
        "adequate precision",
        "box-behnken",
        "central composite",
        "block 1",
    ]
    de_score = sum(1 for ind in de_indicators if ind in content_lower)

    # Minitab detection
    mt_indicators = [
        "minitab",
        "worksheet",
        "session",
        "regression equation",
        "analysis of variance",
        "source      df",
        "stdorder",
        "runorder",
        "pttype",
        "blocks",
        "factorial design",
    ]
    mt_score = sum(1 for ind in mt_indicators if ind in content_lower)

    # JMP detection
    jmp_indicators = [
        "jmp",
        "response surface",
        "factor(", 
        "continuous",
        "nominal",
        "column properties",
    ]
    jmp_score = sum(1 for ind in jmp_indicators if ind in content_lower)

    # Determine format by score
    scores = {
        "design_expert": de_score,
        "minitab": mt_score,
        "jmp": jmp_score,
    }

    max_score = max(scores.values())
    if max_score >= 2:
        return max(scores, key=scores.get)

    return "generic"


def parse_design_expert(content: str) -> dict:
    """
    Parse a Design-Expert export file.
    
    Handles several DE export patterns:
    - CSV data exports with headers like "Std", "Run", "Factor 1: A: Temperature"
    - Full report exports with ANOVA tables embedded
    - Coded vs actual factor levels
    
    Args:
        content: Raw text content
        
    Returns:
        dict with keys: format, data, metadata
    """
    lines = content.strip().split("\n")
    metadata = {
        "factors": [],
        "responses": [],
        "design_type": "Unknown",
    }

    # Try to detect design type
    content_lower = content.lower()
    if "box-behnken" in content_lower:
        metadata["design_type"] = "Box-Behnken"
    elif "central composite" in content_lower or "ccd" in content_lower:
        metadata["design_type"] = "Central Composite Design (CCD)"
    elif "factorial" in content_lower:
        metadata["design_type"] = "Factorial Design"
    elif "optimal" in content_lower:
        metadata["design_type"] = "D-Optimal Design"

    # Find the data table
    # Design-Expert headers typically: Std, Run, Block, Factor 1: A: Name, ..., Response 1: Name
    data_start = None
    header_line = None

    for i, line in enumerate(lines):
        # Look for header row with "Std" or "Run" or "Factor"
        if re.search(r'\b(Std|Run|Factor\s+\d|Response\s+\d)\b', line, re.IGNORECASE):
            header_line = i
            data_start = i + 1
            break

    if header_line is None:
        # Try parsing as plain CSV
        try:
            df = pd.read_csv(io.StringIO(content))
            return {"format": "design_expert", "data": df, "metadata": metadata}
        except Exception:
            raise ValueError("Could not parse Design-Expert format")

    # Parse header to extract factor and response names
    header = lines[header_line]
    # Try comma-separated first, then tab
    if "," in header:
        sep = ","
    elif "\t" in header:
        sep = "\t"
    else:
        sep = ","

    cols = [c.strip() for c in header.split(sep)]
    clean_cols = []

    for col in cols:
        # Parse "Factor 1: A: Temperature" -> "Temperature"
        factor_match = re.match(r'Factor\s+\d+:\s*\w:\s*(.*)', col, re.IGNORECASE)
        if factor_match:
            name = factor_match.group(1).strip()
            clean_cols.append(name)
            metadata["factors"].append({
                "name": name,
                "code": col.split(":")[1].strip() if ":" in col else "",
                "low": None,
                "high": None,
            })
            continue

        # Parse "Response 1: Yield" -> "Yield"
        response_match = re.match(r'Response\s+\d+:\s*(.*)', col, re.IGNORECASE)
        if response_match:
            name = response_match.group(1).strip()
            clean_cols.append(name)
            metadata["responses"].append({"name": name})
            continue

        # Skip Std, Run, Block columns but keep them
        clean_cols.append(col)

    # Parse data rows
    data_lines = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        # Stop at ANOVA section
        if re.search(r'(ANOVA|Analysis of Variance|Model Summary)', line, re.IGNORECASE):
            break
        values = [v.strip() for v in line.split(sep)]
        if len(values) == len(clean_cols):
            data_lines.append(values)

    if data_lines:
        df = pd.DataFrame(data_lines, columns=clean_cols)
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        # Remove Std, Run, Block columns if present
        drop_cols = [c for c in df.columns if c.lower() in ["std", "run", "block", "std order", "run order"]]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Update factor ranges
        for factor in metadata["factors"]:
            if factor["name"] in df.columns:
                factor["low"] = float(df[factor["name"]].min())
                factor["high"] = float(df[factor["name"]].max())

        # Auto-detect responses if not found in header
        if not metadata["responses"]:
            # Last column is typically the response
            non_factor_cols = [c for c in df.columns if c not in [f["name"] for f in metadata["factors"]]]
            for col in non_factor_cols:
                if df[col].dtype in [np.float64, np.int64]:
                    metadata["responses"].append({"name": col})

        return {"format": "design_expert", "data": df, "metadata": metadata}

    raise ValueError("Could not extract data from Design-Expert format")


def parse_minitab(content: str) -> dict:
    """
    Parse a Minitab export file.
    
    Handles:
    - Worksheet data exports (clean tabular CSV/TXT)
    - Session output with embedded ANOVA and regression equations
    - StdOrder/RunOrder/PtType columns
    
    Args:
        content: Raw text content
        
    Returns:
        dict with keys: format, data, metadata
    """
    metadata = {
        "factors": [],
        "responses": [],
        "design_type": "Unknown",
    }

    content_lower = content.lower()

    # Detect design type
    if "box-behnken" in content_lower:
        metadata["design_type"] = "Box-Behnken"
    elif "central composite" in content_lower:
        metadata["design_type"] = "Central Composite Design (CCD)"
    elif "factorial" in content_lower:
        metadata["design_type"] = "Factorial Design"

    # Try to parse regression equation if present
    eq_match = re.search(r'Regression Equation.*?\n(.*?)(?:\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
    anova_text = None
    anova_match = re.search(r'Analysis of Variance(.*?)(?:\n\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
    if anova_match:
        anova_text = anova_match.group(1)

    # Find the data table
    # Minitab worksheets usually start with column headers
    # Often has StdOrder, RunOrder, PtType, Blocks columns

    # Try parsing the full content as CSV
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(content), sep="\t")
        except Exception:
            # Try to find the data section by looking for numeric rows
            lines = content.strip().split("\n")
            data_start = None
            for i, line in enumerate(lines):
                # Look for a line that looks like a header row
                parts = re.split(r'[,\t]+', line.strip())
                if len(parts) >= 3 and all(not p.replace(".", "").replace("-", "").isdigit() for p in parts[:3] if p.strip()):
                    data_start = i
                    break

            if data_start is not None:
                remaining = "\n".join(lines[data_start:])
                sep = "\t" if "\t" in lines[data_start] else ","
                df = pd.read_csv(io.StringIO(remaining), sep=sep)
            else:
                raise ValueError("Could not parse Minitab format")

    # Drop Minitab metadata columns
    minitab_cols = ["stdorder", "runorder", "pttype", "blocks"]
    drop_cols = [c for c in df.columns if c.lower().strip() in minitab_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Detect factors and responses
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Assume last numeric column is response, rest are factors
        for col in numeric_cols[:-1]:
            metadata["factors"].append({
                "name": col,
                "low": float(df[col].min()),
                "high": float(df[col].max()),
            })
        metadata["responses"].append({"name": numeric_cols[-1]})

    return {"format": "minitab", "data": df, "metadata": metadata}


def parse_jmp(content: str) -> dict:
    """
    Parse a JMP export file.
    
    JMP exports may include column roles ("X" for factors, "Y" for responses)
    in the header or in a separate metadata row.
    
    Args:
        content: Raw text content
        
    Returns:
        dict with keys: format, data, metadata
    """
    metadata = {
        "factors": [],
        "responses": [],
        "design_type": "Unknown",
    }

    # Try parsing as CSV
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(content), sep="\t")
        except Exception:
            raise ValueError("Could not parse JMP format")

    # JMP sometimes has a second row with column roles
    # Check if first data row contains role markers like "Continuous", "Nominal", "X", "Y"
    first_row = df.iloc[0].astype(str).str.lower()
    role_indicators = ["continuous", "nominal", "ordinal", "x", "y"]
    
    if any(ind in " ".join(first_row.values) for ind in role_indicators):
        # First row is role/type indicators, remove it
        roles = df.iloc[0].astype(str)
        df = df.iloc[1:].reset_index(drop=True)

        # Convert numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        # Use roles to classify
        for col, role in zip(df.columns, roles):
            role_lower = str(role).lower()
            if "x" in role_lower or "factor" in role_lower:
                metadata["factors"].append({
                    "name": col,
                    "low": float(df[col].min()) if df[col].dtype in [np.float64, np.int64] else None,
                    "high": float(df[col].max()) if df[col].dtype in [np.float64, np.int64] else None,
                })
            elif "y" in role_lower or "response" in role_lower:
                metadata["responses"].append({"name": col})
    else:
        # No role info — treat like generic
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            for col in numeric_cols[:-1]:
                metadata["factors"].append({
                    "name": col,
                    "low": float(df[col].min()),
                    "high": float(df[col].max()),
                })
            metadata["responses"].append({"name": numeric_cols[-1]})

    return {"format": "jmp", "data": df, "metadata": metadata}


def smart_import(source) -> dict:
    """
    Master import function that auto-detects format and routes to appropriate parser.
    
    Args:
        source: File path (str) or Streamlit UploadedFile object
        
    Returns:
        dict: {format, data (DataFrame), metadata}
    """
    # Read raw content
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    else:
        content = source.read().decode("utf-8", errors="ignore")
        source.seek(0)  # Reset for potential re-read

    # Detect format
    fmt = detect_source_format(content)

    if fmt == "design_expert":
        try:
            return parse_design_expert(content)
        except Exception:
            fmt = "generic"

    if fmt == "minitab":
        try:
            return parse_minitab(content)
        except Exception:
            fmt = "generic"

    if fmt == "jmp":
        try:
            return parse_jmp(content)
        except Exception:
            fmt = "generic"

    # Generic fallback
    try:
        if isinstance(source, str):
            if source.endswith((".xlsx", ".xls")):
                df = pd.read_excel(source)
            else:
                df = pd.read_csv(source)
        else:
            name = getattr(source, "name", "").lower()
            if name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(source)
            else:
                df = pd.read_csv(source)
    except Exception:
        df = pd.read_csv(io.StringIO(content))

    return {
        "format": "generic",
        "data": df,
        "metadata": {
            "factors": [],
            "responses": [],
            "design_type": "Unknown",
        },
    }
