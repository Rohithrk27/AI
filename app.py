"""
OptiML — AI-Assisted Experimental Optimization Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="OptiML — AI Experimental Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for Premium Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 50%, #0e1117 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131722 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    section[data-testid="stSidebar"] .stRadio > label {
        color: #a0aec0 !important;
        font-weight: 500;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(26, 31, 46, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
    }

    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }

    .gradient-text-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
    }

    .hero-title {
        font-size: 3.5rem;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #a0aec0;
        font-weight: 400;
        margin-bottom: 32px;
    }

    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 24px 0;
        flex-wrap: wrap;
    }

    .step-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .step-badge-active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .step-badge-inactive {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    .step-badge-done {
        background: rgba(67, 233, 123, 0.15);
        color: #43e97b;
        border: 1px solid rgba(67, 233, 123, 0.3);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(26, 31, 46, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }

    .feature-card {
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 4px;
    }

    .feature-desc {
        font-size: 0.85rem;
        color: #718096;
        line-height: 1.5;
    }

    /* Workflow diagram */
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 20px;
        margin: 6px 0;
        background: rgba(26, 31, 46, 0.4);
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        transition: all 0.3s ease;
    }

    .workflow-step:hover {
        background: rgba(26, 31, 46, 0.7);
        transform: translateX(4px);
    }

    .workflow-arrow {
        text-align: center;
        color: #4a5568;
        font-size: 1.2rem;
        margin: 2px 0;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* File uploader */
    .stFileUploader {
        border-radius: 12px;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 31, 46, 0.5);
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.15);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #4a5568; }

    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Status pills */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-success {
        background: rgba(67, 233, 123, 0.15);
        color: #43e97b;
    }

    .status-warning {
        background: rgba(245, 175, 25, 0.15);
        color: #f5af19;
    }

    .status-info {
        background: rgba(79, 172, 254, 0.15);
        color: #4facfe;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Navigation
        "current_page": "🏠 Home",
        # Data
        "raw_data": None,
        "parsed_data": None,        # Output from rsm_parsers or loader
        "factor_cols": [],
        "response_cols": [],
        "variable_types": {},
        "preprocessed": None,       # {X, y, scaler, encoder, metadata}
        # RSM Parser
        "source_format": None,      # "design_expert", "minitab", "jmp", "generic"
        "parsed_metadata": None,    # Factors, responses, design type from parser
        # ML
        "trained_models": {},       # {model_name: trained_model}
        "tuning_results": {},       # {model_name: optuna_results}
        "evaluation_results": {},   # {model_name: metrics_dict}
        "best_model_name": None,
        "best_model": None,
        # Optimization
        "bo_recommendations": None, # DataFrame of recommended experiments
        "bo_history": [],           # List of (iteration, best_value)
        "optimization_config": {},
        # Iteration
        "iteration_count": 0,
        "all_data": None,           # Combined original + new data
        # Report
        "report_data": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "📄 Upload Data",
    "🧹 Preprocessing",
    "🤖 Train Models",
    "🔬 Explainability",
    "📊 Visualizations",
    "🎯 Optimize",
    "🔄 Iterate",
    "📄 Report",
    "📚 History",
]

def get_step_status(page):
    """Return status for each page step."""
    if page == "🏠 Home":
        return "done"
    if page == "📄 Upload Data":
        return "done" if st.session_state.raw_data is not None else "active" if st.session_state.current_page == page else "inactive"
    if page == "🧹 Preprocessing":
        return "done" if st.session_state.preprocessed is not None else "active" if st.session_state.current_page == page else "inactive"
    if page == "🤖 Train Models":
        return "done" if st.session_state.best_model is not None else "active" if st.session_state.current_page == page else "inactive"
    if page == "🔬 Explainability":
        return "done" if st.session_state.best_model is not None and "shap_done" in st.session_state else "active" if st.session_state.current_page == page else "inactive"
    if page == "📊 Visualizations":
        return "active" if st.session_state.current_page == page else "inactive"
    if page == "🎯 Optimize":
        return "done" if st.session_state.bo_recommendations is not None else "active" if st.session_state.current_page == page else "inactive"
    if page == "🔄 Iterate":
        return "done" if st.session_state.iteration_count > 0 else "active" if st.session_state.current_page == page else "inactive"
    return "inactive"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.5rem;">🧬</div>
        <div class="gradient-text" style="font-size: 1.8rem;">OptiML</div>
        <div style="color: #718096; font-size: 0.8rem; margin-top: 4px;">AI Experimental Optimizer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.current_page),
        label_visibility="collapsed",
    )
    st.session_state.current_page = selected_page

    st.markdown("---")

    # Quick status panel
    st.markdown("##### 📊 Session Status")
    data_status = "✅" if st.session_state.raw_data is not None else "⬜"
    prep_status = "✅" if st.session_state.preprocessed is not None else "⬜"
    model_status = "✅" if st.session_state.best_model is not None else "⬜"
    opt_status = "✅" if st.session_state.bo_recommendations is not None else "⬜"

    st.markdown(f"""
    {data_status} Data loaded  
    {prep_status} Preprocessed  
    {model_status} Models trained  
    {opt_status} Optimized  
    """)


# ──────────────────────────────────────────────
# Page: Home
# ──────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="hero-container animate-in">
        <div class="hero-title">
            <span class="gradient-text">OptiML</span>
        </div>
        <p class="hero-subtitle">
            AI-Assisted Experimental Optimization Platform<br/>
            <span style="color: #667eea;">Train • Optimize • Discover</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    st.markdown("""
    <div class="feature-grid animate-in">
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Smart Data Import</div>
            <div class="feature-desc">Upload CSV, Excel, or native Design-Expert, Minitab, and JMP exports. Auto-detects format and variables.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">7 ML Models</div>
            <div class="feature-desc">Trains GP, Random Forest, XGBoost, LightGBM, CatBoost, ANN, and SVR. Auto-selects the best.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚙️</div>
            <div class="feature-title">Auto Hyperparameter Tuning</div>
            <div class="feature-desc">Optuna-powered hyperparameter optimization for every model. No manual tuning needed.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Bayesian Optimization</div>
            <div class="feature-desc">EI, UCB, and PI acquisition functions intelligently recommend your next experiments.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">SHAP Explainability</div>
            <div class="feature-desc">Understand which variables drive your response using SHAP feature importance.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Interactive Visualizations</div>
            <div class="feature-desc">3D response surfaces, contour plots, convergence charts — all interactive with Plotly.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎛️</div>
            <div class="feature-title">Multi-Objective</div>
            <div class="feature-desc">Optimize multiple responses simultaneously. Visualize Pareto fronts and trade-offs.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Publication Reports</div>
            <div class="feature-desc">Export PDF reports with ANOVA tables, model comparisons, and optimized conditions.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Workflow diagram
    st.markdown('<h3 class="gradient-text-blue" style="text-align: center;">How It Works</h3>', unsafe_allow_html=True)

    workflow_steps = [
        ("📄", "Upload your experimental data", "border-color: #667eea;"),
        ("🔍", "Auto-detect format and variable types", "border-color: #764ba2;"),
        ("🧹", "Preprocess: handle missing values, normalize, encode", "border-color: #f093fb;"),
        ("🤖", "Train & benchmark 7 ML models with auto-tuning", "border-color: #ec4899;"),
        ("🏆", "Automatically select the best-performing model", "border-color: #f5af19;"),
        ("🔬", "Explain predictions with SHAP analysis", "border-color: #ff6b6b;"),
        ("🎯", "Bayesian Optimization recommends next experiments", "border-color: #4facfe;"),
        ("✍️", "Perform experiments and enter results", "border-color: #43e97b;"),
        ("🔄", "Model retrains and improves", "border-color: #38f9d7;"),
        ("📄", "Export publication-ready report", "border-color: #667eea;"),
    ]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        for i, (icon, text, style) in enumerate(workflow_steps):
            st.markdown(f'<div class="workflow-step" style="{style}">{icon}&nbsp;&nbsp;{text}</div>', unsafe_allow_html=True)
            if i < len(workflow_steps) - 1:
                st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Call to action
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀  Get Started — Upload Data", use_container_width=True, type="primary"):
            st.session_state.current_page = "📄 Upload Data"
            st.rerun()

    # Domain examples
    st.markdown("---")
    st.markdown('<h3 class="gradient-text-blue" style="text-align: center;">Works Across Domains</h3>', unsafe_allow_html=True)

    domains = [
        "🧫 Biotechnology", "⚗️ Chemical Engineering", "🔬 Materials Science",
        "💊 Drug Formulation", "🌾 Agriculture", "🍔 Food Science",
        "🌍 Environmental Engineering", "🔮 Nanotechnology", "🧪 Fermentation",
    ]

    cols = st.columns(3)
    for i, domain in enumerate(domains):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="margin: 4px 0;">
                <div style="font-size: 0.95rem; color: #e0e0e0;">{domain}</div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Page: Upload Data
# ──────────────────────────────────────────────
def page_upload():
    st.markdown('<h2 class="gradient-text">📄 Upload Experimental Data</h2>', unsafe_allow_html=True)
    st.markdown("Upload your experimental data from CSV, Excel, or native RSM software exports (Design-Expert, Minitab, JMP).")

    # Import modules
    from modules.data.loader import load_file
    from modules.data.rsm_parsers import smart_import, detect_source_format
    from modules.data.validator import detect_variable_types, validate_dataset, compute_statistics

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your data file here",
            type=["csv", "xlsx", "xls", "txt"],
            help="Supported: CSV, Excel, Design-Expert export, Minitab export, JMP export"
        )

        use_sample = st.checkbox("📂 Use sample dataset (Citric Acid Adsorption)")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="feature-title">Supported Formats</div>
            <div class="feature-desc" style="margin-top: 8px;">
                ✅ Generic CSV / Excel<br/>
                ✅ Design-Expert exports<br/>
                ✅ Minitab worksheets<br/>
                ✅ JMP exports<br/>
            </div>
            <div class="feature-title" style="margin-top: 16px;">Auto-Detection</div>
            <div class="feature-desc" style="margin-top: 8px;">
                OptiML automatically detects your file format and extracts factors, responses, and design type.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Load data
    data_source = None
    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_citric_acid.csv")
        if os.path.exists(sample_path):
            data_source = sample_path
        else:
            st.warning("Sample dataset not found. Please upload your own data.")

    if uploaded_file is not None:
        data_source = uploaded_file

    if data_source is not None:
        try:
            # Try smart import first (for RSM software exports)
            result = smart_import(data_source)

            if result["format"] != "generic":
                st.success(f"🎯 Auto-detected format: **{result['format'].replace('_', ' ').title()}**")
                st.session_state.source_format = result["format"]
                st.session_state.parsed_metadata = result.get("metadata", {})

                # Show parsed metadata
                if result.get("metadata"):
                    with st.expander("📋 Parsed Metadata", expanded=True):
                        meta = result["metadata"]
                        if "design_type" in meta:
                            st.info(f"Design Type: **{meta['design_type']}**")
                        if "factors" in meta:
                            st.markdown("**Factors:**")
                            for f in meta["factors"]:
                                st.markdown(f"- {f['name']} ({f.get('low', '?')} → {f.get('high', '?')})")
                        if "responses" in meta:
                            st.markdown("**Responses:**")
                            for r in meta["responses"]:
                                st.markdown(f"- {r['name']}")

            df = result["data"]
            st.session_state.raw_data = df
            st.session_state.source_format = result["format"]

        except Exception:
            # Fallback to generic loader
            df = load_file(data_source)
            st.session_state.raw_data = df
            st.session_state.source_format = "generic"

        # Data preview
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(st.session_state.raw_data.head(20), use_container_width=True)

        # Statistics
        stats = compute_statistics(st.session_state.raw_data)
        with st.expander("📊 Dataset Statistics", expanded=False):
            st.dataframe(stats, use_container_width=True)

        # Column selection
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">Select Variables</h3>', unsafe_allow_html=True)

        all_cols = list(st.session_state.raw_data.columns)

        # Pre-fill from parser metadata if available
        default_factors = []
        default_responses = []
        if st.session_state.parsed_metadata:
            meta = st.session_state.parsed_metadata
            if "factors" in meta:
                default_factors = [f["name"] for f in meta["factors"] if f["name"] in all_cols]
            if "responses" in meta:
                default_responses = [r["name"] for r in meta["responses"] if r["name"] in all_cols]

        col1, col2 = st.columns(2)
        with col1:
            factor_cols = st.multiselect(
                "🔧 Select Factor Columns (Input Variables)",
                options=all_cols,
                default=default_factors,
                help="These are your experimental factors (e.g., Temperature, pH, Time)"
            )
        with col2:
            remaining = [c for c in all_cols if c not in factor_cols]
            response_cols = st.multiselect(
                "🎯 Select Response Columns (Output Variables)",
                options=remaining,
                default=[r for r in default_responses if r in remaining],
                help="These are what you're optimizing (e.g., Yield, Adsorption Efficiency)"
            )

        if factor_cols and response_cols:
            st.session_state.factor_cols = factor_cols
            st.session_state.response_cols = response_cols

            # Auto-detect variable types
            var_types = detect_variable_types(st.session_state.raw_data, factor_cols)
            st.session_state.variable_types = var_types

            st.markdown("**Detected Variable Types:**")
            type_cols = st.columns(len(factor_cols))
            for i, col_name in enumerate(factor_cols):
                vtype = var_types.get(col_name, "continuous")
                icon = {"continuous": "📈", "categorical": "🏷️", "integer": "🔢"}.get(vtype, "❓")
                with type_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.5rem;">{icon}</div>
                        <div style="font-size: 0.85rem; color: #e0e0e0; margin-top: 4px;">{col_name}</div>
                        <div style="font-size: 0.75rem; color: #667eea;">{vtype}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Validation
            validation = validate_dataset(st.session_state.raw_data, factor_cols, response_cols)
            if validation["valid"]:
                st.success(f"✅ Dataset is valid — {len(st.session_state.raw_data)} experiments, {len(factor_cols)} factors, {len(response_cols)} responses")
            else:
                for err in validation["errors"]:
                    st.error(f"❌ {err}")
            for warn in validation.get("warnings", []):
                st.warning(f"⚠️ {warn}")

            # Next step button
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("➡️  Proceed to Preprocessing", use_container_width=True, type="primary"):
                st.session_state.current_page = "🧹 Preprocessing"
                st.rerun()


# ──────────────────────────────────────────────
# Page: Preprocessing
# ──────────────────────────────────────────────
def page_preprocessing():
    st.markdown('<h2 class="gradient-text">🧹 Data Preprocessing</h2>', unsafe_allow_html=True)

    if st.session_state.raw_data is None:
        st.warning("⬅️ Please upload data first.")
        return

    from modules.data.preprocessor import handle_missing_values, normalize_features, encode_categoricals, preprocess_pipeline

    df = st.session_state.raw_data
    factor_cols = st.session_state.factor_cols
    response_cols = st.session_state.response_cols

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Missing Value Strategy**")
        missing_strategy = st.selectbox(
            "How to handle missing values",
            ["mean", "median", "drop", "interpolate"],
            help="Mean/Median: fill with column average. Drop: remove rows. Interpolate: linear interpolation."
        )

        missing_count = df[factor_cols + response_cols].isnull().sum().sum()
        if missing_count > 0:
            st.warning(f"⚠️ Found {missing_count} missing values")
        else:
            st.success("✅ No missing values")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Normalization Method**")
        norm_method = st.selectbox(
            "Feature scaling method",
            ["standard", "minmax"],
            help="Standard: zero mean, unit variance. MinMax: scale to [0, 1]."
        )

        cat_cols = [c for c, t in st.session_state.variable_types.items() if t == "categorical"]
        if cat_cols:
            st.info(f"🏷️ Categorical columns detected: {', '.join(cat_cols)} — will be one-hot encoded")
        st.markdown('</div>', unsafe_allow_html=True)

    # Run preprocessing
    st.markdown("---")
    if st.button("⚙️  Run Preprocessing", use_container_width=True, type="primary"):
        with st.spinner("Preprocessing data..."):
            config = {
                "missing_strategy": missing_strategy,
                "normalization": norm_method,
            }
            result = preprocess_pipeline(df, factor_cols, response_cols, config)
            st.session_state.preprocessed = result

        st.success(f"✅ Preprocessing complete — {result['X'].shape[0]} samples, {result['X'].shape[1]} features")

        # Show preview
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Processed Features (X)**")
            X_preview = pd.DataFrame(result["X"], columns=result["feature_names"])
            st.dataframe(X_preview.head(10), use_container_width=True)
        with col2:
            st.markdown("**Response (y)**")
            y_preview = pd.DataFrame(result["y"], columns=response_cols)
            st.dataframe(y_preview.head(10), use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("➡️  Proceed to Model Training", use_container_width=True, type="primary"):
            st.session_state.current_page = "🤖 Train Models"
            st.rerun()

    elif st.session_state.preprocessed is not None:
        result = st.session_state.preprocessed
        st.success(f"✅ Data already preprocessed — {result['X'].shape[0]} samples, {result['X'].shape[1]} features")

        if st.button("➡️  Proceed to Model Training", use_container_width=True, type="primary"):
            st.session_state.current_page = "🤖 Train Models"
            st.rerun()


# ──────────────────────────────────────────────
# Page: Train Models
# ──────────────────────────────────────────────
def page_train():
    st.markdown('<h2 class="gradient-text">🤖 Train & Benchmark Models</h2>', unsafe_allow_html=True)

    if st.session_state.preprocessed is None:
        st.warning("⬅️ Please preprocess your data first.")
        return

    from modules.ml.models import get_all_models
    from modules.ml.tuner import tune_all_models
    from modules.ml.evaluator import evaluate_model, rank_models
    from modules.ml.model_selector import auto_select_best

    X = st.session_state.preprocessed["X"]
    y = st.session_state.preprocessed["y"]

    # Training settings
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    models_dict = get_all_models()
    model_names = list(models_dict.keys())
    
    selected_models = st.multiselect(
        "Select Models to Train",
        options=model_names,
        default=["Gaussian Process"],
        help="Select one or more surrogate models to train and benchmark."
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        enable_tuning = st.checkbox("⚙️ Enable Auto-Tuning (Optuna)", value=True)
    with col2:
        n_trials = st.number_input("Trials per model", min_value=10, max_value=200, value=50, step=10)
    with col3:
        cv_folds = st.number_input("CV Folds", min_value=3, max_value=10, value=5)
    with col4:
        timeout = st.number_input("Timeout per model (sec)", min_value=10, max_value=300, value=60)
    st.markdown('</div>', unsafe_allow_html=True)

    # Train button
    if st.button("🚀  Train Selected Models", use_container_width=True, type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return
            
        models = {name: models_dict[name] for name in selected_models}
        progress_bar = st.progress(0, text="Initializing...")
        all_results = {}
        tuning_results = {}

        for i, (name, model) in enumerate(models.items()):
            progress_text = f"Training {name}... ({i+1}/{len(models)})"
            progress_bar.progress((i) / len(models), text=progress_text)

            try:
                if enable_tuning:
                    # Tune hyperparameters first
                    tune_result = tune_all_models(
                        {name: model}, X, y,
                        n_trials=n_trials, cv_folds=cv_folds, timeout=timeout
                    )
                    if name in tune_result:
                        tuning_results[name] = tune_result[name]
                        model.train(X, y, params=tune_result[name].get("best_params"))
                    else:
                        model.train(X, y)
                else:
                    model.train(X, y)

                # Evaluate
                metrics = evaluate_model(model, X, y, cv_folds=cv_folds)
                all_results[name] = metrics
                st.session_state.trained_models[name] = model

            except Exception as e:
                st.warning(f"⚠️ {name} failed: {str(e)}")
                all_results[name] = {"error": str(e)}

        progress_bar.progress(1.0, text="✅ All models trained!")

        st.session_state.evaluation_results = all_results
        st.session_state.tuning_results = tuning_results

        # Auto-select best
        best_name, best_model = auto_select_best(all_results, st.session_state.trained_models)
        st.session_state.best_model_name = best_name
        st.session_state.best_model = best_model

    # Display results
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">Model Benchmarking Results</h3>', unsafe_allow_html=True)

        # Ranking table
        ranking_df = rank_models(st.session_state.evaluation_results)
        st.dataframe(ranking_df, use_container_width=True)

        # Best model highlight
        if st.session_state.best_model_name:
            st.markdown(f"""
            <div class="glass-card" style="border-color: rgba(67, 233, 123, 0.4);">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 2rem;">🏆</div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #43e97b;">Best Model: {st.session_state.best_model_name}</div>
                        <div style="color: #a0aec0; font-size: 0.9rem;">Auto-selected based on cross-validated R²</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Comparison chart
        from modules.visualization.plots import plot_model_comparison
        fig = plot_model_comparison(st.session_state.evaluation_results)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬  Explore SHAP Explainability", use_container_width=True):
                st.session_state.current_page = "🔬 Explainability"
                st.rerun()
        with col2:
            if st.button("🎯  Proceed to Optimization", use_container_width=True, type="primary"):
                st.session_state.current_page = "🎯 Optimize"
                st.rerun()


# ──────────────────────────────────────────────
# Page: Explainability
# ──────────────────────────────────────────────
def page_explainability():
    st.markdown('<h2 class="gradient-text">🔬 SHAP Explainability</h2>', unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.warning("⬅️ Please train models first.")
        return

    from modules.explainability.shap_analysis import compute_shap_values, shap_summary_plot, shap_dependence_plot, feature_importance_table

    X = st.session_state.preprocessed["X"]
    feature_names = st.session_state.preprocessed["feature_names"]
    model = st.session_state.best_model
    model_name = st.session_state.best_model_name

    st.info(f"Analyzing: **{model_name}**")

    if st.button("🔬  Compute SHAP Values", use_container_width=True, type="primary"):
        with st.spinner("Computing SHAP values... This may take a moment."):
            try:
                shap_values = compute_shap_values(model, X, feature_names)
                st.session_state["shap_values"] = shap_values
                st.session_state["shap_done"] = True
            except Exception as e:
                st.error(f"SHAP computation failed: {str(e)}")
                return

    if "shap_values" in st.session_state:
        shap_values = st.session_state["shap_values"]

        # Feature importance table
        st.markdown('<h3 class="gradient-text-blue">Feature Importance Ranking</h3>', unsafe_allow_html=True)
        imp_table = feature_importance_table(shap_values, feature_names)
        st.dataframe(imp_table, use_container_width=True)

        # Summary plot
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">SHAP Summary Plot</h3>', unsafe_allow_html=True)
        fig = shap_summary_plot(shap_values, X, feature_names)
        st.pyplot(fig)

        # Dependence plots
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">SHAP Dependence Plot</h3>', unsafe_allow_html=True)
        selected_feature = st.selectbox("Select a feature", feature_names)
        fig_dep = shap_dependence_plot(shap_values, selected_feature, X, feature_names)
        st.pyplot(fig_dep)


# ──────────────────────────────────────────────
# Page: Visualizations
# ──────────────────────────────────────────────
def page_visualizations():
    st.markdown('<h2 class="gradient-text">📊 Visualizations</h2>', unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.warning("⬅️ Please train models first.")
        return

    from modules.visualization.plots import (
        plot_3d_surface, plot_contour, plot_pred_vs_actual,
        plot_residuals, plot_main_effects
    )

    model = st.session_state.best_model
    X = st.session_state.preprocessed["X"]
    y = st.session_state.preprocessed["y"]
    feature_names = st.session_state.preprocessed["feature_names"]
    factor_cols = st.session_state.factor_cols

    tab_names = [
        "3D Surface", "Contour Plot", "Predicted vs Actual", "Residuals", "Factor Effects"
    ]
    
    selected_plots = st.multiselect(
        "Select Plots to Display",
        options=tab_names,
        default=["3D Surface", "Contour Plot", "Predicted vs Actual", "Residuals", "Factor Effects"]
    )
    
    if not selected_plots:
        st.warning("⚠️ Please select at least one plot to display.")
        return
        
    tabs = st.tabs(selected_plots)

    for i, plot_name in enumerate(selected_plots):
        with tabs[i]:
            if plot_name == "3D Surface":
                if len(feature_names) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        f1 = st.selectbox("X-axis Factor", feature_names, index=0, key="surf_f1")
                    with col2:
                        f2_options = [f for f in feature_names if f != f1]
                        f2 = st.selectbox("Y-axis Factor", f2_options, index=0, key="surf_f2")

                    fig = plot_3d_surface(model, X, feature_names, f1, f2)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 factors for 3D surface plot.")

            elif plot_name == "Contour Plot":
                if len(feature_names) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        cf1 = st.selectbox("X-axis Factor", feature_names, index=0, key="cont_f1")
                    with col2:
                        cf2_options = [f for f in feature_names if f != cf1]
                        cf2 = st.selectbox("Y-axis Factor", cf2_options, index=0, key="cont_f2")

                    fig = plot_contour(model, X, feature_names, cf1, cf2)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 factors for contour plot.")

            elif plot_name == "Predicted vs Actual":
                y_pred = model.predict(X)
                # Handle multi-output y for plotting
                if y.ndim > 1 and y.shape[1] > 1:
                    y_actual_plot = y[:, 0]
                    y_pred_plot = y_pred[:, 0] if y_pred.ndim > 1 else y_pred
                    st.info(f"Plotting for first response column: {st.session_state.response_cols[0]}")
                else:
                    y_actual_plot = y.ravel()
                    y_pred_plot = y_pred.ravel()
                fig = plot_pred_vs_actual(y_actual_plot, y_pred_plot)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_name == "Residuals":
                y_pred = model.predict(X)
                if y.ndim > 1 and y.shape[1] > 1:
                    y_actual_plot = y[:, 0]
                    y_pred_plot = y_pred[:, 0] if y_pred.ndim > 1 else y_pred
                else:
                    y_actual_plot = y.ravel()
                    y_pred_plot = y_pred.ravel()
                fig = plot_residuals(y_actual_plot, y_pred_plot)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_name == "Factor Effects":
                fig = plot_main_effects(model, X, feature_names)
                st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Optimize
# ──────────────────────────────────────────────
def page_optimize():
    st.markdown('<h2 class="gradient-text">🎯 Bayesian Optimization</h2>', unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.warning("⬅️ Please train models first.")
        return

    from modules.optimization.bayesian_opt import recommend_experiments
    from modules.optimization.acquisition import ACQUISITION_FUNCTIONS

    model = st.session_state.best_model
    X = st.session_state.preprocessed["X"]
    y = st.session_state.preprocessed["y"]
    feature_names = st.session_state.preprocessed["feature_names"]

    response_cols = st.session_state.response_cols

    # Settings
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Objective selection
    has_multiple_responses = len(response_cols) > 1
    opt_type = "Single Objective"
    if has_multiple_responses:
        opt_type = st.radio("Optimization Mode", ["Single Objective", "Multi-Objective"])
    
    if opt_type == "Single Objective":
        target_col = st.selectbox("Target Response", response_cols)
        optimize_direction = st.selectbox("Objective Direction", ["Maximize", "Minimize"])
    else:
        st.markdown("**Multi-Objective Setup**")
        objectives_config = []
        for col in response_cols:
            c1, c2 = st.columns(2)
            with c1:
                dir_val = st.selectbox(f"Direction for {col}", ["Maximize", "Minimize"], key=f"dir_{col}")
            with c2:
                weight_val = st.number_input(f"Weight for {col}", min_value=0.1, value=1.0, step=0.1, key=f"wt_{col}")
            objectives_config.append({"name": col, "direction": "maximize" if dir_val == "Maximize" else "minimize", "weight": weight_val})

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        acq_func = st.selectbox("Acquisition Function", ["Expected Improvement (EI)", "Upper Confidence Bound (UCB)", "Probability of Improvement (PI)"])
    with col2:
        n_recommend = st.number_input("Number of experiments to suggest", min_value=1, max_value=20, value=5)
    with col3:
        n_candidates = st.number_input("Candidate pool size", min_value=1000, max_value=100000, value=20000, step=1000)
    st.markdown('</div>', unsafe_allow_html=True)

    acq_map = {
        "Expected Improvement (EI)": "EI",
        "Upper Confidence Bound (UCB)": "UCB",
        "Probability of Improvement (PI)": "PI",
    }

    if st.button("🎯  Run Bayesian Optimization", use_container_width=True, type="primary"):
        with st.spinner("Running Optimization..."):
            bounds = list(zip(X.min(axis=0), X.max(axis=0)))

            if opt_type == "Single Objective":
                # Determine which column index corresponds to the selected target
                target_idx = response_cols.index(target_col)
                if y.ndim > 1 and y.shape[1] > 1:
                    y_target = y[:, target_idx]
                else:
                    y_target = y.ravel()
                
                y_best = y_target.max() if optimize_direction == "Maximize" else y_target.min()
                
                recommendations = recommend_experiments(
                    model=model,
                    bounds=bounds,
                    feature_names=feature_names,
                    y_best=y_best,
                    acq_func=acq_map[acq_func],
                    n_recommend=n_recommend,
                    n_candidates=n_candidates,
                    minimize=(optimize_direction == "Minimize"),
                )
            else:
                from modules.optimization.multi_objective import multi_objective_recommend
                # For multi-objective, we simulate using the same model for all objectives if we only trained one,
                # but to be robust, we pass a list of models (here just duplicates of best_model for simplicity)
                models = [model for _ in range(len(response_cols))]
                
                recommendations = multi_objective_recommend(
                    models=models,
                    bounds=bounds,
                    feature_names=feature_names,
                    objectives_config=objectives_config,
                    n_recommend=n_recommend,
                    n_candidates=n_candidates
                )

            st.session_state.bo_recommendations = recommendations

    if st.session_state.bo_recommendations is not None:
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">📋 Recommended Next Experiments</h3>', unsafe_allow_html=True)
        st.markdown("Perform these experiments in your lab, then enter the results in the **🔄 Iterate** page.")

        # Inverse-transform to original scale if scaler exists
        recs = st.session_state.bo_recommendations.copy()

        st.dataframe(recs, use_container_width=True)

        # Show acquisition function landscape
        from modules.visualization.plots import plot_acquisition_landscape
        if len(feature_names) >= 2:
            st.markdown("---")
            st.markdown('<h3 class="gradient-text-blue">Acquisition Function Landscape</h3>', unsafe_allow_html=True)
            fig = plot_acquisition_landscape(model, X, feature_names, acq_map[acq_func])
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🔄  Go to Iterate (Enter New Results)", use_container_width=True, type="primary"):
            st.session_state.current_page = "🔄 Iterate"
            st.rerun()


# ──────────────────────────────────────────────
# Page: Iterate
# ──────────────────────────────────────────────
def page_iterate():
    st.markdown('<h2 class="gradient-text">🔄 Iterate — Enter New Results</h2>', unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.warning("⬅️ Please train models and run optimization first.")
        return

    feature_names = st.session_state.preprocessed["feature_names"]
    response_cols = st.session_state.response_cols

    st.markdown("""
    After performing the recommended experiments in your lab, enter the results below. 
    The model will retrain and provide updated recommendations.
    """)

    # Show current recommendations
    if st.session_state.bo_recommendations is not None:
        with st.expander("📋 Current Recommendations", expanded=True):
            st.dataframe(st.session_state.bo_recommendations, use_container_width=True)

    # Manual entry form
    st.markdown("---")
    st.markdown('<h3 class="gradient-text-blue">Enter New Experimental Results</h3>', unsafe_allow_html=True)

    n_new = st.number_input("Number of new experiments to enter", min_value=1, max_value=20, value=1)

    new_data = {}
    for col in feature_names + response_cols:
        new_data[col] = []

    for i in range(n_new):
        st.markdown(f"**Experiment {i+1}**")
        cols = st.columns(len(feature_names) + len(response_cols))
        for j, col_name in enumerate(feature_names + response_cols):
            with cols[j]:
                val = st.number_input(f"{col_name}", value=0.0, key=f"new_{i}_{col_name}", format="%.4f")
                new_data[col_name].append(val)

    if st.button("📥  Add Results & Retrain", use_container_width=True, type="primary"):
        new_df = pd.DataFrame(new_data)

        # Add to existing data
        from modules.data.preprocessor import preprocess_pipeline

        combined = pd.concat([st.session_state.raw_data, new_df], ignore_index=True)
        st.session_state.raw_data = combined

        # Repreprocess
        config = {"missing_strategy": "mean", "normalization": "standard"}
        result = preprocess_pipeline(combined, st.session_state.factor_cols, response_cols, config)
        st.session_state.preprocessed = result

        # Retrain best model
        X = result["X"]
        y = result["y"]
        st.session_state.best_model.train(X, y)
        st.session_state.iteration_count += 1

        # Track convergence
        st.session_state.bo_history.append({
            "iteration": st.session_state.iteration_count,
            "best_value": y.max(),
            "n_experiments": len(y),
        })

        st.success(f"✅ Model retrained with {len(combined)} experiments (iteration {st.session_state.iteration_count})")

        # Auto-suggest new recommendations
        st.session_state.bo_recommendations = None  # Reset so optimization can be rerun

        if st.button("🎯  Run New Optimization", use_container_width=True):
            st.session_state.current_page = "🎯 Optimize"
            st.rerun()

    # Convergence plot
    if st.session_state.bo_history:
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">Convergence History</h3>', unsafe_allow_html=True)
        from modules.visualization.plots import plot_convergence
        fig = plot_convergence(st.session_state.bo_history)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Page: Report
# ──────────────────────────────────────────────
def page_report():
    st.markdown('<h2 class="gradient-text">📄 Generate Report</h2>', unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.warning("⬅️ Please complete the optimization workflow first.")
        return

    from modules.reporting.report_generator import generate_report

    st.markdown("""
    Generate a publication-ready PDF report containing your full optimization workflow:
    dataset summary, model benchmarking, SHAP analysis, optimization results, and recommended conditions.
    """)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Report preview sections
    st.markdown("**Report will include:**")
    sections = [
        "✅ Dataset Summary (factors, responses, sample size)",
        "✅ Preprocessing Configuration",
        "✅ Model Benchmarking Table (all 7 models)",
        "✅ Best Model Details & Tuned Hyperparameters",
        "✅ SHAP Feature Importance",
        "✅ Optimization Results & Recommended Conditions",
        "✅ Convergence History",
        "✅ All Plots (3D Surface, Contour, Predicted vs Actual)",
    ]
    for s in sections:
        st.markdown(s)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("📄  Generate PDF Report", use_container_width=True, type="primary"):
        with st.spinner("Generating report..."):
            report_data = {
                "raw_data": st.session_state.raw_data,
                "factor_cols": st.session_state.factor_cols,
                "response_cols": st.session_state.response_cols,
                "preprocessed": st.session_state.preprocessed,
                "evaluation_results": st.session_state.evaluation_results,
                "best_model_name": st.session_state.best_model_name,
                "tuning_results": st.session_state.tuning_results,
                "bo_recommendations": st.session_state.bo_recommendations,
                "bo_history": st.session_state.bo_history,
                "iteration_count": st.session_state.iteration_count,
            }

            pdf_bytes = generate_report(report_data)

        st.success("✅ Report generated!")
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name="OptiML_Optimization_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# Page: History
# ──────────────────────────────────────────────
def page_history():
    st.markdown('<h2 class="gradient-text">📚 Experiment History</h2>', unsafe_allow_html=True)

    from database.history import init_database, load_experiment_history

    conn = init_database()
    history = load_experiment_history(conn)

    if history is not None and len(history) > 0:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No experiment history yet. Complete an optimization workflow to save results.")

    # Current session info
    st.markdown("---")
    st.markdown('<h3 class="gradient-text-blue">Current Session</h3>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_exp = len(st.session_state.raw_data) if st.session_state.raw_data is not None else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_exp}</div>
            <div class="metric-label">Experiments</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        n_models = len(st.session_state.trained_models)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_models}</div>
            <div class="metric-label">Models Trained</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.iteration_count}</div>
            <div class="metric-label">BO Iterations</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best = st.session_state.best_model_name or "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1rem;">{best}</div>
            <div class="metric-label">Best Model</div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Page Router
# ──────────────────────────────────────────────
PAGE_FUNCTIONS = {
    "🏠 Home": page_home,
    "📄 Upload Data": page_upload,
    "🧹 Preprocessing": page_preprocessing,
    "🤖 Train Models": page_train,
    "🔬 Explainability": page_explainability,
    "📊 Visualizations": page_visualizations,
    "🎯 Optimize": page_optimize,
    "🔄 Iterate": page_iterate,
    "📄 Report": page_report,
    "📚 History": page_history,
}

# Render the selected page
page_fn = PAGE_FUNCTIONS.get(st.session_state.current_page, page_home)
page_fn()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.8rem; padding: 12px;">
    🧬 OptiML v1.0 — AI-Assisted Experimental Optimization Platform<br/>
    Built with Streamlit • scikit-learn • Optuna • SHAP • Plotly
</div>
""", unsafe_allow_html=True)
