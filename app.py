"""
OptiLab — AI-Assisted Experimental Optimization Platform
Main Streamlit Application
"""

import streamlit as st
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass



def change_phase(new_phase):
    st.session_state.current_phase = new_phase

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="OptiLab — AI Experimental Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for Premium Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    /* ═══════════════════════════════════════
       ANIMATIONS
       ═══════════════════════════════════════ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3); }
        50%      { box-shadow: 0 4px 30px rgba(67, 233, 123, 0.55); }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes borderRotate {
        0%   { --angle: 0deg; }
        100% { --angle: 360deg; }
    }
    @keyframes floatIn {
        from { opacity: 0; transform: translateY(12px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes dotPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(67, 233, 123, 0.5); }
        50%      { box-shadow: 0 0 0 6px rgba(67, 233, 123, 0); }
    }

    /* ═══════════════════════════════════════
       GLOBAL TYPOGRAPHY
       ═══════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.3px;
    }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.35rem !important; }

    /* ═══════════════════════════════════════
       ANIMATED BACKGROUND
       ═══════════════════════════════════════ */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a2333 0%, #0a0e17 60%, #05070a 100%);
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 600px 300px at 20% 15%, rgba(67,233,123,0.04), transparent),
            radial-gradient(ellipse 500px 400px at 80% 80%, rgba(56,249,215,0.03), transparent),
            radial-gradient(ellipse 400px 250px at 60% 40%, rgba(96,165,250,0.03), transparent);
        background-size: 200% 200%;
        animation: gradientShift 20s ease infinite;
        pointer-events: none;
        z-index: 0;
    }

    /* ═══════════════════════════════════════
       PAGE CONTENT ANIMATION
       ═══════════════════════════════════════ */
    .main .block-container {
        animation: fadeInUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    /* ═══════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1522 0%, #0a0e17 100%);
        border-right: 1px solid rgba(67, 233, 123, 0.12);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #94a3b8 !important;
        font-weight: 500;
        padding: 8px 12px;
        border-radius: 10px;
        transition: all 0.25s ease;
        border-left: 3px solid transparent;
    }
    section[data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(67, 233, 123, 0.06);
        color: #e2e8f0 !important;
        border-left-color: rgba(67, 233, 123, 0.4);
    }

    /* Sidebar stepper styles */
    .stepper-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        position: relative;
    }
    .stepper-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
        transition: all 0.3s ease;
    }
    .stepper-dot.completed {
        background: #43e97b;
        box-shadow: 0 0 8px rgba(67, 233, 123, 0.5);
    }
    .stepper-dot.active {
        background: #43e97b;
        animation: dotPulse 1.5s ease-in-out infinite;
    }
    .stepper-dot.locked {
        background: #334155;
        border: 1px solid #475569;
    }
    .stepper-label {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .stepper-label.active {
        color: #43e97b;
        font-weight: 600;
    }
    .stepper-label.completed {
        color: #6ee7a0;
    }
    .stepper-line {
        width: 2px; height: 14px;
        background: #1e293b;
        margin-left: 4px;
    }
    .stepper-line.completed {
        background: rgba(67, 233, 123, 0.4);
    }

    /* ═══════════════════════════════════════
       GLASSMORPHISM CARDS — Rotating Border
       ═══════════════════════════════════════ */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
        transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                    box-shadow 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                    border-color 0.35s ease;
        animation: floatIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
        position: relative;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 50px -10px rgba(67, 233, 123, 0.12);
        border-color: rgba(67, 233, 123, 0.25);
    }
    .glass-card::after {
        content: '';
        position: absolute;
        inset: -1px;
        border-radius: 21px;
        padding: 1px;
        background: conic-gradient(from 0deg, transparent 60%, rgba(67,233,123,0.3) 80%, transparent 100%);
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
        animation: borderRotate 4s linear infinite;
    }
    .glass-card:hover::after {
        opacity: 1;
    }

    /* ═══════════════════════════════════════
       GRADIENT TEXT
       ═══════════════════════════════════════ */
    .gradient-text {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .gradient-text-blue {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* ═══════════════════════════════════════
       PRIMARY BUTTONS — Pulsing Glow
       ═══════════════════════════════════════ */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #43e97b 0%, #22c55e 100%) !important;
        border: none !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        border-radius: 12px !important;
        animation: pulseGlow 2.5s ease-in-out infinite !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(67, 233, 123, 0.5) !important;
        animation: none !important;
    }

    /* Secondary buttons */
    [data-testid="baseButton-secondary"] {
        border: 1px solid rgba(67, 233, 123, 0.25) !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="baseButton-secondary"]:hover {
        border-color: rgba(67, 233, 123, 0.5) !important;
        color: #e2e8f0 !important;
        background: rgba(67, 233, 123, 0.06) !important;
    }

    /* ═══════════════════════════════════════
       DATAFRAMES & EXPANDERS
       ═══════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        animation: floatIn 0.4s ease both;
    }
    [data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] > details > summary {
        background: rgba(255, 255, 255, 0.02);
    }

    /* ═══════════════════════════════════════
       HERO SECTION
       ═══════════════════════════════════════ */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px;
        animation: fadeInUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        font-family: 'Outfit', sans-serif;
        letter-spacing: -2px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 12px;
        margin-bottom: 40px;
        letter-spacing: 0.3px;
    }

    /* Feature pills */
    .feature-pill {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(67, 233, 123, 0.08);
        border: 1px solid rgba(67, 233, 123, 0.25);
        border-radius: 20px;
        color: #43e97b;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0 6px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        transition: all 0.25s ease;
    }
    .feature-pill:hover {
        background: rgba(67, 233, 123, 0.15);
        border-color: rgba(67, 233, 123, 0.5);
        transform: translateY(-1px);
    }

    /* ═══════════════════════════════════════
       WORKFLOW PIPELINE
       ═══════════════════════════════════════ */
    .workflow-step {
        padding: 14px 22px;
        background: rgba(15, 23, 42, 0.55);
        border-left: 4px solid;
        border-radius: 10px;
        margin-bottom: 10px;
        font-weight: 500;
        display: flex;
        align-items: center;
        color: #f8fafc;
        transition: transform 0.25s ease, background 0.25s ease;
        animation: floatIn 0.4s ease both;
    }
    .workflow-step:hover {
        transform: translateX(6px);
        background: rgba(30, 41, 59, 0.75);
    }
    .workflow-arrow {
        text-align: center;
        color: #334155;
        margin-bottom: 8px;
        font-size: 1rem;
    }

    /* ═══════════════════════════════════════
       METRIC CARDS
       ═══════════════════════════════════════ */
    .metric-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        transition: all 0.3s ease;
        animation: floatIn 0.4s ease both;
    }
    .metric-card:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(67, 233, 123, 0.15);
        transform: translateY(-2px);
    }

    /* ═══════════════════════════════════════
       PROGRESS / LOADING HELPERS
       ═══════════════════════════════════════ */
    .progress-checklist {
        list-style: none;
        padding: 0;
        margin: 10px 0;
    }
    .progress-checklist li {
        padding: 6px 0;
        font-size: 0.92rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 8px;
        animation: floatIn 0.35s ease both;
    }
    .progress-checklist li.done { color: #43e97b; }
    .progress-checklist li.active { color: #60a5fa; }

    @keyframes spin { to { transform: rotate(360deg); } }
    .mini-spinner {
        display: inline-block;
        width: 14px; height: 14px;
        border: 2px solid rgba(67, 233, 123, 0.2);
        border-top-color: #43e97b;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    /* Success banner */
    .success-banner {
        background: linear-gradient(135deg, rgba(67,233,123,0.1) 0%, rgba(56,249,215,0.08) 100%);
        border: 1px solid rgba(67, 233, 123, 0.3);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        animation: floatIn 0.5s ease both;
    }
    .success-banner .big-score {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #43e97b, #38f9d7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ═══════════════════════════════════════
       FOOTER
       ═══════════════════════════════════════ */
    .optilab-footer {
        text-align: center;
        padding: 16px 0 8px;
        color: #334155;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
    }
    .optilab-footer .version-badge {
        display: inline-block;
        padding: 2px 10px;
        background: rgba(67, 233, 123, 0.08);
        border: 1px solid rgba(67, 233, 123, 0.15);
        border-radius: 12px;
        color: #43e97b;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 6px;
        letter-spacing: 0.8px;
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
PHASES = [
    "🏠 Home",
    "1️⃣ Data Input & DOE",
    "2️⃣ Preprocessing",
    "3️⃣ Model Training",
    "4️⃣ Bayesian Optimization",
    "5️⃣ Iterate (New Results)",
    "6️⃣ Report & History"
]

def get_step_status(page):
    return "inactive"

if "current_phase" not in st.session_state:
    st.session_state.current_phase = "🏠 Home"

with st.sidebar:
    st.markdown('''
    <div style="text-align: center; padding: 24px 0 12px 0;">
        <div style="font-size: 2.8rem;">🧬</div>
        <div class="gradient-text" style="font-size: 1.9rem; letter-spacing: -1px;">OptiLab</div>
        <div style="color: #475569; font-size: 0.75rem; margin-top: 6px; letter-spacing: 1.5px; text-transform: uppercase;">AI Experimental Optimizer</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("---")

    st.radio(
        "Navigation Workflow",
        PHASES,
        key="current_phase",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Animated progress stepper
    stepper_steps = [
        ("Data Upload", st.session_state.raw_data is not None),
        ("Preprocessed", st.session_state.preprocessed is not None),
        ("GP Trained", st.session_state.best_model is not None),
        ("Optimized", st.session_state.bo_recommendations is not None),
    ]

    stepper_html = '<div style="padding: 0 8px;">'
    stepper_html += '<div style="font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 600;">Progress</div>'

    for i, (label, is_done) in enumerate(stepper_steps):
        if is_done:
            dot_class = "completed"
            label_class = "completed"
        elif i == 0 or stepper_steps[i-1][1]:
            dot_class = "active"
            label_class = "active"
        else:
            dot_class = "locked"
            label_class = ""

        stepper_html += f'<div class="stepper-item"><div class="stepper-dot {dot_class}"></div><span class="stepper-label {label_class}">{"✓ " if is_done else ""}{label}</span></div>'
        if i < len(stepper_steps) - 1:
            line_class = "completed" if is_done else ""
            stepper_html += f'<div class="stepper-line {line_class}"></div>'

    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)

    st.markdown("---")

    # Reset session button
    def reset_session():
        for key in list(st.session_state.keys()):
            if key != "current_phase":
                del st.session_state[key]
        st.session_state.current_phase = "🏠 Home"

    st.button("🔄 Reset Session", use_container_width=True, on_click=reset_session)


# ──────────────────────────────────────────────
# Page: Home
# ──────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="hero-container animate-in">
        <div class="hero-title">
            <span class="gradient-text">OptiLab</span>
        </div>
        <p class="hero-subtitle">
            AI-Assisted Experimental Optimization Platform<br/>
            <span style="color: #667eea;">Train • Optimize • Discover</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    st.markdown("""
    <div class="glass-card animate-in" style="max-width: 800px; margin: 0 auto; text-align: center; padding: 40px;">
        <h3 class="gradient-text-blue" style="margin-top: 0;">What is OptiLab?</h3>
        <p style="font-size: 1.1rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 20px;">
            OptiLab is a next-generation AI platform designed to replace classical Response Surface Methodology (RSM) with advanced Machine Learning. 
        </p>
        <p style="font-size: 1.1rem; color: #cbd5e1; line-height: 1.6;">
            By combining state-of-the-art surrogate modeling with Bayesian Optimization, OptiLab helps researchers discover absolute optimum conditions faster, more accurately, and with fewer physical experiments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Workflow diagram
    st.markdown('<h3 class="gradient-text-blue" style="text-align: center;">How It Works</h3>', unsafe_allow_html=True)

    workflow_steps = [
        ("📄", "Upload your experimental data", "border-color: #667eea;"),
        ("🔍", "Auto-detect format and variable types", "border-color: #764ba2;"),
        ("🧹", "Preprocess: handle missing values, normalize, encode", "border-color: #f093fb;"),
        ("🤖", "Train Gaussian Process surrogate with auto-tuning", "border-color: #ec4899;"),
        ("🏆", "Optimize Gaussian Process hyperparameters", "border-color: #f5af19;"),
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
        st.button("🚀 Get Started", use_container_width=True, type="primary", on_click=change_phase, args=("1️⃣ Data Input & DOE",))

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
    st.markdown('<h2 class="gradient-text">📄 Data Input & DOE</h2>', unsafe_allow_html=True)
    st.markdown("Upload your experimental data or generate a new Design of Experiments (DOE) matrix.")

    # Import modules
    from modules.data.loader import load_file
    from modules.data.rsm_parsers import smart_import, detect_source_format
    from modules.data.validator import detect_variable_types, validate_dataset, compute_statistics
    from modules.data.doe_generator import generate_doe

    tab1, tab2 = st.tabs(["📂 Upload Existing Data", "🧬 Generate New DOE Matrix"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Drop your data file here",
                type=["csv", "xlsx", "xls", "txt"],
                help="Supported: CSV, Excel, Design-Expert export, Minitab export, JMP export"
            )

            use_sample = st.checkbox("📂 Use sample dataset (Citric Acid Adsorption)", help="Loads a pre-configured biological dataset for testing.")

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
                    OptiLab automatically detects your file format and extracts factors, responses, and design type.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Load data
        data_source = None
        if use_sample:
            import os
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

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            design_type = st.selectbox("Design Type", ["Box-Behnken (BBD)", "Central Composite (CCD)"])
        with col2:
            n_factors = st.number_input("Number of Factors", min_value=2, max_value=10, value=3)
            n_responses = st.number_input("Number of Responses", min_value=1, max_value=5, value=1)
            
        st.markdown("---")
        st.markdown("**Define Factors:**")
        factors = []
        for i in range(int(n_factors)):
            f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
            with f_col1:
                f_name = st.text_input(f"Factor {i+1} Name", value=f"Factor_{i+1}", key=f"f_name_{i}")
            with f_col2:
                f_min = st.number_input(f"Min", value=-1.0, key=f"f_min_{i}")
            with f_col3:
                f_max = st.number_input(f"Max", value=1.0, key=f"f_max_{i}")
            factors.append({"name": f_name, "min": f_min, "max": f_max})
            
        st.markdown("**Define Responses:**")
        resp_names = []
        r_cols = st.columns(min(3, int(n_responses)))
        for i in range(int(n_responses)):
            with r_cols[i % 3]:
                r_name = st.text_input(f"Response {i+1} Name", value=f"Response_{i+1}", key=f"r_name_{i}")
                resp_names.append(r_name)
                
        if st.button("🧬 Generate DOE Matrix", type="primary"):
            import numpy as np
            dtype = "BBD" if "Box-Behnken" in design_type else "CCD"
            try:
                doe_df = generate_doe(factors, design_type=dtype)
                # Add empty response columns
                for r_name in resp_names:
                    doe_df[r_name] = np.nan
                st.session_state.raw_data = doe_df
                st.session_state.source_format = "generated"
                st.session_state.parsed_metadata = {
                    "factors": [{"name": f["name"], "low": f["min"], "high": f["max"]} for f in factors],
                    "responses": [{"name": r} for r in resp_names],
                    "design_type": dtype
                }
                st.success("DOE Matrix generated successfully! You can now fill out the response columns in the table below.")
            except Exception as e:
                st.error(f"Error generating DOE: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.raw_data is not None:

        # Data preview
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">Data Preview (Editable)</h3>', unsafe_allow_html=True)
        st.markdown("You can edit the data directly in the table below. All numerical columns are forced to floats.")
        
        # Use data_editor so user can modify data directly.
        edited_df = st.data_editor(st.session_state.raw_data, use_container_width=True, num_rows="dynamic")
        
        # Force all columns to float where possible, to fulfill the "all factors are float" requirement.
        for col in edited_df.columns:
            try:
                edited_df[col] = edited_df[col].astype(float)
            except Exception:
                pass
                
        st.session_state.raw_data = edited_df

        # Provide a way to download the template/edited data
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Data as CSV",
            data=csv,
            file_name="OptiLab_Data.csv",
            mime="text/csv",
        )

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
            st.button("➡️ Proceed to Preprocessing", use_container_width=True, type="primary", on_click=change_phase, args=("2️⃣ Preprocessing",))


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
        import time
        progress_container = st.empty()

        # Step 1: Missing values
        progress_container.markdown('''<ul class="progress-checklist">
            <li class="active"><span class="mini-spinner"></span> Handling missing values...</li>
        </ul>''', unsafe_allow_html=True)

        config = {
            "missing_strategy": missing_strategy,
            "normalization": norm_method,
        }
        result = preprocess_pipeline(df, factor_cols, response_cols, config)

        # Step 2: Normalization done
        missing_count_resolved = df[factor_cols + response_cols].isnull().sum().sum()
        progress_container.markdown(f'''<ul class="progress-checklist">
            <li class="done">✅ Missing values handled ({missing_count_resolved} found)</li>
            <li class="active"><span class="mini-spinner"></span> Normalizing features ({norm_method})...</li>
        </ul>''', unsafe_allow_html=True)
        time.sleep(0.3)

        # Step 3: All done
        progress_container.markdown(f'''<ul class="progress-checklist">
            <li class="done">✅ Missing values handled ({missing_count_resolved} found)</li>
            <li class="done">✅ Features normalized ({norm_method.title()}Scaler)</li>
            <li class="done">✅ Dataset validated — {result['X'].shape[0]} samples, {result['X'].shape[1]} features</li>
        </ul>''', unsafe_allow_html=True)

        st.session_state.preprocessed = result
        st.toast("Preprocessing complete!", icon="✅")

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
        st.button("➡️  Proceed to Model Training", use_container_width=True, type="primary", on_click=change_phase, args=("3️⃣ Model Training",))

    elif st.session_state.preprocessed is not None:
        result = st.session_state.preprocessed
        st.markdown(f'''<ul class="progress-checklist">
            <li class="done">✅ Data already preprocessed — {result['X'].shape[0]} samples, {result['X'].shape[1]} features</li>
        </ul>''', unsafe_allow_html=True)

        st.button("➡️  Proceed to Model Training", use_container_width=True, type="primary", on_click=change_phase, args=("3️⃣ Model Training",))


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
        enable_tuning = st.checkbox("⚙️ Enable Auto-Tuning (Optuna)", value=True, help="Automatically searches for the best hyperparameter configuration for each model to maximize R-squared.")
    with col2:
        n_trials = st.number_input("Trials per model", min_value=10, max_value=200, value=50, step=10, help="How many different hyperparameter combinations Optuna will test per model.")
    with col3:
        cv_folds = st.number_input("CV Folds", min_value=3, max_value=10, value=5, help="Number of cross-validation splits. Higher folds give a more robust evaluation but take longer to train.")
    with col4:
        timeout = st.number_input("Timeout per model (sec)", min_value=10, max_value=300, value=60, help="Maximum time limit (in seconds) to spend tuning a single model.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Train button
    if st.button("🚀  Train Selected Models", use_container_width=True, type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return
            
        models = {name: models_dict[name] for name in selected_models}
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        all_results = {}
        tuning_results = {}

        for i, (name, model) in enumerate(models.items()):
            progress_frac = i / len(models)
            progress_bar.progress(progress_frac, text=f"Training {name}... ({i+1}/{len(models)})")

            try:
                if enable_tuning:
                    status_text.markdown(f'''<ul class="progress-checklist">
                        <li class="active"><span class="mini-spinner"></span> Tuning <strong>{name}</strong> kernel &bull; {n_trials} trials &bull; {cv_folds}-fold CV...</li>
                    </ul>''', unsafe_allow_html=True)

                    tune_result = tune_all_models(
                        {name: model}, X, y,
                        n_trials=n_trials, cv_folds=cv_folds, timeout=timeout
                    )
                    if name in tune_result:
                        tuning_results[name] = tune_result[name]
                        best_cv = tune_result[name].get("best_cv_score", 0)
                        status_text.markdown(f'''<ul class="progress-checklist">
                            <li class="done">✅ Tuning complete &bull; Best CV R²: <strong>{best_cv:.4f}</strong></li>
                            <li class="active"><span class="mini-spinner"></span> Training final model with best params...</li>
                        </ul>''', unsafe_allow_html=True)
                        model.train(X, y, params=tune_result[name].get("best_params"))
                    else:
                        model.train(X, y)
                else:
                    status_text.markdown(f'''<ul class="progress-checklist">
                        <li class="active"><span class="mini-spinner"></span> Training <strong>{name}</strong> (no tuning)...</li>
                    </ul>''', unsafe_allow_html=True)
                    model.train(X, y)

                metrics = evaluate_model(model, X, y, cv_folds=cv_folds)
                all_results[name] = metrics
                st.session_state.trained_models[name] = model

            except Exception as e:
                st.warning(f"⚠️ {name} failed: {str(e)}")
                all_results[name] = {"error": str(e)}

        progress_bar.progress(1.0, text="✅ Training complete!")

        st.session_state.evaluation_results = all_results
        st.session_state.tuning_results = tuning_results

        # Auto-select best
        best_name, best_model = auto_select_best(all_results, st.session_state.trained_models)
        st.session_state.best_model_name = best_name
        st.session_state.best_model = best_model

        # Success banner with big R² score
        best_r2 = all_results.get(best_name, {}).get("R²", all_results.get(best_name, {}).get("CV_R²_mean", 0))
        if isinstance(best_r2, (int, float)):
            status_text.markdown(f'''
            <div class="success-banner">
                <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Gaussian Process R²</div>
                <div class="big-score">{best_r2:.4f}</div>
                <div style="font-size: 0.8rem; color: #6ee7a0; margin-top: 4px;">Model trained successfully</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            status_text.empty()

        st.toast("Model training complete!", icon="🏆")

    # Display results
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">Model Benchmarking Results</h3>', unsafe_allow_html=True)

        with st.expander("📊 Input Classical RSM Baseline (Optional)"):
            st.markdown("Upload your classical Design-Expert (RSM) results to benchmark against the AI models.")
            col1, col2 = st.columns(2)
            with col1:
                rsm_r2 = st.number_input("RSM R² Score", min_value=-10.0, max_value=1.0, value=0.0, step=0.01, help="Enter the R-squared value reported by your classical RSM software (e.g., Design-Expert).")
            with col2:
                rsm_rmse = st.number_input("RSM RMSE (Error)", min_value=0.0, value=0.0, step=0.01, help="Enter the Root Mean Square Error reported by your classical RSM software.")
            
            if st.button("Save RSM Baseline"):
                st.session_state.rsm_baseline = {"R²": rsm_r2, "RMSE": rsm_rmse}
                st.toast("RSM Baseline saved!", icon="📊")

        # Ranking table
        ranking_df = rank_models(st.session_state.evaluation_results)
        
        # Inject RSM Baseline
        if "rsm_baseline" in st.session_state and st.session_state.rsm_baseline.get("R²", 0) != 0.0:
            import pandas as pd
            rsm_row = pd.DataFrame([{
                "Model": "Classical RSM (Uploaded)",
                "R²": st.session_state.rsm_baseline["R²"],
                "Adj_R²": "-",
                "RMSE": st.session_state.rsm_baseline["RMSE"],
                "MAE": "-",
                "MAPE_%": "-",
                "CV_R²_mean": "-",
                "CV_R²_std": "-",
                "CV_RMSE_mean": "-"
            }])
            ranking_df = pd.concat([ranking_df, rsm_row], ignore_index=True)
            
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



        st.markdown("<br/>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.button("➡️ Proceed to Optimization", use_container_width=True, type="primary", on_click=change_phase, args=("4️⃣ Bayesian Optimization",))
def page_optimize():
    class SingleObjectiveProxy:
        def __init__(self, base_model, target_idx):
            self.base_model = base_model
            self.target_idx = target_idx
        def predict_with_uncertainty(self, X):
            mean, std = self.base_model.predict_with_uncertainty(X)
            if mean.ndim > 1 and mean.shape[1] > 1:
                return mean[:, self.target_idx], std[:, self.target_idx]
            return mean, std
        def predict(self, X):
            mean = self.base_model.predict(X)
            if mean.ndim > 1 and mean.shape[1] > 1:
                return mean[:, self.target_idx]
            return mean

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
        opt_type = st.radio("Optimization Mode", ["Single Objective", "Multi-Objective"], help="Choose whether you want to optimize one specific response (e.g., just Yield) or balance multiple conflicting responses (e.g., maximizing Yield while minimizing Cost).")
    
    if opt_type == "Single Objective":
        target_col = st.selectbox("Target Response", response_cols, help="Select the specific response you want the AI to optimize.")
        target_idx = response_cols.index(target_col)
        optimize_direction = st.selectbox("Objective Direction", ["Maximize", "Minimize"], help="Choose whether the AI should try to push the response value as high as possible or as low as possible.")
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
        acq_func = st.selectbox("Acquisition Function", ["Expected Improvement (EI)", "Upper Confidence Bound (UCB)", "Probability of Improvement (PI)"], help="EI: Balances exploration/exploitation (Best default). UCB: Optimistic exploration. PI: Safest improvements.")
    with col2:
        n_recommend = st.number_input("Number of experiments to suggest", min_value=1, max_value=20, value=5, help="How many unique, optimal theoretical experiments the AI should recommend to you.")
    with col3:
        n_candidates = st.number_input("Candidate pool size", min_value=1000, max_value=100000, value=20000, step=1000, help="The number of simulated factor combinations the AI will test behind the scenes to find the absolute peak.")
    st.markdown('</div>', unsafe_allow_html=True)

    acq_map = {
        "Expected Improvement (EI)": "EI",
        "Upper Confidence Bound (UCB)": "UCB",
        "Probability of Improvement (PI)": "PI",
    }

    with st.expander("📈 Input Classical RSM Optimal Conditions (Optional)"):
        st.markdown("If you calculated an optimal point using Classical RSM, input it here. OptiLab will compare your classical optimum to the AI optimum.")
        
        if "rsm_optimum" not in st.session_state:
            st.session_state.rsm_optimum = {}
            
        rsm_cols = st.columns(min(4, len(feature_names)))
        for i, fname in enumerate(feature_names):
            with rsm_cols[i % len(rsm_cols)]:
                val = st.number_input(f"{fname}", value=0.0, format="%.4f", step=0.01, key=f"rsm_opt_{fname}")
                st.session_state.rsm_optimum[fname] = float(val)
                
        rsm_r2 = st.number_input("R-Squared (R²) at this Optimum", value=0.0, format="%.4f", step=0.01)
        if st.button("Save RSM Optimum"):
            st.session_state.rsm_optimum["_rsm_r2"] = rsm_r2
            st.toast("RSM Optimum saved!", icon="✅")

    st.markdown("---")
    st.markdown('<h3 class="gradient-text-blue">🔍 Search Space Bounds</h3>', unsafe_allow_html=True)
    st.markdown("State the minimum and maximum values for each continuous factor. The AI will strictly search for optimal conditions within these limits.")
    
    user_bounds = []
    if "scaler" in st.session_state.preprocessed:
        scaler = st.session_state.preprocessed["scaler"]
        X_orig = st.session_state.preprocessed["X_original"]
        
        # Display inputs in columns
        bound_cols = st.columns(min(3, len(feature_names)))
        b_idx = 0
        
        for i, fname in enumerate(feature_names):
            is_cat = False
            for cat_col in st.session_state.factor_cols:
                if st.session_state.variable_types.get(cat_col) == "categorical" and fname.startswith(cat_col + "_"):
                    is_cat = True
                    break
            
            if not is_cat:
                orig_min = float(X_orig[:, i].min())
                orig_max = float(X_orig[:, i].max())
                
                with bound_cols[b_idx % len(bound_cols)]:
                    st.markdown(f"**{fname}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        umin = st.number_input("Min", value=orig_min, step=0.1, key=f"min_{fname}")
                    with c2:
                        umax = st.number_input("Max", value=orig_max, step=0.1, key=f"max_{fname}")
                        
                # Transform to scaled space
                dmin = np.zeros(len(feature_names))
                dmin[i] = umin
                dmax = np.zeros(len(feature_names))
                dmax[i] = umax
                
                smin = scaler.transform([dmin])[0][i]
                smax = scaler.transform([dmax])[0][i]
                user_bounds.append((smin, smax))
                b_idx += 1
            else:
                user_bounds.append((0.0, 1.0))
    else:
        user_bounds = list(zip(X.min(axis=0), X.max(axis=0)))
        
    st.markdown("---")

    if st.button("🎯  Run Bayesian Optimization", use_container_width=True, type="primary"):
        import time
        bounds = user_bounds
        opt_status = st.empty()

        # Step 1: Generate candidates
        opt_status.markdown(f'''<ul class="progress-checklist">
            <li class="active"><span class="mini-spinner"></span> Generating {n_candidates:,} candidate points across search space...</li>
        </ul>''', unsafe_allow_html=True)

        if opt_type == "Single Objective":
            target_idx = response_cols.index(target_col)
            if y.ndim > 1 and y.shape[1] > 1:
                y_target = y[:, target_idx]
            else:
                y_target = y.ravel()
            
            y_best = y_target.max() if optimize_direction == "Maximize" else y_target.min()

            # Step 2: Acquisition function
            opt_status.markdown(f'''<ul class="progress-checklist">
                <li class="done">✅ Generated {n_candidates:,} candidate points</li>
                <li class="active"><span class="mini-spinner"></span> Evaluating {acq_func} acquisition function...</li>
            </ul>''', unsafe_allow_html=True)
            
            recommendations = recommend_experiments(
                model=SingleObjectiveProxy(model, target_idx),
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
            models = [SingleObjectiveProxy(model, i) for i in range(len(response_cols))]

            opt_status.markdown(f'''<ul class="progress-checklist">
                <li class="done">✅ Generated {n_candidates:,} candidate points</li>
                <li class="active"><span class="mini-spinner"></span> Running multi-objective Pareto optimization...</li>
            </ul>''', unsafe_allow_html=True)
            
            recommendations = multi_objective_recommend(
                models=models,
                bounds=bounds,
                feature_names=feature_names,
                objectives_config=objectives_config,
                n_recommend=n_recommend,
                n_candidates=n_candidates
            )

        # Step 3: Done
        opt_status.markdown(f'''<ul class="progress-checklist">
            <li class="done">✅ Generated {n_candidates:,} candidate points</li>
            <li class="done">✅ Acquisition function evaluated</li>
            <li class="done">✅ Top {n_recommend} experiments selected</li>
        </ul>''', unsafe_allow_html=True)

        st.session_state.bo_recommendations = recommendations
        st.toast("Optimization complete!", icon="🎯")

    if st.session_state.bo_recommendations is not None:
        st.markdown("---")
        st.markdown('<h3 class="gradient-text-blue">📋 Recommended Next Experiments</h3>', unsafe_allow_html=True)
        st.markdown("Perform these experiments in your lab, then enter the results in the **🔄 Iterate** page.")
        st.info("**Data Scientist Insight:** Bayesian Optimization uses uncertainty to find the absolute best conditions. The table below shows the specific experiments you should run next in your lab to efficiently maximize your results.")

        # Inverse-transform to original scale if scaler exists
        recs = st.session_state.bo_recommendations.copy()
        if "scaler" in st.session_state.preprocessed:
            scaler = st.session_state.preprocessed["scaler"]
            scaled_factors = recs[feature_names].values
            orig_factors = scaler.inverse_transform(scaled_factors)
            for i, fname in enumerate(feature_names):
                recs[fname] = [round(val, 4) for val in orig_factors[:, i]]

        st.dataframe(recs, use_container_width=True)

        if opt_type == "Single Objective":
            st.markdown("---")
            st.markdown('<h3 class="gradient-text-blue">⭐ Absolute Optimal Theoretical Conditions (AI)</h3>', unsafe_allow_html=True)
            st.markdown("According to the surrogate model's predicted response surface, setting your factors to exactly these values will yield the absolute best possible outcome.")
            
            from modules.optimization.bayesian_opt import generate_candidates
            dense_candidates = generate_candidates(user_bounds, n_candidates=50000)
            
            target_idx = response_cols.index(target_col)
            proxy = SingleObjectiveProxy(model, target_idx)
            mean_preds = proxy.predict(dense_candidates)
            
            if optimize_direction == "Maximize":
                best_idx = int(np.argmax(mean_preds))
            else:
                best_idx = int(np.argmin(mean_preds))
                
            best_pred = mean_preds[best_idx]
            best_point_scaled = dense_candidates[best_idx]
            
            if "scaler" in st.session_state.preprocessed:
                best_point_orig = st.session_state.preprocessed["scaler"].inverse_transform([best_point_scaled])[0]
            else:
                best_point_orig = best_point_scaled
                
            best_dict = {}
            for f, v in zip(feature_names, best_point_orig):
                best_dict[f] = round(float(v), 4)
            
            import pandas as pd
            st.dataframe(pd.DataFrame([best_dict]), use_container_width=True)
            st.success(f"**Predicted {target_col}:** {best_pred:.4f}")
            
            # Hybrid Comparison
            if "rsm_optimum" in st.session_state and "_rsm_r2" in st.session_state.rsm_optimum:
                st.markdown("---")
                st.markdown('<h3 class="gradient-text-blue">⚖️ Hybrid Comparison: AI vs Classical RSM</h3>', unsafe_allow_html=True)
                
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown("**Classical Statistical Optimum (RSM)**")
                    rsm_disp = {k: v for k, v in st.session_state.rsm_optimum.items() if k != "_rsm_r2"}
                    st.dataframe(pd.DataFrame([rsm_disp]), use_container_width=True)
                    st.info(f"**RSM R² Score:** {st.session_state.rsm_optimum['_rsm_r2']:.4f}")
                    
                with comp_col2:
                    st.markdown("**Modern AI Optimum (ML Surrogate)**")
                    st.dataframe(pd.DataFrame([best_dict]), use_container_width=True)
                    st.success(f"**Predicted Response:** {best_pred:.4f}")



        st.button("🔄 Proceed to Iterate (Enter New Results)", use_container_width=True, type="primary", on_click=change_phase, args=("5️⃣ Iterate (New Results)",))


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

    n_new = st.number_input("Number of new experiments to enter", min_value=1, max_value=20, value=1, help="How many new physical lab results you want to feed back into the AI to improve its accuracy.")

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

    # Allow user to proceed to report without iterating
    st.markdown("---")
    st.button("➡️ Proceed to Report & History", use_container_width=True, type="primary", on_click=change_phase, args=("6️⃣ Report & History",))


# ──────────────────────────────────────────────
# Page: Report
# ──────────────────────────────────────────────
def page_report():

    if st.session_state.best_model is None:
        st.warning("⬅️ Please complete the optimization workflow first.")
        return

    from modules.reporting.report_generator import generate_report

    st.markdown("""
    Generate a publication-ready PDF report containing your full optimization workflow.
    """)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("**Report will include:**")
    sections = [
        "Methodology (GP + Bayesian Optimization)",
        "Dataset Summary (factors, responses, sample size)",
        "Gaussian Process Evaluation Metrics",
        "Tuned Hyperparameters (Optuna)",
        "Optimization Results & Recommended Conditions",
        "Convergence History",
    ]
    for s in sections:
        st.markdown(f"- {s}")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("📄  Generate PDF Report", use_container_width=True, type="primary"):
        import time
        report_status = st.empty()

        report_status.markdown('''<ul class="progress-checklist">
            <li class="active"><span class="mini-spinner"></span> Building methodology section...</li>
        </ul>''', unsafe_allow_html=True)

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

        report_status.markdown('''<ul class="progress-checklist">
            <li class="done">✅ Methodology section</li>
            <li class="active"><span class="mini-spinner"></span> Building dataset summary...</li>
        </ul>''', unsafe_allow_html=True)
        time.sleep(0.2)

        report_status.markdown('''<ul class="progress-checklist">
            <li class="done">✅ Methodology section</li>
            <li class="done">✅ Dataset summary</li>
            <li class="active"><span class="mini-spinner"></span> Compiling PDF...</li>
        </ul>''', unsafe_allow_html=True)

        pdf_bytes = generate_report(report_data)

        report_status.markdown('''<ul class="progress-checklist">
            <li class="done">✅ Methodology section</li>
            <li class="done">✅ Dataset summary</li>
            <li class="done">✅ PDF compiled successfully</li>
        </ul>''', unsafe_allow_html=True)

        st.toast("Report generated!", icon="📄")
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name="OptiLab_Optimization_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# Page: History
# ──────────────────────────────────────────────
def page_history():
    st.markdown('<h2 class="gradient-text">📚 Report & History</h2>', unsafe_allow_html=True)
    st.markdown("Download your publication-ready PDF report or view current session statistics below.")

    # Render the report generator here instead of a separate page
    page_report()

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
# Phase Routers
# ──────────────────────────────────────────────
PHASE_FUNCTIONS = {
    "🏠 Home": page_home,
    "1️⃣ Data Input & DOE": page_upload,
    "2️⃣ Preprocessing": page_preprocessing,
    "3️⃣ Model Training": page_train,
    "4️⃣ Bayesian Optimization": page_optimize,
    "5️⃣ Iterate (New Results)": page_iterate,
    "6️⃣ Report & History": page_history,
}

# Render the selected phase
page_fn = PHASE_FUNCTIONS.get(st.session_state.current_phase, page_home)
page_fn()

# Footer
st.markdown("---")
st.markdown('''
<div class="optilab-footer">
    🧬 OptiLab <span class="version-badge">v2.0</span><br/>
    <span style="color: #1e293b;">—</span> AI-Assisted Experimental Optimization <span style="color: #1e293b;">—</span><br/>
    <span style="font-size: 0.7rem; color: #1e293b; margin-top: 4px; display: inline-block;">Streamlit &bull; Gaussian Process &bull; Bayesian Optimization &bull; Optuna</span>
</div>
''', unsafe_allow_html=True)

