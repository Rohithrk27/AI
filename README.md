<<<<<<< HEAD
# 🧬 OptiLab — AI-Assisted Experimental Optimization Platform

**OptiLab** is a domain-agnostic, AI-assisted experimental optimization platform. It helps researchers find optimal experimental conditions with fewer experiments by combining machine learning with Bayesian Optimization.
=======
# 🧬 AI-Assisted Experimental Optimization Platform

This is a domain-agnostic, AI-assisted experimental optimization platform. It helps researchers find optimal experimental conditions with fewer experiments by combining machine learning with Bayesian Optimization.
>>>>>>> origin/master

## ✨ Features

- **📄 Smart Data Import** — Upload CSV, Excel, or native Design-Expert / Minitab / JMP exports with auto-format detection
- **🤖 7 ML Surrogate Models** — Gaussian Process, Random Forest, XGBoost, LightGBM, CatBoost, Neural Network, SVR
- **⚙️ Auto Hyperparameter Tuning** — Optuna-powered optimization for every model
- **🎯 Bayesian Optimization** — EI, UCB, and PI acquisition functions to recommend next experiments
- **🔬 SHAP Explainability** — Understand which variables drive your response
- **🎛️ Multi-Objective Optimization** — Optimize multiple responses simultaneously with Pareto front
- **📄 Publication Reports** — Export PDF reports with ANOVA tables, model comparisons, and optimal conditions
- **💾 Experiment History** — SQLite persistence for past sessions and model versions

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd c:\Users\ROHITH\Downloads\AI
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the application

```bash
.\venv\Scripts\streamlit.exe run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Upload data and optimize!

1. Upload your CSV/Excel file or use the sample citric acid dataset
2. Select factor and response columns
3. Preprocess (handle missing values, normalize)
4. Train all 7 models with auto hyperparameter tuning
5. Explore SHAP feature importance
6. Run Bayesian Optimization to get recommended next experiments
7. Enter new results and iterate
8. Download your PDF report

## 📁 Project Structure

```
AI/
├── app.py                          ← Main Streamlit app
├── requirements.txt                ← Python dependencies
├── .streamlit/config.toml          ← Dark theme config
├── modules/
│   ├── data/                       ← Data loading, parsing, preprocessing
│   │   ├── loader.py
│   │   ├── rsm_parsers.py         ← Design-Expert, Minitab, JMP parsers
│   │   ├── validator.py
│   │   └── preprocessor.py
│   ├── ml/                         ← Machine learning models
│   │   ├── models.py              ← 7 surrogate models
│   │   ├── tuner.py               ← Optuna hyperparameter tuning
│   │   ├── evaluator.py
│   │   └── model_selector.py
│   ├── optimization/               ← Bayesian Optimization
│   │   ├── acquisition.py         ← EI, UCB, PI
│   │   ├── bayesian_opt.py
│   │   └── multi_objective.py     ← Pareto optimization
│   ├── explainability/
│   │   └── shap_analysis.py       ← SHAP feature importance
│   ├── visualization/
│   │   └── plots.py               ← All Plotly charts
│   └── reporting/
│       └── report_generator.py    ← PDF report export
├── database/
│   └── history.py                 ← SQLite persistence
└── data/
    └── sample_citric_acid.csv     ← Demo dataset
```

## 🧪 Supported Domains

OptiLab works for **any experimental optimization** problem:

- 🧫 Biotechnology & Fermentation
- ⚗️ Chemical Engineering
- 🔬 Materials Science
- 💊 Drug Formulation
- 🌾 Agriculture
- 🍔 Food Science
- 🌍 Environmental Engineering
- 🔮 Nanotechnology

## 📊 ML Models

| Model | Library | Uncertainty |
|---|---|---|
| Gaussian Process | scikit-learn | Native σ(x) |
| Random Forest | scikit-learn | Tree variance |
| XGBoost | xgboost | Staged predictions |
| LightGBM | lightgbm | Staged predictions |
| CatBoost | catboost | Virtual ensembles |
| Neural Network | PyTorch | MC Dropout |
| SVR | scikit-learn | Distance heuristic |

## 📄 License

This project is for research and educational purposes.
