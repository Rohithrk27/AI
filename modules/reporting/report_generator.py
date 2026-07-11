"""
Report Generator Module
Generates publication-ready PDF reports for the optimization workflow.
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF


class OptiLabReport(FPDF):
    """Custom PDF report with OptiLab branding."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(67, 233, 123)
        self.cell(0, 8, "OptiLab — AI-Assisted Experimental Optimization Report", align="R")
        self.ln(10)
        self.set_draw_color(67, 233, 123)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | OptiLab v2.0", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(67, 233, 123)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_table(self, df, col_widths=None):
        """Add a DataFrame as a table to the PDF."""
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(67, 233, 123)
        self.set_text_color(0, 0, 0)

        cols = list(df.columns)
        n_cols = len(cols)

        if col_widths is None:
            available = self.w - 20
            col_widths = [available / n_cols] * n_cols

        # Header
        for i, col in enumerate(cols):
            w = col_widths[i] if i < len(col_widths) else col_widths[-1]
            self.cell(w, 7, str(col)[:15], border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(40, 40, 40)

        for _, row in df.iterrows():
            for i, col in enumerate(cols):
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                val = row[col]
                if isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)[:15]

                # Alternate row colors
                if _ % 2 == 0:
                    self.set_fill_color(240, 250, 244)
                else:
                    self.set_fill_color(255, 255, 255)

                self.cell(w, 6, text, border=1, fill=True, align="C")
            self.ln()

        self.ln(4)


def generate_report(report_data: dict) -> bytes:
    """
    Generate a complete PDF report.
    
    Args:
        report_data: Dict containing all session data
        
    Returns:
        bytes: PDF content
    """
    pdf = OptiLabReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ──────────────────────────────────────
    # Title Page
    # ──────────────────────────────────────
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(67, 233, 123)
    pdf.cell(0, 15, "OptiLab", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "AI-Assisted Experimental Optimization Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")

    # ──────────────────────────────────────
    # Methodology
    # ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("1. Methodology")
    pdf.body_text(
        "OptiLab employs a Gaussian Process (GP) surrogate model combined with Bayesian Optimization "
        "to find optimal experimental conditions. The GP is a non-parametric, probabilistic model that "
        "smoothly interpolates between data points while intrinsically penalizing overly complex curves."
    )
    pdf.body_text(
        "Unlike classical Response Surface Methodology (RSM), which forces data into a rigid quadratic "
        "polynomial equation, the GP captures complex non-linear relationships in the experimental "
        "landscape. Because the GP outputs both a prediction (mean) and a mathematical uncertainty "
        "(standard deviation), Bayesian Optimization can use Acquisition Functions (Expected Improvement, "
        "Upper Confidence Bound, Probability of Improvement) to intelligently recommend the next "
        "experiment to run."
    )
    pdf.body_text(
        "Hyperparameters of the GP (kernel type, noise level, restarts) are automatically tuned using "
        "the Optuna framework with cross-validation, ensuring the best possible fit for the given data."
    )

    # ──────────────────────────────────────
    # 2. Dataset Summary
    # ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("2. Dataset Summary")

    raw_data = report_data.get("raw_data")
    factor_cols = report_data.get("factor_cols", [])
    response_cols = report_data.get("response_cols", [])

    if raw_data is not None:
        pdf.body_text(f"Total experiments: {len(raw_data)}")
        pdf.body_text(f"Factors ({len(factor_cols)}): {', '.join(factor_cols)}")
        pdf.body_text(f"Responses ({len(response_cols)}): {', '.join(response_cols)}")
        pdf.body_text(f"Iterations performed: {report_data.get('iteration_count', 0)}")

        # Data statistics
        pdf.subsection_title("Dataset Statistics")
        numeric_cols = factor_cols + response_cols
        stats = raw_data[numeric_cols].describe().round(4).reset_index()
        stats.columns = ["Statistic"] + numeric_cols
        pdf.add_table(stats)

    # ──────────────────────────────────────
    # 3. Model Benchmarking
    # ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("3. Gaussian Process Evaluation")

    eval_results = report_data.get("evaluation_results", {})
    if eval_results:
        rows = []
        for name, metrics in eval_results.items():
            if "error" in metrics:
                continue
            rows.append({
                "Model": name,
                "R2": metrics.get("R2", metrics.get("R²", "")),
                "CV R2": metrics.get("CV_R2_mean", metrics.get("CV_R²_mean", "")),
                "RMSE": metrics.get("RMSE", ""),
                "MAE": metrics.get("MAE", ""),
            })

        if rows:
            bench_df = pd.DataFrame(rows)
            pdf.add_table(bench_df)

        best_name = report_data.get("best_model_name", "N/A")
        pdf.body_text(f"Best model (auto-selected): {best_name}")

    # Tuning results
    tuning_results = report_data.get("tuning_results", {})
    if tuning_results:
        pdf.subsection_title("Optuna Hyperparameter Tuning")
        for name, result in tuning_results.items():
            if "best_params" in result and result["best_params"]:
                pdf.body_text(f"{name}: Best CV R2 = {result.get('best_cv_score', 'N/A'):.4f}")
                params_str = ", ".join(f"{k}={v}" for k, v in list(result["best_params"].items())[:5])
                pdf.body_text(f"  Parameters: {params_str}")

    # ──────────────────────────────────────
    # 4. Optimization Results
    # ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("4. Optimization Results")

    bo_recs = report_data.get("bo_recommendations")
    if bo_recs is not None and len(bo_recs) > 0:
        pdf.subsection_title("Recommended Optimal Conditions")
        # Show top recommendation
        top = bo_recs.iloc[0]
        for col in bo_recs.columns:
            if col not in ["Rank"]:
                val = top[col]
                if isinstance(val, float):
                    pdf.body_text(f"  {col}: {val:.4f}")
                else:
                    pdf.body_text(f"  {col}: {val}")

        pdf.ln(5)
        pdf.subsection_title("All Recommendations")
        pdf.add_table(bo_recs)

    # Convergence
    bo_history = report_data.get("bo_history", [])
    if bo_history:
        pdf.subsection_title("Convergence History")
        for h in bo_history:
            pdf.body_text(f"  Iteration {h['iteration']}: Best = {h['best_value']:.4f} ({h.get('n_experiments', '?')} experiments)")

    # ──────────────────────────────────────
    # 5. Conclusion
    # ──────────────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Summary & Conclusions")
    pdf.body_text(
        "This report was generated by OptiLab, an AI-assisted experimental optimization platform. "
        "The platform trained a Gaussian Process surrogate model on the provided experimental data, "
        "automatically tuned hyperparameters using Optuna, and used Bayesian Optimization to recommend "
        "optimal experimental conditions."
    )

    if report_data.get("best_model_name"):
        pdf.body_text(f"The surrogate model used was: {report_data['best_model_name']}.")

    if report_data.get("iteration_count", 0) > 0:
        pdf.body_text(
            f"A total of {report_data['iteration_count']} active learning iterations were performed, "
            f"progressively improving the model's predictions through new experimental data."
        )

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "— End of Report —", align="C")

    # Output
    return pdf.output()


def save_report(report_bytes: bytes, filepath: str):
    """Save report bytes to a file."""
    with open(filepath, "wb") as f:
        f.write(report_bytes)
