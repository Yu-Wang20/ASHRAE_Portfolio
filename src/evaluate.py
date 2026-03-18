"""
ASHRAE Great Energy Predictor III — Model Evaluation & Comparison

This module provides :class:`ModelEvaluator`, a single class that handles
every post-training evaluation task:

    - Per-model metrics (RMSLE, RMSE, MAE, R²)
    - Cross-model comparison tables
    - Classical residual diagnostics (4-panel figure)
    - Visual model comparison (bar chart)
    - Competition submission generation

Design note
-----------
All targets in this project use the ``log1p`` transform applied in
:meth:`~src.feature_engineering.FeatureEngineer.get_feature_matrix`.
Metrics that operate in the original kWh scale (RMSE, MAE) first apply
``np.expm1`` to reverse the transform.  RMSLE is computed directly on
the log-space predictions, which is algebraically equivalent to
``sqrt(mean((log(1+y_true) - log(1+y_pred))²))``.

Usage
-----
    from src.evaluate import ModelEvaluator

    evaluator = ModelEvaluator(models_dir, output_dir)
    metrics   = evaluator.compute_metrics(y_true, y_pred, "LightGBM")
    evaluator.plot_residuals(y_true, y_pred, "LightGBM")
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Consistent style for all figures produced by this module.
plt.style.use("seaborn-v0_8-whitegrid")


class ModelEvaluator:
    """Evaluate, compare, and visualise model performance.

    Parameters
    ----------
    models_dir : str or Path
        Directory containing saved model artefacts
        (e.g. ``outputs/models``).
    output_dir : str or Path
        Root output directory (e.g. ``outputs``).  Sub-directories
        ``figures/`` and ``predictions/`` are used for saved files.

    Attributes
    ----------
    models_dir : Path
    figures_dir : Path
    predictions_dir : Path
    """

    def __init__(self, models_dir: str | Path, output_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.figures_dir = Path(output_dir) / "figures"
        self.predictions_dir = Path(output_dir) / "predictions"

        # Ensure output directories exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ModelEvaluator initialised")
        logger.info("  models_dir     : %s", self.models_dir)
        logger.info("  figures_dir    : %s", self.figures_dir)
        logger.info("  predictions_dir: %s", self.predictions_dir)

    # ================================================================== #
    # 1. Metrics
    # ================================================================== #
    def compute_metrics(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str,
    ) -> Dict[str, Any]:
        """Compute regression metrics for a single model.

        Both ``y_true`` and ``y_pred`` are expected in **log-space**
        (i.e. the ``log1p(meter_reading)`` values produced by
        :meth:`~src.feature_engineering.FeatureEngineer.get_feature_matrix`).

        Parameters
        ----------
        y_true : array-like
            Ground-truth target in log-space.
        y_pred : array-like
            Predicted target in log-space.
        model_name : str
            Human-readable model identifier stored in the result dict.

        Returns
        -------
        dict
            Keys: ``model_name``, ``RMSLE``, ``RMSE``, ``MAE``, ``R2``.
            RMSE and MAE are in the **original kWh scale** (after
            ``expm1``).  RMSLE and R² are computed in log-space.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # --- Log-space metrics ---
        # RMSLE: primary competition metric.  Since y values are already
        # log1p-transformed, RMSLE = RMSE in log-space.
        rmsle = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # --- Original-scale metrics (kWh) ---
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae = mean_absolute_error(y_true_orig, y_pred_orig)

        result = {
            "model_name": model_name,
            "RMSLE": round(float(rmsle), 6),
            "RMSE": round(float(rmse), 4),
            "MAE": round(float(mae), 4),
            "R2": round(float(r2), 6),
        }

        logger.info(
            "[%s] RMSLE=%.6f  RMSE=%.2f  MAE=%.2f  R²=%.6f",
            model_name,
            rmsle,
            rmse,
            mae,
            r2,
        )
        return result

    # ================================================================== #
    # 2. Model comparison table
    # ================================================================== #
    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build a sorted comparison DataFrame from multiple metric dicts.

        Parameters
        ----------
        results_list : list of dict
            Each dict is the output of :meth:`compute_metrics`.

        Returns
        -------
        pd.DataFrame
            One row per model, sorted by RMSLE ascending (best first).
        """
        df = pd.DataFrame(results_list)
        df = df.sort_values("RMSLE", ascending=True).reset_index(drop=True)
        df.index.name = "rank"
        df.index += 1  # 1-based ranking

        # Pretty-print to console
        logger.info("Model comparison (sorted by RMSLE):")
        header = (
            f"  {'Rank':<5} {'Model':<25} {'RMSLE':>10} "
            f"{'RMSE':>12} {'MAE':>12} {'R²':>10}"
        )
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for rank, row in df.iterrows():
            logger.info(
                "  %-5d %-25s %10.6f %12.2f %12.2f %10.6f",
                rank,
                row["model_name"],
                row["RMSLE"],
                row["RMSE"],
                row["MAE"],
                row["R2"],
            )

        return df

    # ================================================================== #
    # 3. Residual diagnostics (2×2 figure)
    # ================================================================== #
    def plot_residuals(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str,
        save: bool = True,
    ) -> plt.Figure:
        """Create a 2×2 classical residual-diagnostic figure.

        Layout::

            [0,0] Residuals vs Fitted values
            [0,1] Q-Q plot of residuals (scipy.stats.probplot)
            [1,0] Scale-Location plot (√|standardised residuals| vs fitted)
            [1,1] Histogram of residuals with normal-curve overlay

        These four plots are inherited from classical linear-regression
        diagnostics.  In the portfolio notebooks we deliberately produce
        them for the OLS baseline to **motivate the transition to
        non-linear models**: visible heteroscedasticity and heavy tails
        in the Q-Q plot provide visual evidence that OLS assumptions are
        violated, justifying the move to LightGBM / XGBoost.

        Parameters
        ----------
        y_true, y_pred : array-like
            Log-space ground truth and predictions.
        model_name : str
            Used in the figure title and output filename.
        save : bool
            If *True*, save to ``outputs/figures/{model_name}_diagnostics.png``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        residuals = y_true - y_pred
        fitted = y_pred

        # Standardised residuals for scale-location plot
        std_resid = residuals / (residuals.std() + 1e-12)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{model_name} — Residual Diagnostics", fontsize=15, y=0.98)

        # --- [0,0] Residuals vs Fitted ---
        ax = axes[0, 0]
        ax.scatter(fitted, residuals, alpha=0.15, s=4, edgecolors="none")
        ax.axhline(0, color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Fitted values (log-space)")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")

        # --- [0,1] Q-Q plot ---
        ax = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")
        ax.get_lines()[0].set(markersize=2, alpha=0.3)

        # --- [1,0] Scale-Location ---
        ax = axes[1, 0]
        ax.scatter(
            fitted,
            np.sqrt(np.abs(std_resid)),
            alpha=0.15,
            s=4,
            edgecolors="none",
        )
        ax.set_xlabel("Fitted values (log-space)")
        ax.set_ylabel(r"$\sqrt{|\mathrm{standardised\ residuals}|}$")
        ax.set_title("Scale-Location")

        # --- [1,1] Histogram + normal curve ---
        ax = axes[1, 1]
        ax.hist(residuals, bins=100, density=True, alpha=0.7, edgecolor="white")
        # Overlay normal PDF
        x_range = np.linspace(residuals.min(), residuals.max(), 300)
        ax.plot(
            x_range,
            stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
            color="red",
            linewidth=1.5,
            label="Normal fit",
        )
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Residual Distribution")
        ax.legend()

        fig.tight_layout()

        if save:
            path = self.figures_dir / f"{model_name}_diagnostics.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved residual diagnostics → %s", path)

        return fig

    # ================================================================== #
    # 4. Model comparison bar chart
    # ================================================================== #
    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """Horizontal bar chart of RMSLE by model (best at top).

        Expected model names (in the order they appear in the portfolio
        narrative):
        ``OLS_baseline``, ``OLS_log_transformed``, ``WLS``,
        ``LightGBM``, ``XGBoost``.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame returned by :meth:`compare_models`.
        save : bool
            If *True*, save to ``outputs/figures/model_comparison.png``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # Sort worst → best so the best model appears at the top of the
        # horizontal bar chart (matplotlib draws bars bottom-up).
        plot_df = results_df.sort_values("RMSLE", ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.8)))

        colours = []
        for name in plot_df["model_name"]:
            if "LightGBM" in name or "XGBoost" in name:
                colours.append("#2ecc71")  # green for tree models
            elif "WLS" in name:
                colours.append("#3498db")  # blue for improved linear
            else:
                colours.append("#e74c3c")  # red for baselines

        bars = ax.barh(
            plot_df["model_name"],
            plot_df["RMSLE"],
            color=colours,
            edgecolor="white",
            height=0.5,
        )

        # Annotate each bar with its RMSLE value
        for bar, rmsle_val in zip(bars, plot_df["RMSLE"]):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{rmsle_val:.4f}",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("RMSLE (lower is better)")
        ax.set_title("Model Comparison — RMSLE")
        ax.invert_yaxis()  # best (lowest) at top after the sort flip
        fig.tight_layout()

        if save:
            path = self.figures_dir / "model_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved model comparison chart → %s", path)

        return fig

    # ================================================================== #
    # 5. Competition submission
    # ================================================================== #
    def generate_submission(
        self,
        model: Any,
        X_test: pd.DataFrame,
        test_df: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """Generate a Kaggle-format submission CSV.

        Parameters
        ----------
        model
            Any fitted estimator exposing a ``.predict(X)`` method that
            returns log-space predictions (consistent with the
            ``log1p(meter_reading)`` target).
        X_test : pd.DataFrame
            Feature matrix for the test set (same schema as training
            features).
        test_df : pd.DataFrame
            Original test DataFrame containing the ``row_id`` column
            required by the competition.
        save : bool
            If *True*, save to
            ``outputs/predictions/submission_{timestamp}.csv``.

        Returns
        -------
        pd.DataFrame
            Submission DataFrame with columns ``[row_id, meter_reading]``.

        Notes
        -----
        - ``np.expm1`` reverses the ``log1p`` target transform.
        - Predictions are clipped at 0 — negative energy consumption is
          physically impossible.
        - ``meter_reading`` is rounded to 4 decimal places, matching the
          competition sample submission format.
        """
        logger.info("Generating submission (%s test rows) …", f"{len(X_test):,}")

        # Predict in log-space, then invert
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        # Clip negative values (physically impossible)
        y_pred = np.clip(y_pred, 0, None)

        submission = pd.DataFrame(
            {
                "row_id": test_df["row_id"].values,
                "meter_reading": np.round(y_pred, 4),
            }
        )

        logger.info(
            "Submission stats: min=%.4f, median=%.4f, max=%.4f",
            submission["meter_reading"].min(),
            submission["meter_reading"].median(),
            submission["meter_reading"].max(),
        )

        if save:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.predictions_dir / f"submission_{ts}.csv"
            submission.to_csv(path, index=False)
            logger.info("Saved submission → %s", path)

        return submission
