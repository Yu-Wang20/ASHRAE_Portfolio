"""
ASHRAE Portfolio — Pre-compute Streamlit App Data

This script generates all artefacts the Streamlit dashboard needs so that
the app itself performs **zero heavy computation** at runtime.

Outputs
-------
    outputs/app_building_summary.parquet  — per-building aggregate stats
    outputs/app_timeseries.parquet        — slim time-series for plotting
    outputs/app_shap_values.parquet       — SHAP values for top-20 buildings

Usage
-----
    cd /Users/wangyu/Desktop/ASHRAE_Portfolio
    source venv/bin/activate
    python src/prepare_app_data.py
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FeatureEngineer  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Cast float64 → float32, int64 → int32."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    return df


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Load validation set with features
# ═══════════════════════════════════════════════════════════════════════════

def prepare_validation_set() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Load clean data, engineer features, return validation-period rows.

    Returns
    -------
    val_df : pd.DataFrame
        Engineered DataFrame filtered to validation period (timestamp >= 2016-10-01),
        with NaN lags dropped.  Retains all columns for downstream slicing.
    X_val : pd.DataFrame
        Feature matrix matching the trained model's expected columns.
    y_val : np.ndarray
        Target in log-space: log1p(meter_reading).
    feature_names : list[str]
        Feature column names (from model metadata).
    """
    logger.info("=" * 65)
    logger.info("STEP 1: Load & engineer features for validation set")
    logger.info("=" * 65)

    # Load model metadata for feature list
    with open(MODELS_DIR / "lgbm_v1_metadata.json") as f:
        metadata = json.load(f)
    feature_names: list[str] = metadata["feature_names"]

    # Load clean data
    df = pd.read_parquet(OUTPUT_DIR / "clean_data.parquet")
    logger.info("Loaded clean_data: %s rows", f"{len(df):,}")
    df = _downcast(df)

    # Feature engineering
    fe = FeatureEngineer(df)
    del df; gc.collect()

    fe.add_temporal_features()
    fe.add_building_features()
    fe.add_interaction_features()
    fe.add_lag_features(lags=[24, 168])

    engineered = fe.df
    del fe; gc.collect()

    # Filter to validation period
    val_mask = engineered["timestamp"] >= pd.Timestamp("2016-10-01")
    val_df = engineered[val_mask].copy()
    del engineered; gc.collect()
    logger.info("Validation period (>= 2016-10-01): %s rows", f"{len(val_df):,}")

    # Drop NaN lags
    before = len(val_df)
    val_df = val_df.dropna(subset=["lag_24", "lag_168"]).reset_index(drop=True)
    logger.info("After dropping NaN lags: %s rows (removed %s)",
                f"{len(val_df):,}", f"{before - len(val_df):,}")

    # Build feature matrix
    X_val = val_df[feature_names].copy()
    for col in X_val.columns:
        if X_val[col].dtype == bool:
            X_val[col] = X_val[col].astype(np.int8)

    y_val = np.log1p(val_df["meter_reading"].values).astype(np.float32)

    return val_df, X_val, y_val, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Generate predictions
# ═══════════════════════════════════════════════════════════════════════════

def add_predictions(val_df: pd.DataFrame, X_val: pd.DataFrame,
                    y_val: np.ndarray) -> pd.DataFrame:
    """Add actual_reading and predicted_reading columns."""
    logger.info("=" * 65)
    logger.info("STEP 2: Generate predictions")
    logger.info("=" * 65)

    model = joblib.load(MODELS_DIR / "lgbm_v1.pkl")
    logger.info("Loaded model from %s", MODELS_DIR / "lgbm_v1.pkl")

    y_pred_log = model.predict(X_val).astype(np.float32)

    val_df["predicted_reading"] = np.clip(np.expm1(y_pred_log), 0, None).astype(np.float32)
    val_df["actual_reading"] = np.expm1(y_val).astype(np.float32)

    # Quick validation metric
    rmsle = float(np.sqrt(np.mean((y_val - y_pred_log) ** 2)))
    logger.info("Validation RMSLE: %.4f", rmsle)
    logger.info("Prediction range: [%.1f, %.1f] kWh",
                val_df["predicted_reading"].min(), val_df["predicted_reading"].max())

    return val_df, model


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Per-building summary
# ═══════════════════════════════════════════════════════════════════════════

def compute_building_summary(val_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (building_id, meter) statistics."""
    logger.info("=" * 65)
    logger.info("STEP 3: Per-building summary")
    logger.info("=" * 65)

    def _building_rmsle(group: pd.DataFrame) -> float:
        y_true_log = np.log1p(group["actual_reading"].values)
        y_pred_log = np.log1p(group["predicted_reading"].values)
        return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))

    # First-value columns
    first_cols = {}
    for col in ["primary_use", "site_id", "log_square_feet", "building_age"]:
        if col in val_df.columns:
            first_cols[col] = (col, "first")

    agg = val_df.groupby(["building_id", "meter"]).agg(
        primary_use=("primary_use", "first"),
        site_id=("site_id", "first"),
        log_square_feet=("log_square_feet", "first"),
        building_age=("building_age", "first"),
        n_rows=("actual_reading", "count"),
        mean_actual=("actual_reading", "mean"),
    ).reset_index()

    # Per-building RMSLE
    rmsle_series = val_df.groupby(["building_id", "meter"]).apply(
        _building_rmsle, include_groups=False
    ).reset_index(name="rmsle_building")

    summary = agg.merge(rmsle_series, on=["building_id", "meter"])
    summary = _downcast(summary)

    path = OUTPUT_DIR / "app_building_summary.parquet"
    summary.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved building summary: %s rows → %s (%.1f MB)",
                f"{len(summary):,}", path, _file_size_mb(path))

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Slim time-series data
# ═══════════════════════════════════════════════════════════════════════════

def save_timeseries(val_df: pd.DataFrame) -> None:
    """Save only the columns the Streamlit app needs for time-series plots."""
    logger.info("=" * 65)
    logger.info("STEP 4: Slim time-series data")
    logger.info("=" * 65)

    keep_cols = [
        "building_id", "meter", "timestamp",
        "actual_reading", "predicted_reading",
        "air_temperature", "is_weekend",
        "hour_of_day", "day_of_week",
    ]
    slim = val_df[keep_cols].copy()
    slim = _downcast(slim)

    # Convert bool to int8 for parquet compatibility
    if slim["is_weekend"].dtype == bool:
        slim["is_weekend"] = slim["is_weekend"].astype(np.int8)

    # Budget check: if projected size > 30 MB, keep only top-N buildings
    # by row count to stay within budget.  A test write gives the true size.
    path = OUTPUT_DIR / "app_timeseries.parquet"
    slim.to_parquet(path, index=False, engine="pyarrow")
    size_mb = _file_size_mb(path)

    if size_mb > 30:
        target_frac = 28.0 / size_mb  # aim for ~28 MB with margin
        logger.info("Trimming time-series: %.1f MB > 30 MB target (keeping %.0f%% of buildings)",
                    size_mb, target_frac * 100)
        # Keep only buildings with the most rows (most useful for dashboards)
        bldg_counts = slim.groupby(["building_id", "meter"]).size().reset_index(name="cnt")
        bldg_counts = bldg_counts.sort_values("cnt", ascending=False)
        n_keep = max(1, int(len(bldg_counts) * target_frac))
        keep_keys = bldg_counts.head(n_keep)[["building_id", "meter"]]
        slim = slim.merge(keep_keys, on=["building_id", "meter"], how="inner")
        slim.to_parquet(path, index=False, engine="pyarrow")
        size_mb = _file_size_mb(path)

    logger.info("Saved time-series: %s rows → %s (%.1f MB)",
                f"{len(slim):,}", path, size_mb)

    if size_mb > 30:
        logger.warning("⚠ File still exceeds 30 MB target (%.1f MB)", size_mb)
    else:
        logger.info("✓ Under 30 MB target")


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Pre-compute SHAP values for top-20 buildings
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap_top20(val_df: pd.DataFrame, model, feature_names: list[str],
                       summary: pd.DataFrame) -> None:
    """Compute SHAP values for 200 rows from each of the top-20 buildings."""
    logger.info("=" * 65)
    logger.info("STEP 5: SHAP values for top-20 buildings")
    logger.info("=" * 65)

    # Top 20 buildings by row count
    top20 = summary.nlargest(20, "n_rows")[["building_id", "meter"]].copy()
    logger.info("Top 20 buildings selected (by n_rows)")

    explainer = shap.TreeExplainer(model)

    shap_records = []
    for _, row in top20.iterrows():
        bid, met = int(row["building_id"]), int(row["meter"])
        mask = (val_df["building_id"] == bid) & (val_df["meter"] == met)
        subset = val_df[mask]

        # Sample up to 200 rows
        n_sample = min(200, len(subset))
        if n_sample == 0:
            continue
        sampled = subset.sample(n=n_sample, random_state=42)

        # Build X
        X = sampled[feature_names].copy()
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(np.int8)

        # Compute SHAP (use numpy to avoid LightGBM/DataFrame compat issues)
        sv = explainer.shap_values(X.values)

        # Build records DataFrame
        # Prefix SHAP columns to avoid collision with ID columns like 'meter'
        shap_col_names = [f"shap_{c}" for c in feature_names]
        rec = pd.DataFrame(sv, columns=shap_col_names).astype(np.float32)
        rec.insert(0, "building_id", sampled["building_id"].values)
        rec.insert(1, "meter", sampled["meter"].values)
        rec.insert(2, "timestamp", sampled["timestamp"].values)

        shap_records.append(rec)
        logger.info("  building_id=%d, meter=%d → %d SHAP rows", bid, met, n_sample)

    shap_df = pd.concat(shap_records, ignore_index=True)
    shap_df = _downcast(shap_df)

    path = OUTPUT_DIR / "app_shap_values.parquet"
    shap_df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved SHAP values: %s rows → %s (%.1f MB)",
                f"{len(shap_df):,}", path, _file_size_mb(path))


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Final verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_outputs() -> None:
    """List sizes of all app data files and check the 50 MB budget."""
    logger.info("=" * 65)
    logger.info("STEP 6: Verify outputs")
    logger.info("=" * 65)

    files = [
        OUTPUT_DIR / "app_building_summary.parquet",
        OUTPUT_DIR / "app_timeseries.parquet",
        OUTPUT_DIR / "app_shap_values.parquet",
    ]

    total = 0.0
    for f in files:
        if f.exists():
            sz = _file_size_mb(f)
            total += sz
            logger.info("  ✓ %-40s %6.1f MB", f.name, sz)
        else:
            logger.error("  ✗ MISSING: %s", f)

    logger.info("  %-40s %6.1f MB", "TOTAL", total)

    if total > 50:
        logger.warning("⚠ Combined size %.1f MB exceeds 50 MB budget!", total)
    else:
        logger.info("✓ All files within 50 MB budget (%.1f MB)", total)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full app-data preparation pipeline."""
    logger.info("╔═══════════════════════════════════════════════════════════╗")
    logger.info("║   ASHRAE Portfolio — Preparing Streamlit App Data        ║")
    logger.info("╚═══════════════════════════════════════════════════════════╝")

    # Step 1
    val_df, X_val, y_val, feature_names = prepare_validation_set()

    # Step 2
    val_df, model = add_predictions(val_df, X_val, y_val)
    del X_val, y_val; gc.collect()

    # Step 3
    summary = compute_building_summary(val_df)

    # Step 4
    save_timeseries(val_df)

    # Step 5
    compute_shap_top20(val_df, model, feature_names, summary)
    del val_df, model; gc.collect()

    # Step 6
    verify_outputs()

    logger.info("Done.")


if __name__ == "__main__":
    main()
