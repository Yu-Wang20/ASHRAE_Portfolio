"""
ASHRAE Great Energy Predictor III — Memory-Optimised LightGBM Training

Designed for 8 GB RAM machines.  Key memory strategies:

    1. **float32 everywhere** — halves memory vs float64.
    2. **Selective lag features** — only lag_24 and lag_168 (skip lag_1,
       lag_2 and all rolling features to save ~1.5 GB).
    3. **Immediate dtype down-casting** after every allocation.
    4. **50 % data fallback** — if a MemoryError is caught during loading,
       training continues on a random 50 % sample with a logged warning.

Usage
-----
    cd /Users/wangyu/Desktop/ASHRAE_Portfolio
    source venv/bin/activate
    python src/run_training.py
"""

from __future__ import annotations

import datetime
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FeatureEngineer  # noqa: E402

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


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _mem_report(tag: str) -> None:
    """Log current RAM usage with a descriptive tag."""
    vm = psutil.virtual_memory()
    logger.info(
        "[MEM] %-30s  RAM used: %5.1f%%  (%.1f / %.1f GB)",
        tag,
        vm.percent,
        vm.used / 1e9,
        vm.total / 1e9,
    )


def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Down-cast float64 → float32 and int64 → int32 in place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to down-cast.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with reduced memory footprint.
    """
    mem_before = df.memory_usage(deep=True).sum() / 1e6

    float64_cols = df.select_dtypes(include=["float64"]).columns
    df[float64_cols] = df[float64_cols].astype(np.float32)

    int64_cols = df.select_dtypes(include=["int64"]).columns
    df[int64_cols] = df[int64_cols].astype(np.int32)

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    logger.info(
        "  Dtype downcast: %.1f MB → %.1f MB (saved %.1f MB)",
        mem_before, mem_after, mem_before - mem_after,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Load and cast dtypes
# ═══════════════════════════════════════════════════════════════════════════

def load_data(parquet_path: Path) -> pd.DataFrame:
    """Load clean parquet with immediate dtype down-casting.

    If a ``MemoryError`` is raised, the function retries on a 50 %
    random sample and logs a prominent warning.

    Parameters
    ----------
    parquet_path : Path
        Path to ``clean_data.parquet``.

    Returns
    -------
    pd.DataFrame
        Down-cast DataFrame ready for feature engineering.
    """
    logger.info("=" * 65)
    logger.info("STEP 1: Load data")
    logger.info("=" * 65)
    _mem_report("before load")

    memory_fallback = False
    try:
        df = pd.read_parquet(parquet_path)
    except MemoryError:
        logger.warning("⚠ MEMORY FALLBACK: loading with row-group limit")
        df = pd.read_parquet(parquet_path)
        df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
        memory_fallback = True
        logger.warning("⚠ MEMORY FALLBACK: training on 50%% sample (%s rows)", f"{len(df):,}")

    logger.info("Loaded %s rows × %d cols", f"{len(df):,}", df.shape[1])
    _mem_report("after load (raw)")

    df = _downcast_df(df)
    _mem_report("after downcast")

    # If RAM is already >85 % used, proactively sample
    if not memory_fallback and psutil.virtual_memory().percent > 85:
        logger.warning("⚠ MEMORY FALLBACK: RAM > 85%% — sampling 50%% of data")
        df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
        gc.collect()
        memory_fallback = True
        logger.warning("⚠ Training on %s rows", f"{len(df):,}")
        _mem_report("after 50%% sample")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Feature engineering (memory-constrained)
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering with a reduced lag set.

    Memory strategy:
        - Only ``lag_24`` and ``lag_168`` (skip lag_1, lag_2).
        - No rolling features (~1.5 GB saved).
        - Immediate float32 re-cast after feature creation.

    Parameters
    ----------
    df : pd.DataFrame
        Down-cast DataFrame from :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame (still includes raw columns
        that will be dropped at the split step).
    """
    logger.info("=" * 65)
    logger.info("STEP 2: Feature engineering (memory-optimised)")
    logger.info("=" * 65)
    _mem_report("before FE")

    fe = FeatureEngineer(df)
    # Release our reference — FeatureEngineer already copied
    del df
    gc.collect()

    fe.add_temporal_features()
    fe.add_building_features()
    fe.add_interaction_features()

    # Selective lags — only the two with highest predictive value
    fe.add_lag_features(lags=[24, 168])
    logger.info("Skipped lag_1, lag_2, and all rolling features (memory saving)")

    # Re-downcast any new float64 columns created by FE
    fe.df = _downcast_df(fe.df)
    gc.collect()
    _mem_report("after FE + downcast")

    return fe.df


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Time-based train / validation split
# ═══════════════════════════════════════════════════════════════════════════

def time_split(
    df: pd.DataFrame,
    cutoff: str = "2016-10-01",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split into train / val by timestamp and build X, y arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    cutoff : str
        ISO date string.  Rows before this → train; on or after → val.

    Returns
    -------
    X_train, y_train, X_val, y_val
        Feature matrices and log1p targets.
    """
    logger.info("=" * 65)
    logger.info("STEP 3: Time-based split (cutoff=%s)", cutoff)
    logger.info("=" * 65)
    _mem_report("before split")

    cutoff_ts = pd.Timestamp(cutoff)

    # Drop rows where lags are NaN (first 168 h per building group)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    rows_before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    logger.info(
        "Dropped %s rows with NaN lags (first 168 h per group)",
        f"{rows_before - len(df):,}",
    )

    # Split
    train_mask = df["timestamp"] < cutoff_ts
    val_mask = ~train_mask

    # Target: log1p(meter_reading)
    y_train = np.log1p(df.loc[train_mask, "meter_reading"]).astype(np.float32)
    y_val = np.log1p(df.loc[val_mask, "meter_reading"]).astype(np.float32)

    # Drop non-feature columns
    drop_cols = [
        "timestamp", "meter_reading", "building_id", "site_id",
        "year_built", "square_feet", "floor_count",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X_train = df.loc[train_mask].drop(columns=drop_cols)
    X_val = df.loc[val_mask].drop(columns=drop_cols)

    # Free the full DataFrame
    del df
    gc.collect()

    # Ensure bool → int for LightGBM
    for col in X_train.columns:
        if X_train[col].dtype == bool:
            X_train[col] = X_train[col].astype(np.int8)
            X_val[col] = X_val[col].astype(np.int8)

    logger.info("Train: %s rows × %d features", f"{len(X_train):,}", X_train.shape[1])
    logger.info("Val  : %s rows × %d features", f"{len(X_val):,}", X_val.shape[1])
    logger.info("Features: %s", list(X_train.columns))
    _mem_report("after split")

    return X_train, y_train, X_val, y_val


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Train LightGBM
# ═══════════════════════════════════════════════════════════════════════════

def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMRegressor:
    """Train a LightGBM model with early stopping.

    Parameters
    ----------
    X_train, y_train : training features and log1p target.
    X_val, y_val : validation features and log1p target.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model (best iteration selected by early stopping).
    """
    logger.info("=" * 65)
    logger.info("STEP 4: Train LightGBM")
    logger.info("=" * 65)
    _mem_report("before training")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        random_state=42,
        **params,
    )

    logger.info("LightGBM params: %s", params)
    logger.info("n_estimators=1000, early_stopping_rounds=50")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    logger.info("Best iteration: %d", model.best_iteration_)
    _mem_report("after training")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Evaluate
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    model: lgb.LGBMRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Compute validation metrics and print feature importances.

    Parameters
    ----------
    model : fitted LGBMRegressor.
    X_val, y_val : validation data (log-space).

    Returns
    -------
    dict
        Metric dictionary with keys ``RMSLE``, ``R2``, ``RMSE``, ``MAE``.
    """
    logger.info("=" * 65)
    logger.info("STEP 5: Evaluate")
    logger.info("=" * 65)

    y_pred_log = model.predict(X_val)

    # --- RMSLE (= RMSE in log-space since target is already log1p) ---
    rmsle = np.sqrt(mean_squared_error(y_val, y_pred_log))

    # --- R² in log-space ---
    r2 = r2_score(y_val, y_pred_log)

    # --- Original-scale metrics ---
    y_true_orig = np.expm1(y_val.values)
    y_pred_orig = np.clip(np.expm1(y_pred_log), 0, None)

    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig = np.mean(np.abs(y_true_orig - y_pred_orig))

    logger.info("─" * 50)
    logger.info("LightGBM Validation RMSLE : %.4f", rmsle)
    logger.info("LightGBM Validation R²    : %.4f", r2)
    logger.info("LightGBM Validation RMSE  : %.2f kWh", rmse_orig)
    logger.info("LightGBM Validation MAE   : %.2f kWh", mae_orig)
    logger.info("─" * 50)

    # --- Feature importances (top 10) ---
    importance = pd.Series(
        model.feature_importances_, index=X_val.columns
    ).sort_values(ascending=False)

    logger.info("Top 10 Feature Importances (split-based):")
    for rank, (feat, imp) in enumerate(importance.head(10).items(), 1):
        logger.info("  %2d. %-25s %d", rank, feat, imp)

    return {
        "RMSLE": round(float(rmsle), 6),
        "R2": round(float(r2), 6),
        "RMSE_kWh": round(float(rmse_orig), 4),
        "MAE_kWh": round(float(mae_orig), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Save model + metadata
# ═══════════════════════════════════════════════════════════════════════════

def save_artefacts(
    model: lgb.LGBMRegressor,
    metrics: dict,
    feature_names: list,
    n_train: int,
    n_val: int,
) -> None:
    """Persist model and training metadata.

    Parameters
    ----------
    model : fitted LGBMRegressor.
    metrics : dict from :func:`evaluate`.
    feature_names : list of feature column names.
    n_train, n_val : row counts for metadata.
    """
    logger.info("=" * 65)
    logger.info("STEP 6: Save artefacts")
    logger.info("=" * 65)

    models_dir = PROJECT_ROOT / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Model pickle ---
    model_path = models_dir / "lgbm_v1.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved model → %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)

    # --- Metadata JSON ---
    metadata = {
        "model_name": "LightGBM_v1",
        "training_date": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_train_rows": n_train,
        "n_val_rows": n_val,
        "n_estimators_best": model.best_iteration_,
        "feature_names": feature_names,
        "metrics": metrics,
        "params": model.get_params(),
    }

    meta_path = models_dir / "lgbm_v1_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata → %s", meta_path)

    _mem_report("final")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute the full memory-optimised LightGBM training pipeline."""
    start = datetime.datetime.now()
    logger.info("╔═══════════════════════════════════════════════════════════╗")
    logger.info("║   ASHRAE — Memory-Optimised LightGBM Training Pipeline  ║")
    logger.info("╚═══════════════════════════════════════════════════════════╝")
    _mem_report("startup")

    # Step 1
    parquet_path = PROJECT_ROOT / "outputs" / "clean_data.parquet"
    df = load_data(parquet_path)

    # Step 2
    df = engineer_features(df)

    # Step 3
    X_train, y_train, X_val, y_val = time_split(df)
    del df
    gc.collect()
    _mem_report("after df deleted")

    # Step 4
    model = train_lgbm(X_train, y_train, X_val, y_val)

    # Step 5
    metrics = evaluate(model, X_val, y_val)

    # Step 6
    save_artefacts(
        model=model,
        metrics=metrics,
        feature_names=list(X_train.columns),
        n_train=len(X_train),
        n_val=len(X_val),
    )

    elapsed = datetime.datetime.now() - start
    logger.info("Pipeline completed in %s", str(elapsed).split(".")[0])


if __name__ == "__main__":
    main()
