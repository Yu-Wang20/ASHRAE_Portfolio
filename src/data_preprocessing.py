"""
ASHRAE Great Energy Predictor III — Data Loading & Preprocessing Pipeline

This module provides two core classes:

    ASHRAEDataLoader   – Reads raw CSVs and merges them into a single
                         analysis-ready DataFrame.
    ASHRAEPreprocessor – Applies domain-specific cleaning, imputation,
                         encoding, and sampling to the merged DataFrame.

Design choices are documented inline and draw heavily on the competition
discussion threads and top-scoring Kaggle solutions.

Usage
-----
    python -m src.data_preprocessing          # from project root
    python src/data_preprocessing.py          # direct invocation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _configure_root_logger() -> None:
    """Configure the root logger with a consistent format for CLI runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

class ASHRAEDataLoader:
    """Load and merge the three core ASHRAE competition CSVs.

    Parameters
    ----------
    config_path : str or Path
        Path to the project ``config.yaml`` file.

    Attributes
    ----------
    config : dict
        Parsed YAML configuration.
    data_dir : Path
        Resolved directory containing the raw CSV files.
    """

    def __init__(self, config_path: str | Path) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        # Resolve data_dir relative to the config file's parent (project root)
        self.data_dir = config_path.parent.parent / self.config["data_dir"]
        logger.info("Config loaded from %s", config_path)
        logger.info("Data directory: %s", self.data_dir)

    # ------------------------------------------------------------------ #
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read train.csv, building_metadata.csv, and weather_train.csv.

        Returns
        -------
        train_df : pd.DataFrame
            Hourly meter readings with columns
            [building_id, meter, timestamp, meter_reading].
        meta_df : pd.DataFrame
            Building metadata with columns
            [site_id, building_id, primary_use, square_feet, …].
        weather_df : pd.DataFrame
            Weather observations per site with columns
            [site_id, timestamp, air_temperature, …].
        """
        logger.info("Loading raw CSV files …")

        train_df = pd.read_csv(
            self.data_dir / "train.csv",
            parse_dates=["timestamp"],
            dtype={"building_id": np.int16, "meter": np.int8},
        )
        logger.info("  train.csv        : %s rows", f"{len(train_df):,}")

        meta_df = pd.read_csv(
            self.data_dir / "building_metadata.csv",
            dtype={"site_id": np.int8, "building_id": np.int16},
        )
        logger.info("  building_metadata: %s rows", f"{len(meta_df):,}")

        weather_df = pd.read_csv(
            self.data_dir / "weather_train.csv",
            parse_dates=["timestamp"],
            dtype={"site_id": np.int8},
        )
        logger.info("  weather_train    : %s rows", f"{len(weather_df):,}")

        return train_df, meta_df, weather_df

    # ------------------------------------------------------------------ #
    def merge_data(
        self,
        train_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge the three source tables into a single analysis DataFrame.

        Merge order:
            1. ``train`` ← ``building_metadata`` on **building_id**
            2. result   ← ``weather_train``      on **[site_id, timestamp]**

        Parameters
        ----------
        train_df, meta_df, weather_df : pd.DataFrame
            DataFrames returned by :meth:`load_raw_data`.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all columns from the three sources.
        """
        logger.info("Merging datasets …")

        merged = train_df.merge(meta_df, on="building_id", how="left")
        logger.info("  After train ⟕ metadata  : %s rows", f"{len(merged):,}")

        merged = merged.merge(weather_df, on=["site_id", "timestamp"], how="left")
        logger.info("  After ⟕ weather         : %s rows", f"{len(merged):,}")

        return merged


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class ASHRAEPreprocessor:
    """Domain-aware cleaning and feature engineering for the ASHRAE dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame produced by :meth:`ASHRAEDataLoader.merge_data`.

    Attributes
    ----------
    df : pd.DataFrame
        The working copy that is mutated in-place by each pipeline step.
    label_encoder : LabelEncoder or None
        Fitted encoder for ``primary_use``; available after
        :meth:`encode_categoricals`.
    primary_use_mapping : dict or None
        ``{original_label: encoded_int}`` mapping; available after
        :meth:`encode_categoricals`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.label_encoder: LabelEncoder | None = None
        self.primary_use_mapping: dict | None = None
        self._step_log: list[Tuple[str, int, int]] = []  # (step, before, after)
        logger.info(
            "ASHRAEPreprocessor initialized with %s rows", f"{len(self.df):,}"
        )

    # ------------------------------------------------------------------ #
    # Internal helper
    # ------------------------------------------------------------------ #
    def _record_step(self, step_name: str, rows_before: int) -> None:
        """Log and store row-count changes for the summary report."""
        rows_after = len(self.df)
        removed = rows_before - rows_after
        self._step_log.append((step_name, rows_before, rows_after))
        logger.info(
            "  [%s] %s → %s rows (removed %s)",
            step_name,
            f"{rows_before:,}",
            f"{rows_after:,}",
            f"{removed:,}",
        )

    # ------------------------------------------------------------------ #
    # Step 1 — Unit correction
    # ------------------------------------------------------------------ #
    def fix_site0_electricity(self) -> None:
        """Convert Site 0 electricity readings from kBTU to kWh.

        **Domain knowledge** (from competition discussion):
        Site 0 electricity meters (``meter == 0``) report energy in
        *kBTU* rather than *kWh*.  All other site/meter combinations
        already use kWh.

        Conversion factor: ``1 kBTU = 0.293071 kWh``.
        """
        KBTU_TO_KWH = 0.293071

        mask = (self.df["site_id"] == 0) & (self.df["meter"] == 0)
        n_affected = mask.sum()

        self.df.loc[mask, "meter_reading"] *= KBTU_TO_KWH

        logger.info(
            "fix_site0_electricity: converted %s rows (kBTU → kWh)",
            f"{n_affected:,}",
        )

    # ------------------------------------------------------------------ #
    # Step 2 — Anomaly removal
    # ------------------------------------------------------------------ #
    def remove_anomalous_readings(self) -> None:
        """Remove physically impossible or known-bad meter readings.

        Three sub-steps, each motivated by top Kaggle solutions:

        1. **Negative readings** — meters cannot produce negative energy.
        2. **Stuck meters** — ≥ 24 consecutive identical non-zero readings
           for the same (building_id, meter) indicate a frozen sensor.
        3. **Site 0 calibration period** — the first 141 days
           (timestamps before 2016-05-21) for Site 0 contain
           installation/calibration artefacts documented in the
           competition discussion.
        """
        rows_before = len(self.df)

        # --- 2a. Negative readings ---
        neg_mask = self.df["meter_reading"] < 0
        n_negative = neg_mask.sum()
        self.df = self.df[~neg_mask]
        logger.info("  Removed %s negative-reading rows", f"{n_negative:,}")

        # --- 2b. Stuck meters (≥ 24 h identical non-zero readings) ---
        # Sort to guarantee temporal order within each group
        self.df = self.df.sort_values(
            ["building_id", "meter", "timestamp"]
        ).reset_index(drop=True)

        # Identify runs of consecutive identical values per (building, meter)
        group_cols = ["building_id", "meter"]
        shifted = self.df.groupby(group_cols)["meter_reading"].shift(1)
        same_as_prev = (self.df["meter_reading"] == shifted) & (
            self.df["meter_reading"] != 0
        )

        # Compute run lengths using cumulative group technique
        block_id = (~same_as_prev).cumsum()
        run_lengths = block_id.map(block_id.groupby(block_id).transform("count"))
        stuck_mask = same_as_prev & (run_lengths >= 24)

        n_stuck = stuck_mask.sum()
        self.df = self.df[~stuck_mask]
        logger.info("  Flagged & removed %s stuck-meter rows", f"{n_stuck:,}")

        # --- 2c. Site 0 calibration period (first 141 days) ---
        calibration_cutoff = pd.Timestamp("2016-05-21")
        site0_cal_mask = (self.df["site_id"] == 0) & (
            self.df["timestamp"] < calibration_cutoff
        )
        n_cal = site0_cal_mask.sum()
        self.df = self.df[~site0_cal_mask]
        logger.info(
            "  Removed %s Site 0 calibration-period rows (< %s)",
            f"{n_cal:,}",
            calibration_cutoff.date(),
        )

        self._record_step("remove_anomalous_readings", rows_before)

    # ------------------------------------------------------------------ #
    # Step 3 — Weather imputation
    # ------------------------------------------------------------------ #
    def impute_weather(self) -> None:
        """Fill missing weather observations via interpolation and fill.

        Strategy per ``site_id`` group (sorted by timestamp):
            1. **Linear interpolation** with ``limit=4`` — fills gaps of
               up to 4 consecutive hours, matching the temporal resolution
               of the weather stations.
            2. **Forward-fill then backward-fill** — handles remaining
               NaN at series edges.

        Only numeric weather columns are imputed.
        """
        weather_cols = [
            "air_temperature",
            "cloud_coverage",
            "dew_temperature",
            "precip_depth_1_hr",
            "sea_level_pressure",
            "wind_direction",
            "wind_speed",
        ]
        # Keep only columns that actually exist in the DataFrame
        weather_cols = [c for c in weather_cols if c in self.df.columns]

        nans_before = self.df[weather_cols].isna().sum()
        logger.info("Weather NaN counts BEFORE imputation:")
        for col, cnt in nans_before.items():
            if cnt > 0:
                logger.info("  %-25s %s", col, f"{cnt:,}")

        # Sort once for all interpolation work
        self.df = self.df.sort_values(["site_id", "timestamp"]).reset_index(
            drop=True
        )

        # Interpolate within each site
        self.df[weather_cols] = self.df.groupby("site_id")[weather_cols].transform(
            lambda grp: grp.interpolate(method="linear", limit=4)
        )

        # Forward / backward fill for remaining edge NaN
        self.df[weather_cols] = self.df.groupby("site_id")[weather_cols].transform(
            lambda grp: grp.ffill().bfill()
        )

        nans_after = self.df[weather_cols].isna().sum()
        remaining = nans_after[nans_after > 0]
        if remaining.empty:
            logger.info("Weather imputation complete — no remaining NaN.")
        else:
            logger.warning("Weather columns with remaining NaN after imputation:")
            for col, cnt in remaining.items():
                logger.warning("  %-25s %s", col, f"{cnt:,}")

    # ------------------------------------------------------------------ #
    # Step 4 — Categorical encoding & temporal features
    # ------------------------------------------------------------------ #
    def encode_categoricals(self) -> None:
        """Encode ``primary_use`` and create temporal feature columns.

        Encodings
        ---------
        - ``primary_use`` → integer via :class:`~sklearn.preprocessing.LabelEncoder`.
          The fitted encoder and ``{label: code}`` mapping are stored on
          the instance for downstream use.
        - ``meter`` is already integer-coded; left as-is.

        New columns
        -----------
        - ``is_weekend``   : bool — Saturday or Sunday.
        - ``hour_of_day``  : int 0–23.
        - ``month``        : int 1–12.
        - ``day_of_week``  : int 0–6 (Monday = 0).
        """
        # --- primary_use encoding ---
        self.label_encoder = LabelEncoder()
        self.df["primary_use"] = self.label_encoder.fit_transform(
            self.df["primary_use"]
        )
        self.primary_use_mapping = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )
        logger.info(
            "Encoded primary_use (%d categories): %s",
            len(self.primary_use_mapping),
            self.primary_use_mapping,
        )

        # --- Temporal features ---
        ts = self.df["timestamp"]
        self.df["is_weekend"] = ts.dt.dayofweek >= 5
        self.df["hour_of_day"] = ts.dt.hour.astype(np.int8)
        self.df["month"] = ts.dt.month.astype(np.int8)
        self.df["day_of_week"] = ts.dt.dayofweek.astype(np.int8)

        logger.info(
            "Added temporal features: is_weekend, hour_of_day, month, day_of_week"
        )

    # ------------------------------------------------------------------ #
    # Step 5 — Stratified sample
    # ------------------------------------------------------------------ #
    def get_stratified_sample(
        self,
        n: int = 100_000,
        stratify_col: str = "primary_use",
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Return a stratified random sample of the cleaned DataFrame.

        The sample preserves the distribution of ``stratify_col`` as
        closely as possible.  Classes with fewer rows than their
        proportional share are included in full; surplus budget is
        redistributed to larger classes.

        Parameters
        ----------
        n : int
            Target sample size.
        stratify_col : str
            Column used for stratification.
        random_seed : int
            RNG seed for reproducibility (typically from ``config.yaml``).

        Returns
        -------
        pd.DataFrame
            Stratified subsample of ``self.df``.
        """
        if n >= len(self.df):
            logger.warning(
                "Requested sample size (%s) ≥ dataset size (%s); returning full dataset.",
                f"{n:,}",
                f"{len(self.df):,}",
            )
            return self.df.copy()

        # Proportional allocation with floor guarantee
        group_sizes = self.df[stratify_col].value_counts()
        proportions = group_sizes / len(self.df)
        alloc = (proportions * n).apply(np.floor).astype(int)

        # Ensure every group gets at least 1 row
        alloc = alloc.clip(lower=1)

        # Cap groups that are smaller than their allocation
        alloc = alloc.clip(upper=group_sizes)

        # Redistribute any surplus budget
        deficit = n - alloc.sum()
        if deficit > 0:
            eligible = group_sizes - alloc
            eligible = eligible[eligible > 0]
            extra = eligible.clip(upper=deficit)
            alloc[extra.index] += extra
            # Re-check total (may still be slightly under due to small groups)

        samples = []
        for group_val, group_n in alloc.items():
            group_df = self.df[self.df[stratify_col] == group_val]
            samples.append(
                group_df.sample(n=int(group_n), random_state=random_seed)
            )

        sample_df = pd.concat(samples, ignore_index=True)
        logger.info(
            "Stratified sample: %s rows (target %s)",
            f"{len(sample_df):,}",
            f"{n:,}",
        )
        return sample_df

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def run_full_pipeline(
        self, sample_size: int = 100_000, random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute every preprocessing step in sequence.

        Pipeline order:
            1. :meth:`fix_site0_electricity`
            2. :meth:`remove_anomalous_readings`
            3. :meth:`impute_weather`
            4. :meth:`encode_categoricals`
            5. :meth:`get_stratified_sample`

        Parameters
        ----------
        sample_size : int
            Rows in the stratified sample (passed to
            :meth:`get_stratified_sample`).
        random_seed : int
            RNG seed (passed to :meth:`get_stratified_sample`).

        Returns
        -------
        clean_df : pd.DataFrame
            Full cleaned dataset.
        sample_df : pd.DataFrame
            Stratified subsample for statistical notebooks.
        """
        total_before = len(self.df)
        logger.info("=" * 60)
        logger.info("Starting full preprocessing pipeline (%s rows)", f"{total_before:,}")
        logger.info("=" * 60)

        # 1 ─ Unit fix
        logger.info("─ Step 1/5: fix_site0_electricity")
        self.fix_site0_electricity()

        # 2 ─ Anomaly removal
        logger.info("─ Step 2/5: remove_anomalous_readings")
        self.remove_anomalous_readings()

        # 3 ─ Weather imputation
        logger.info("─ Step 3/5: impute_weather")
        self.impute_weather()

        # 4 ─ Encoding & features
        logger.info("─ Step 4/5: encode_categoricals")
        self.encode_categoricals()

        # 5 ─ Stratified sample
        logger.info("─ Step 5/5: get_stratified_sample")
        sample_df = self.get_stratified_sample(
            n=sample_size, random_seed=random_seed
        )

        # --- Summary report ---
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info("  %-35s %s", "Initial row count", f"{total_before:,}")
        for step_name, before, after in self._step_log:
            logger.info(
                "  %-35s %s → %s (-%s)",
                step_name,
                f"{before:,}",
                f"{after:,}",
                f"{before - after:,}",
            )
        logger.info("  %-35s %s", "Final cleaned rows", f"{len(self.df):,}")
        logger.info("  %-35s %s", "Stratified sample rows", f"{len(sample_df):,}")
        logger.info("  %-35s %d", "Columns", self.df.shape[1])
        logger.info("=" * 60)

        return self.df, sample_df


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _configure_root_logger()

    # Resolve project root (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "config.yaml"

    # --- Load & merge ---
    loader = ASHRAEDataLoader(config_path)
    train_df, meta_df, weather_df = loader.load_raw_data()
    merged_df = loader.merge_data(train_df, meta_df, weather_df)

    # Free memory from the raw frames
    del train_df, meta_df, weather_df

    # --- Preprocess ---
    preprocessor = ASHRAEPreprocessor(merged_df)
    del merged_df

    clean_df, sample_df = preprocessor.run_full_pipeline(
        sample_size=loader.config.get("sample_size_for_stats", 100_000),
        random_seed=loader.config.get("random_seed", 42),
    )

    # --- Save to parquet ---
    output_dir = project_root / loader.config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "clean_data.parquet"
    sample_path = output_dir / "sample_data.parquet"

    clean_df.to_parquet(clean_path, index=False, engine="pyarrow")
    logger.info("Saved cleaned data  → %s", clean_path)

    sample_df.to_parquet(sample_path, index=False, engine="pyarrow")
    logger.info("Saved sample data   → %s", sample_path)

    logger.info("Done.")
