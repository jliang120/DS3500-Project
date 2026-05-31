"""End-to-end pipeline for the housing affordability project."""

from __future__ import annotations

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_acq  = os.path.join(_here, '..', 'data_acquisition')
if _acq not in sys.path:
    sys.path.insert(0, _acq)

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from fred_fetcher import fetch_all_fred_series
from zillow_fetcher import fetch_zhvi_metro, fetch_zori_metro
from census_fetcher import fetch_all_acs_metros, compute_affordability_metrics
from cleaning import clean_acs_data, clean_fred_data, clean_zillow_data, run_basic_validations
from merging import build_metro_panel, build_national_timeseries, summarize_merge_quality

logger = logging.getLogger(__name__)


def _save(df: pd.DataFrame, path: Path) -> Path:
    """Save as parquet, fall back to CSV if pyarrow isn't available."""
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logger.warning("pyarrow unavailable, saved as CSV: %s", fallback)
        return fallback


def _load_cached(save_dir: str) -> Dict[str, pd.DataFrame]:
    """Load previously saved raw data from disk to skip re-fetching APIs."""
    raw_dir = Path(save_dir) / "raw"
    expected = {
        "fred": raw_dir / "fred_raw.parquet",
        "zhvi_metro": raw_dir / "zillow_zhvi_metro.csv",
        "zori_metro": raw_dir / "zillow_zori_metro.csv",
        "acs": raw_dir / "census_acs_metros.parquet",
    }

    missing = [k for k, p in expected.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"skip_fetch=True but cached files missing: {missing}. "
            "Run with skip_fetch=False first."
        )

    raw = {}
    for key, path in expected.items():
        raw[key] = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info("Loaded cached %s from %s", key, path)
    return raw


def fetch_raw_data(
    save_dir: str = "data_storage",
    fred_api_key: Optional[str] = None,
    census_api_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(save_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fred_df = fetch_all_fred_series(api_key=fred_api_key)
    zhvi_metro = fetch_zhvi_metro(save_dir=str(raw_dir))
    zori_metro = fetch_zori_metro(save_dir=str(raw_dir))

    acs_df = fetch_all_acs_metros(api_key=census_api_key)
    if acs_df.empty:
        raise ValueError("ACS fetch returned no data — check your Census API key and response logs.")
    acs_df = compute_affordability_metrics(acs_df)

    # Cache raw FRED and ACS for skip_fetch runs
    _save(fred_df, raw_dir / "fred_raw.parquet")
    _save(acs_df, raw_dir / "census_acs_metros.parquet")

    return {"fred": fred_df, "zhvi_metro": zhvi_metro, "zori_metro": zori_metro, "acs": acs_df}


def clean_source_data(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = {
        "fred": clean_fred_data(raw["fred"]),
        "zhvi_metro": clean_zillow_data(raw["zhvi_metro"], value_col="zhvi"),
        "zori_metro": clean_zillow_data(raw["zori_metro"], value_col="zori"),
        "acs": clean_acs_data(raw["acs"]),
    }
    run_basic_validations(cleaned["fred"], cleaned["zhvi_metro"], cleaned["zori_metro"], cleaned["acs"])
    return cleaned


def merge_data(cleaned: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    national = build_national_timeseries(cleaned["fred"], cleaned["zhvi_metro"], cleaned["zori_metro"])
    metro = build_metro_panel(cleaned["zhvi_metro"], cleaned["zori_metro"], cleaned["acs"])
    logger.info("Metro merge quality: %s", summarize_merge_quality(metro))
    return {"national_timeseries": national, "metro_panel": metro}


def save_outputs(
    cleaned: Dict[str, pd.DataFrame],
    merged: Dict[str, pd.DataFrame],
    save_dir: str = "data_storage",
) -> Dict[str, str]:
    base = Path(save_dir)
    cleaned_dir = base / "cleaned"
    processed_dir = base / "processed"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in cleaned.items():
        paths[name] = str(_save(df, cleaned_dir / f"{name}.parquet"))
    for name, df in merged.items():
        paths[name] = str(_save(df, processed_dir / f"{name}.parquet"))
    return paths


def run_pipeline(
    save_dir: str = "data_storage",
    fred_api_key: Optional[str] = None,
    census_api_key: Optional[str] = None,
    skip_fetch: bool = False,
) -> Dict[str, object]:
    """Run the full pipeline: fetch -> clean -> merge -> save.

    skip_fetch=True loads previously cached raw files from disk instead of
    hitting the APIs again — useful when iterating on cleaning or merge logic.
    """
    if skip_fetch:
        logger.info("skip_fetch=True — loading cached raw data")
        raw = _load_cached(save_dir)
    else:
        raw = fetch_raw_data(save_dir=save_dir, fred_api_key=fred_api_key, census_api_key=census_api_key)

    cleaned = clean_source_data(raw)
    merged = merge_data(cleaned)
    saved_paths = save_outputs(cleaned, merged, save_dir=save_dir)

    return {"raw": raw, "cleaned": cleaned, "merged": merged, "saved_paths": saved_paths}


if __name__ == "__main__":
    import argparse
    import dotenv

    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Housing affordability pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Load cached raw data, skip API calls")
    parser.add_argument("--output-dir", default=os.environ.get("HOUSING_PROJECT_OUTPUT_DIR", "data_storage"))
    args = parser.parse_args()

    results = run_pipeline(save_dir=args.output_dir, skip_fetch=args.skip_fetch)
    print("Pipeline complete.")
    for name, path in results["saved_paths"].items():
        print(f"  {name}: {path}")
