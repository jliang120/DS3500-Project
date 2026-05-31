"""Cleaning and validation for each data source."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_FRED_REQUIRED = ["date"]
_ZILLOW_REQUIRED = ["RegionName", "StateName", "date"]
_ACS_REQUIRED = ["metro_name", "metro_cbsa_code", "year"]


def _check_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    skip = set(exclude)
    for col in df.columns:
        if col not in skip:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df


def clean_fred_data(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, _FRED_REQUIRED, "FRED")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = _coerce_numeric(out, exclude=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def clean_zillow_data(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    _check_columns(df, _ZILLOW_REQUIRED + [value_col], "Zillow")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    for col in ["RegionName", "StateName", "RegionType"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    out = out.dropna(subset=["date", "RegionName", value_col])
    dedup_keys = [c for c in ["RegionID", "RegionName", "StateName", "date"] if c in out.columns]
    out = out.drop_duplicates(subset=dedup_keys, keep="last")
    sort_cols = [c for c in ["RegionName", "date"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def clean_acs_data(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, _ACS_REQUIRED, "ACS")
    out = df.copy()

    str_cols = {"metro_name", "metro_cbsa_code", "acs_period"}
    for col in out.columns:
        if col not in str_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["metro_name"] = out["metro_name"].astype(str).str.strip()
    out["metro_cbsa_code"] = out["metro_cbsa_code"].astype(str).str.strip()
    out = out.dropna(subset=["metro_name", "metro_cbsa_code", "year"])
    out = out.drop_duplicates(subset=["metro_cbsa_code", "year"], keep="last")
    return out.sort_values(["metro_cbsa_code", "year"]).reset_index(drop=True)


def validate_no_duplicate_keys(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    dupes = df.duplicated(subset=list(keys)).sum()
    if dupes:
        raise ValueError(f"{name}: {dupes} duplicate rows on keys {list(keys)}")


def validate_no_null_keys(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    nulls = df[list(keys)].isna().any(axis=1).sum()
    if nulls:
        raise ValueError(f"{name}: {nulls} rows with null key values in {list(keys)}")


def validate_nonnegative(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    bad = [c for c in columns if c in df.columns and (df[c].dropna() < 0).any()]
    if bad:
        raise ValueError(f"{name}: negative values found in {bad}")


def run_basic_validations(
    fred_df: pd.DataFrame,
    zhvi_df: pd.DataFrame,
    zori_df: pd.DataFrame,
    acs_df: pd.DataFrame,
) -> None:
    validate_no_null_keys(fred_df, ["date"], "FRED")
    validate_no_duplicate_keys(fred_df, ["date"], "FRED")

    for label, df, val_col in [("ZHVI", zhvi_df, "zhvi"), ("ZORI", zori_df, "zori")]:
        validate_no_null_keys(df, ["RegionName", "StateName", "date"], label)
        validate_no_duplicate_keys(df, ["RegionName", "StateName", "date"], label)
        validate_nonnegative(df, [val_col], label)

    validate_no_null_keys(acs_df, ["metro_name", "metro_cbsa_code", "year"], "ACS")
    validate_no_duplicate_keys(acs_df, ["metro_cbsa_code", "year"], "ACS")
    validate_nonnegative(
        acs_df,
        ["median_household_income", "median_home_value", "median_gross_rent", "total_population"],
        "ACS",
    )

    logger.info("All validation checks passed")
