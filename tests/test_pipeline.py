import sys
import os

# Insert paths at position 0 so these modules are found before anything else.
# Using insert(0) rather than append avoids Python picking up pipeline.py
# from data_processing/ as a side-effect during merging/cleaning import.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [os.path.join(_root, 'data_acquisition'),
           os.path.join(_root, 'data_processing')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
import pandas as pd
import numpy as np

from cleaning import (
    clean_fred_data,
    clean_zillow_data,
    clean_acs_data,
    validate_no_duplicate_keys,
    validate_nonnegative,
    validate_no_null_keys,
)
from merging import build_national_timeseries, build_metro_panel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fred_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-07", "2020-01-14", "2020-02-01"]),
        "MORTGAGE30US": [3.5, 3.6, np.nan, 3.4],
        "FEDFUNDS": [1.75, np.nan, np.nan, 1.75],
    })


@pytest.fixture
def zhvi_df():
    return pd.DataFrame({
        "RegionID": [1, 1, 2, 2],
        "RegionName": ["Boston, MA", "Boston, MA", "United States", "United States"],
        "RegionType": ["msa", "msa", "country", "country"],
        "StateName": ["MA", "MA", np.nan, np.nan],
        "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"]),
        "zhvi": [450000.0, 455000.0, 250000.0, 252000.0],
    })


@pytest.fixture
def zori_df():
    return pd.DataFrame({
        "RegionID": [1, 1, 2, 2],
        "RegionName": ["Boston, MA", "Boston, MA", "United States", "United States"],
        "RegionType": ["msa", "msa", "country", "country"],
        "StateName": ["MA", "MA", np.nan, np.nan],
        "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"]),
        "zori": [2200.0, 2250.0, 1500.0, 1520.0],
    })


@pytest.fixture
def acs_df():
    return pd.DataFrame({
        "metro_name": [
            "Boston-Cambridge-Newton, MA-NH Metro Area",
            "Boston-Cambridge-Newton, MA-NH Metro Area",
        ],
        "metro_cbsa_code": ["14460", "14460"],
        "year": [2020, 2021],
        "acs_period": ["2016-2020", "2017-2021"],
        "median_household_income": [85000, 87000],
        "median_home_value": [450000, 460000],
        "median_gross_rent": [1800, 1850],
        "median_rent_pct_income": [25.0, 25.5],
        "total_housing_units_tenure": [500000, 505000],
        "owner_occupied_units": [300000, 303000],
        "renter_occupied_units": [200000, 202000],
        "total_population": [1000000, 1010000],
        "home_value_to_income_ratio": [5.29, 5.29],
        "annual_rent_to_income_ratio": [0.254, 0.255],
        "homeownership_rate": [60.0, 60.0],
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_parquet_files_exist():
    """Integration test — checks that pipeline outputs actually exist on disk."""
    base = os.path.join(os.path.dirname(__file__), "..", "data_processing", "data_storage")
    for fname in [
        "cleaned/fred.parquet",
        "cleaned/acs.parquet",
        "cleaned/zhvi_metro.parquet",
        "cleaned/zori_metro.parquet",
        "processed/national_timeseries.parquet",
        "processed/metro_panel.parquet",
    ]:
        path = os.path.join(base, fname)
        assert os.path.exists(path), f"Missing: {path}"
        assert len(pd.read_parquet(path)) > 0, f"Empty: {path}"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_clean_fred_removes_duplicates(fred_df):
    duped = pd.concat([fred_df, fred_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_fred_data(duped)
    assert cleaned["date"].duplicated().sum() == 0


def test_clean_fred_date_is_datetime(fred_df):
    df = fred_df.copy()
    df["date"] = df["date"].astype(str)
    cleaned = clean_fred_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_clean_zillow_removes_duplicates(zhvi_df):
    duped = pd.concat([zhvi_df, zhvi_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_zillow_data(duped, value_col="zhvi")
    assert cleaned.duplicated(subset=["RegionName", "StateName", "date"]).sum() == 0


def test_clean_acs_removes_duplicates(acs_df):
    duped = pd.concat([acs_df, acs_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_acs_data(duped)
    assert cleaned.duplicated(subset=["metro_cbsa_code", "year"]).sum() == 0


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def test_national_timeseries_is_nonempty(fred_df, zhvi_df, zori_df):
    result = build_national_timeseries(fred_df, zhvi_df, zori_df)
    assert len(result) > 0
    assert "date" in result.columns


def test_metro_panel_has_expected_columns(zhvi_df, zori_df, acs_df):
    result = build_metro_panel(zhvi_df, zori_df, acs_df)
    for col in ["RegionName", "year", "zhvi"]:
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_zhvi_values_nonnegative(zhvi_df):
    cleaned = clean_zillow_data(zhvi_df, value_col="zhvi")
    assert (cleaned["zhvi"].dropna() >= 0).all()


def test_acs_income_nonnegative(acs_df):
    cleaned = clean_acs_data(acs_df)
    assert (cleaned["median_household_income"].dropna() >= 0).all()


def test_acs_no_null_keys(acs_df):
    cleaned = clean_acs_data(acs_df)
    assert cleaned["metro_name"].isna().sum() == 0
    assert cleaned["metro_cbsa_code"].isna().sum() == 0
