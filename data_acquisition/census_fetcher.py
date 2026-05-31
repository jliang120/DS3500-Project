"""Census ACS fetcher — metro-level housing and income data from the 5-year estimates."""

import os
import time
import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

CENSUS_BASE_URL = "https://api.census.gov/data"
_RATE_LIMIT_DELAY = 0.3

# Census returns large negative integers instead of null for suppressed/missing data
_SENTINEL_VALUES = {
    -666666666,
    -999999999,
    -888888888,
    -222222222,
    -333333333,
    -555555555,
}

ACS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",
    "B25071_001E": "median_rent_pct_income",
    "B25003_001E": "total_housing_units_tenure",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",
    "B01003_001E": "total_population",
}

# 2009 = 2005-2009 estimates. Latest release as of early 2026 is 2023 (2019-2023).
ACS_YEARS = list(range(2009, 2024))


def _get_api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise ValueError(
            "CENSUS_API_KEY not set. Get a free key at "
            "https://api.census.gov/data/key_signup.html"
        )
    return key


def _replace_sentinels(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert Census columns to numeric and replace sentinel missing values with NaN."""
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(~df[col].isin(_SENTINEL_VALUES))
    return df


def fetch_acs_metro_year(year: int, api_key: str, variables: Optional[dict] = None) -> pd.DataFrame:
    """Fetch ACS 5-year estimates for all MSAs in a given vintage year."""
    if variables is None:
        variables = ACS_VARIABLES

    var_string = ",".join(["NAME"] + list(variables.keys()))
    url = f"{CENSUS_BASE_URL}/{year}/acs/acs5"
    params = {
        "get": var_string,
        "for": "metropolitan statistical area/micropolitan statistical area:*",
        "key": api_key,
    }

    logger.info("Fetching ACS %d", year)

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(f"ACS {year} request timed out")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.HTTPError(f"ACS {year} HTTP error: {e}") from e

    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Rename the CBSA geo column (name varies slightly across years)
    geo_col = [c for c in df.columns if "metropolitan" in c.lower() or "micropolitan" in c.lower()]
    if geo_col:
        df = df.rename(columns={geo_col[0]: "metro_cbsa_code"})

    df = df.rename(columns={"NAME": "metro_name", **variables})
    df = _replace_sentinels(df, list(variables.values()))

    # Drop micropolitan areas — only keep Metro Areas
    df = df[df["metro_name"].str.contains("Metro Area", case=False, na=False)]

    df["year"] = year
    df["acs_period"] = f"{year - 4}-{year}"

    logger.info("  %d metros for %d", len(df), year)
    return df


def fetch_all_acs_metros(api_key: Optional[str] = None, years: Optional[list] = None) -> pd.DataFrame:
    """Fetch ACS data across all available years, return a combined panel."""
    if api_key is None:
        api_key = _get_api_key()
    if years is None:
        years = ACS_YEARS

    frames = []
    for year in years:
        try:
            frames.append(fetch_acs_metro_year(year, api_key))
            time.sleep(_RATE_LIMIT_DELAY)
        except requests.exceptions.Timeout:
            logger.warning("Skipping ACS %d — timed out", year)
        except Exception as e:
            logger.warning("Skipping ACS %d — %s", year, e)

    if not frames:
        logger.error("No ACS data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["metro_cbsa_code", "year"]).reset_index(drop=True)
    logger.info(
        "ACS combined: %d rows, %d metros, %d years",
        len(combined), combined["metro_cbsa_code"].nunique(), combined["year"].nunique(),
    )
    return combined


def compute_affordability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add home-value-to-income ratio, rent-to-income ratio, and homeownership rate."""
    df = df.copy()
    income = df["median_household_income"].replace(0, pd.NA)
    tenure = df["total_housing_units_tenure"].replace(0, pd.NA)

    df["home_value_to_income_ratio"] = df["median_home_value"] / income
    df["annual_rent_to_income_ratio"] = (df["median_gross_rent"] * 12) / income
    df["homeownership_rate"] = df["owner_occupied_units"] / tenure * 100
    return df


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)

    df = fetch_all_acs_metros()
    df = compute_affordability_metrics(df)
    print(f"Fetched {len(df)} rows")
    print(df.head(10))

    os.makedirs("data_storage", exist_ok=True)
    df.to_parquet("data_storage/census_acs_metros.parquet", index=False)
    print("Saved to data_storage/census_acs_metros.parquet")
