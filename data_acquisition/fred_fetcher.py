"""FRED API fetcher — pulls macro time series for the housing affordability pipeline."""

import os
import time
import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

FRED_SERIES = {
    "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average (%)",
    "MSPUS": "Median Sales Price of Houses Sold (USD)",
    "MEHOINUSA672N": "Real Median Household Income (USD)",
    "HOUST": "Housing Starts: Total New Privately Owned (Thousands)",
    "CUSR0000SAH1": "CPI Shelter Index (1982-84=100)",
    "FEDFUNDS": "Federal Funds Effective Rate (%)",
    "RHORUSQ156N": "Homeownership Rate (%)",
    "HOSINVUSM495N": "Housing Inventory: Existing Home Sales (Thousands)",
    "CPIAUCSL": "CPI All Urban Consumers (1982-84=100)",
    "MSACSR": "Monthly Supply of New Houses (Months)",
    "PERMIT": "New Privately-Owned Housing Units Authorized (Thousands)",
    "TTLHH": "Total Households (Thousands)",
}

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "2026-03-01"
_RATE_LIMIT_DELAY = 0.5


def _get_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def fetch_single_series(
    series_id: str,
    api_key: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Fetch one FRED series and return a two-column DataFrame (date, series_id)."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }

    logger.info("Fetching FRED series: %s", series_id)

    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(f"Timed out fetching {series_id}")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.HTTPError(f"HTTP {e.response.status_code} for {series_id}") from e

    data = response.json()
    if "observations" not in data:
        raise ValueError(f"Unexpected response for {series_id}: {data}")

    # FRED uses "." as its missing value sentinel
    records = [
        {"date": obs["date"], series_id: float(obs["value"])}
        for obs in data["observations"]
        if obs["value"] != "."
    ]

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    logger.info("  %s: %d observations", series_id, len(df))
    return df


def fetch_all_fred_series(
    api_key: Optional[str] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    series_ids: Optional[list] = None,
) -> pd.DataFrame:
    """Fetch all configured FRED series and outer-join them on date.

    Different series publish at different frequencies (weekly, monthly, annual)
    so an outer join is intentional — don't want to drop dates where only some
    series have data.
    """
    if api_key is None:
        api_key = _get_api_key()

    targets = series_ids or list(FRED_SERIES.keys())
    merged = None

    for sid in targets:
        try:
            df = fetch_single_series(sid, api_key, start_date, end_date)
            merged = df if merged is None else pd.merge(merged, df, on="date", how="outer")
            time.sleep(_RATE_LIMIT_DELAY)
        except requests.exceptions.Timeout:
            logger.warning("Skipping %s — request timed out", sid)
        except Exception as e:
            logger.warning("Skipping %s — %s", sid, e)

    if merged is not None:
        merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def get_series_metadata() -> dict:
    return FRED_SERIES.copy()


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)

    df = fetch_all_fred_series()
    print(f"Fetched {len(df)} rows, {len(df.columns)} columns")
    print(df.head(10))

    os.makedirs("data_storage", exist_ok=True)
    df.to_parquet("data_storage/fred_raw.parquet", index=False)
    print("Saved to data_storage/fred_raw.parquet")
