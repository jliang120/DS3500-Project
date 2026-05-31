"""Zillow Research data fetcher — downloads ZHVI and ZORI CSVs and converts to long format."""

import os
import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# CSV download URLs from https://www.zillow.com/research/data/
ZILLOW_DATASETS = {
    "zhvi_metro": {
        "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "description": "ZHVI All Homes - Metro Level, Middle Tier, Smoothed & Seasonally Adjusted",
    },
    "zhvi_state": {
        "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/State_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "description": "ZHVI All Homes - State Level, Middle Tier, Smoothed & Seasonally Adjusted",
    },
    "zori_metro": {
        "url": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv",
        "description": "ZORI All Homes + Multifamily - Metro Level, Smoothed & Seasonally Adjusted",
    },
    # Note: Zillow doesn't publish state-level ZORI. Use derive_zori_state() instead.
}

_TIMEOUT = 120


def download_zillow_csv(dataset_key: str, save_dir: str = "data_storage") -> str:
    """Download a Zillow CSV to disk, return the saved filepath."""
    if dataset_key not in ZILLOW_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Options: {list(ZILLOW_DATASETS)}")

    info = ZILLOW_DATASETS[dataset_key]
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"zillow_{dataset_key}.csv")

    logger.info("Downloading %s", info["description"])
    response = requests.get(info["url"], timeout=_TIMEOUT, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info("  Saved to %s", filepath)
    return filepath


def parse_zillow_wide_to_long(filepath: str, value_name: str = "value") -> pd.DataFrame:
    """Melt a Zillow wide-format CSV into long format.

    Zillow CSVs ship with one column per month (e.g. '2000-01-31', '2000-02-29').
    This unpivots those date columns into rows so the data is easier to filter
    and merge downstream.
    """
    df = pd.read_csv(filepath)

    id_cols, date_cols = [], []
    for col in df.columns:
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except (ValueError, TypeError):
            id_cols.append(col)

    logger.info("  %d date columns, %d metadata columns", len(date_cols), len(id_cols))

    long = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name=value_name)
    long["date"] = pd.to_datetime(long["date"])
    long = long.dropna(subset=[value_name])
    return long


def fetch_zhvi_metro(save_dir: str = "data_storage") -> pd.DataFrame:
    fp = download_zillow_csv("zhvi_metro", save_dir)
    df = parse_zillow_wide_to_long(fp, value_name="zhvi")
    logger.info("  ZHVI metro: %d rows across %d metros", len(df), df["RegionName"].nunique())
    return df


def fetch_zori_metro(save_dir: str = "data_storage") -> pd.DataFrame:
    fp = download_zillow_csv("zori_metro", save_dir)
    df = parse_zillow_wide_to_long(fp, value_name="zori")
    logger.info("  ZORI metro: %d rows across %d metros", len(df), df["RegionName"].nunique())
    return df


def fetch_zhvi_state(save_dir: str = "data_storage") -> pd.DataFrame:
    fp = download_zillow_csv("zhvi_state", save_dir)
    return parse_zillow_wide_to_long(fp, value_name="zhvi")


def derive_zori_state(
    zori_metro_df: Optional[pd.DataFrame] = None,
    acs_population_df: Optional[pd.DataFrame] = None,
    save_dir: str = "data_storage",
) -> pd.DataFrame:
    """Aggregate metro ZORI to state level using population-weighted averaging.

    Zillow doesn't publish native state-level rent data. If acs_population_df
    is provided (needs columns: StateName, total_population), this weights each
    metro by its population. Otherwise falls back to a simple unweighted mean.
    """
    if zori_metro_df is None:
        zori_metro_df = fetch_zori_metro(save_dir)

    if "StateName" not in zori_metro_df.columns:
        raise ValueError("Metro ZORI data is missing 'StateName'")

    if acs_population_df is not None:
        # population-weighted mean
        pop = acs_population_df[["StateName", "total_population"]].drop_duplicates()
        merged = zori_metro_df.merge(pop, on="StateName", how="left")
        merged["total_population"] = merged["total_population"].fillna(1)

        # weighted mean: sum(zori * pop) / sum(pop) per state-month
        merged["weighted_zori"] = merged["zori"] * merged["total_population"]
        df_state = (
            merged.groupby(["StateName", "date"], as_index=False)
            .agg(weighted_zori=("weighted_zori", "sum"), total_population=("total_population", "sum"))
        )
        df_state["zori"] = df_state["weighted_zori"] / df_state["total_population"]
        df_state = df_state[["StateName", "date", "zori"]]
        logger.info("  ZORI state (population-weighted): %d rows", len(df_state))
    else:
        logger.warning("No population data provided — using unweighted state ZORI mean")
        df_state = (
            zori_metro_df
            .groupby(["StateName", "date"], as_index=False)["zori"]
            .mean()
        )

    return df_state


def fetch_all_zillow(save_dir: str = "data_storage") -> dict:
    results = {}
    for key in ("zhvi_metro", "zori_metro", "zhvi_state"):
        try:
            fetcher = {"zhvi_metro": fetch_zhvi_metro, "zori_metro": fetch_zori_metro,
                       "zhvi_state": fetch_zhvi_state}[key]
            results[key] = fetcher(save_dir)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", key, e)

    try:
        results["zori_state"] = derive_zori_state(
            zori_metro_df=results.get("zori_metro"),
            save_dir=save_dir,
        )
    except Exception as e:
        logger.warning("Failed to derive zori_state: %s", e)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets = fetch_all_zillow()
    for key, df in datasets.items():
        print(f"\n{key}: {df.shape}")
        print(df.head(3))
