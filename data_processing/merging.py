"""Merge logic — builds the national timeseries and metro panel datasets."""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "puerto rico": "PR", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

_METRO_SUFFIXES = [
    r"\s+Metro Area$",
    r"\s+Metropolitan Statistical Area$",
    r"\s+Micro Area$",
    r"\s+Micropolitan Statistical Area$",
]

# Manual overrides for metros where first-city matching fails (e.g. "New York" vs "New York-Newark").
# Add rows here as needed rather than touching the merge logic.
_DEFAULT_CROSSWALK = [
    ("new york|", "new york|NY"),
    ("washington|", "washington|DC"),
    ("minneapolis|", "minneapolis|MN"),
    ("virginia beach|", "virginia beach|VA"),
]


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("–", "-").replace("/", "-")
    return re.sub(r"\s+", " ", text)


def _state_abbrev(state) -> str:
    if pd.isna(state):
        return ""
    s = str(state).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return _STATE_ABBREV.get(s.lower(), s[:2].upper())


def _primary_city(name: str) -> str:
    name = _normalize(name)
    if "," in name:
        name = name.split(",", 1)[0]
    name = name.split("-")[0]
    name = re.sub(r"\bsaint\b", "st", name)
    name = re.sub(r"\bst\.\b", "st", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _standardize_metro_name(name: str) -> str:
    if pd.isna(name):
        return ""
    cleaned = _normalize(name)
    for pat in _METRO_SUFFIXES:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _apply_crosswalk(series: pd.Series, crosswalk: list[tuple]) -> pd.Series:
    mapping = dict(crosswalk)
    return series.replace(mapping)


def prepare_fred_monthly(fred_df: pd.DataFrame) -> pd.DataFrame:
    """Resample mixed-frequency FRED series down to month-end."""
    return (
        fred_df.copy()
        .sort_values("date")
        .set_index("date")
        .resample("ME").last()
        .reset_index()
    )


def _zillow_national(zhvi_metro: pd.DataFrame, zori_metro: pd.DataFrame) -> pd.DataFrame:
    zhvi = zhvi_metro.loc[zhvi_metro["RegionName"] == "United States", ["date", "zhvi"]].copy()
    zori = zori_metro.loc[zori_metro["RegionName"] == "United States", ["date", "zori"]].copy()
    return pd.merge(zhvi, zori, on="date", how="outer").sort_values("date").reset_index(drop=True)


def build_national_timeseries(
    fred_df: pd.DataFrame,
    zhvi_metro: pd.DataFrame,
    zori_metro: pd.DataFrame,
) -> pd.DataFrame:
    """Merge monthly FRED series with Zillow national aggregates on date."""
    fred = prepare_fred_monthly(fred_df)
    zillow = _zillow_national(zhvi_metro, zori_metro)
    merged = pd.merge(fred, zillow, on="date", how="outer").sort_values("date").reset_index(drop=True)
    return merged


def _build_zillow_panel(zhvi_metro: pd.DataFrame, zori_metro: pd.DataFrame) -> pd.DataFrame:
    zhvi = zhvi_metro.copy()
    zori = zori_metro.copy()

    for df in (zhvi, zori):
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["state_abbr"] = df["StateName"].apply(_state_abbrev)
        df["metro_std"] = df["RegionName"].apply(_standardize_metro_name)
        df["city"] = df["RegionName"].apply(_primary_city)
        df["join_key"] = df["city"] + "|" + df["state_abbr"]

    keep = ["RegionName", "StateName", "date", "year", "state_abbr", "metro_std", "city", "join_key"]
    panel = pd.merge(
        zhvi[keep + ["zhvi"]],
        zori[keep + ["zori"]],
        on=keep,
        how="outer",
    )

    panel = panel[panel["RegionName"] != "United States"]
    panel = panel[panel["year"].between(2009, 2023)]
    return panel.sort_values(["RegionName", "date"]).reset_index(drop=True)


def _prep_acs(acs_df: pd.DataFrame) -> pd.DataFrame:
    acs = acs_df.copy()
    acs["metro_std"] = acs["metro_name"].apply(_standardize_metro_name)
    acs["state_abbr"] = (
        acs["metro_name"].astype(str)
        .str.extract(r",\s*([A-Z]{2})(?:-[A-Z]{2})*")
        .fillna("")
    )
    acs["city"] = acs["metro_std"].apply(_primary_city)
    acs["join_key"] = acs["city"] + "|" + acs["state_abbr"]
    return acs[acs["year"].between(2009, 2023)].copy()


def _add_yoy_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year percent change for home values and rents."""
    panel = panel.sort_values(["RegionName", "date"]).reset_index(drop=True)
    panel["zhvi_yoy_pct"] = (
        panel.groupby("RegionName", group_keys=False)["zhvi"]
        .pct_change(12, fill_method=None)
        .mul(100)
        .round(2)
    )
    panel["zori_yoy_pct"] = (
        panel.groupby("RegionName", group_keys=False)["zori"]
        .pct_change(12, fill_method=None)
        .mul(100)
        .round(2)
    )
    return panel


def build_metro_panel(
    zhvi_metro: pd.DataFrame,
    zori_metro: pd.DataFrame,
    acs_df: pd.DataFrame,
    crosswalk: Optional[list[tuple]] = None,
) -> pd.DataFrame:
    """Merge Zillow metro panel with ACS metro-year data.

    crosswalk: list of (bad_key, good_key) tuples to fix name-matching failures
    for metros where the first-city heuristic breaks (e.g. New York, Washington DC).
    Defaults to _DEFAULT_CROSSWALK.
    """
    zillow = _build_zillow_panel(zhvi_metro, zori_metro)
    acs = _prep_acs(acs_df)

    overrides = crosswalk if crosswalk is not None else _DEFAULT_CROSSWALK
    if overrides:
        zillow["join_key"] = _apply_crosswalk(zillow["join_key"], overrides)
        acs["join_key"] = _apply_crosswalk(acs["join_key"], overrides)

    merged = pd.merge(zillow, acs, on=["join_key", "year"], how="left", suffixes=("", "_acs"))

    income = merged.get("median_household_income")
    if income is not None:
        income = income.replace(0, pd.NA)
        merged["zhvi_to_income_ratio"] = merged["zhvi"] / income
        merged["annual_zori_to_income_ratio"] = (merged["zori"] * 12) / income

    merged = _add_yoy_columns(merged)
    return merged.sort_values(["RegionName", "date"]).reset_index(drop=True)


def summarize_merge_quality(metro_panel: pd.DataFrame) -> dict:
    total = len(metro_panel)
    matched = int(metro_panel["metro_cbsa_code"].notna().sum()) if "metro_cbsa_code" in metro_panel.columns else 0
    return {
        "total_rows": total,
        "matched_rows": matched,
        "match_rate": round(matched / total, 3) if total else 0.0,
    }
