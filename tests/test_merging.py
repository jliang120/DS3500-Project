import pandas as pd
from merging import build_national_timeseries, build_metro_panel, summarize_merge_quality

# fake FRED data
fred = pd.DataFrame({
    "date": pd.date_range("2015-01", periods=12, freq="MS"),
    "mortgage_rate": [3.5] * 12,
})

# fake Zillow data
dates = pd.date_range("2015-01", periods=12, freq="MS")
zhvi = pd.DataFrame({
    "RegionName": ["United States"] * 12 + ["Houston, TX"] * 12,
    "StateName": [""] * 12 + ["TX"] * 12,
    "date": list(dates) * 2,
    "zhvi": [200000] * 12 + [180000] * 12,
})
zori = pd.DataFrame({
    "RegionName": ["United States"] * 12 + ["Houston, TX"] * 12,
    "StateName": [""] * 12 + ["TX"] * 12,
    "date": list(dates) * 2,
    "zori": [1500] * 12 + [1200] * 12,
})

# fake ACS data
acs = pd.DataFrame({
    "metro_name": ["Houston-The Woodlands-Sugar Land, TX"],
    "metro_cbsa_code": ["26420"],
    "year": [2015],
    "median_household_income": [60000],
    "median_home_value": [180000],
    "median_gross_rent": [1200],
    "total_population": [700000],
})

# run it
national = build_national_timeseries(fred, zhvi, zori)
print("national shape:", national.shape)
print(national.head())

metro = build_metro_panel(zhvi, zori, acs)
print("\nmetro shape:", metro.shape)
print(metro[["RegionName", "date", "zhvi", "zori", "zhvi_yoy_pct"]].head())

print("\nmerge quality:", summarize_merge_quality(metro))