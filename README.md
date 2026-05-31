# Housing Affordability Pipeline

**Team:** Qianru Guo, Shifan Zhao, Jeremy Sutikno, Joyce Liang

---

## Setup

### 1. Install dependencies

```bash
pip install pandas pyarrow requests python-dotenv pytest panel plotly
```

### 2. Set API keys

Create a file named `.env` in the root project folder:

```
FRED_API_KEY=your_fred_key_here
CENSUS_API_KEY=your_census_key_here
```

On Windows (terminal):

```bash
echo FRED_API_KEY=your_key > .env
echo CENSUS_API_KEY=your_key >> .env
```

Zillow data requires no API key.

---

## Running the Pipeline

Run from the root project folder:

```bash
python data_processing/pipeline.py
```

This fetches from FRED, Zillow, and Census APIs — takes about 3–5 minutes on first run.

On subsequent runs, skip the API calls and just re-run cleaning and merging:

```bash
python data_processing/pipeline.py --skip-fetch
```

---

## Running the Dashboard

```bash
panel serve data_processing/dashboard.py --show
```

Opens at `http://localhost:5006/dashboard`

The dashboard has four tabs:
- **National Overview** — home values, mortgage rates, and CPI vs Zillow rent trends
- **Metro Drill-Down** — compare home values and rents across up to 50 metros
- **Growth Race** — animated bar chart of the 15 most expensive metros over time
- **Rent vs. Buy Calculator** — estimate affordability based on your income and local market

---

## Running Tests

```bash
pytest tests/ -v
```

`test_parquet_files_exist` requires the pipeline to have run at least once. All other 9 tests run immediately without API keys.

---

## Architecture

```
DS3500/
  data_acquisition/     <- fred_fetcher.py, zillow_fetcher.py, census_fetcher.py
  data_processing/      <- cleaning.py, merging.py, pipeline.py, dashboard.py
  tests/                <- test_pipeline.py
  .env                  <- API keys (not committed to git)
  .gitignore
  README.md
```

Data flow:

```
FRED API + Zillow CSVs + Census ACS API
  -> data_storage/raw/          (cached during pipeline run)
  -> cleaning.py                (per-source cleaning and validation)
  -> data_storage/cleaned/      (one parquet per source)
  -> merging.py                 (national timeseries + metro panel)
  -> data_storage/processed/    (final analytical datasets)
  -> dashboard.py               (reads from processed/ only)
```

### Data sources

| Source | Type | Key fields |
|---|---|---|
| FRED API | API | Mortgage rates, home prices, CPI, Fed funds rate, income, housing starts |
| Zillow Research | CSV download | ZHVI (home values), ZORI (rents) at metro and state level |
| Census ACS 5-year | API | Median income, home value, rent, homeownership rate by metro (2009–2023) |

---

## Data Cleaning Decisions

### FRED
All series are merged on date after being fetched as raw JSON. We deduplicate by retaining the most recent record for each date and drop rows where the date could not be parsed. Different series publish at different frequencies (weekly, monthly, quarterly) so all are resampled to month-end using `.resample('ME').last()` before the national merge.

### Zillow
Raw CSVs arrive in wide format (one column per month) and are melted to long format first. Whitespace is stripped from region and state name columns. Rows missing a date or value are dropped. Duplicate `(RegionName, StateName, date)` rows are removed. The United States row is used for the national timeseries and excluded from the metro panel.

### Census ACS
The Census API uses large negative integers (`-666666666`, `-999999999`, etc.) instead of null for suppressed or missing data. All sentinel values are replaced with NaN before any computation. Micropolitan areas are excluded to keep the data comparable with Zillow's metro coverage. Affordability ratios (home value to income, rent to income, homeownership rate) are computed after cleaning with zero and NaN denominators protected against.

---

## Known Data Quality Issues

| Issue | How It Was Handled |
|---|---|
| FRED series publish at different frequencies | Resampled to month-end using `.last()` before national merge |
| Zillow CSVs in wide format | Melted to long format during fetch |
| Census API uses sentinel integers instead of NaN | All known sentinel values replaced with NaN before computation |
| Metro names differ between Zillow and Census | Primary city + state abbreviation used as join key; default crosswalk included for known problem metros (New York, Washington DC, Minneapolis, Virginia Beach) |
| ACS only covers 2009–2023 while Zillow/FRED go back to 2000 | Metro panel limited to overlapping years; national timeseries uses full range |
| Zillow does not publish state-level rent data | Approximated by population-weighted average of metro-level ZORI by state |
| Some Zillow metros have no matching ACS record (~58% unmatched) | Left join preserves all Zillow rows; unmatched ACS columns come through as NaN |
