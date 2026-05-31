"""Housing Affordability Dashboard — Panel + Plotly."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_acquisition'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import panel as pn

pn.extension("plotly", sizing_mode="stretch_width")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_data_dir() -> Path:
    candidates = [
        Path(__file__).parent / "data_storage" / "processed",
        Path(__file__).parent.parent / "data_processing" / "data_storage" / "processed",
    ]
    for p in candidates:
        if (p / "national_timeseries.parquet").exists():
            return p
    raise FileNotFoundError("Run pipeline.py first.")


def load_data():
    d = _find_data_dir()
    nat = pd.read_parquet(d / "national_timeseries.parquet")
    metro = pd.read_parquet(d / "metro_panel.parquet")
    nat["date"] = pd.to_datetime(nat["date"])
    metro["date"] = pd.to_datetime(metro["date"])
    return nat, metro


national_df, metro_df = load_data()

_top_metros = sorted(
    metro_df[metro_df["RegionName"] != "United States"]["RegionName"]
    .dropna().unique().tolist()
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

NAVY     = "#0F2040"
BLUE     = "#1A6FBF"
SLATE    = "#4A6080"
TEAL     = "#0D8C6A"
AMBER    = "#D4820A"
RED      = "#B83232"
OFFWHITE = "#F8F9FB"
CARD_BG  = "#FFFFFF"
BORDER   = "#E2E6ED"
MUTED    = "#8A95A3"
TEXT     = "#1C2B3A"

CHART_COLORS = [BLUE, TEAL, AMBER, RED, NAVY, SLATE,
                "#7B4FAF", "#C4621D", "#1A9490", "#6B7A8D"]


_AXIS = dict(gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False)


def _fig_base(height=420, margin=None):
    m = margin or dict(l=56, r=24, t=48, b=44)
    return dict(
        font=dict(family="'Inter','Helvetica Neue',Arial,sans-serif", size=12, color=TEXT),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=OFFWHITE,
        height=height,
        margin=m,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)", bordercolor=BORDER, borderwidth=1,
            font=dict(size=11),
        ),
        colorway=CHART_COLORS,
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font_size=12),
    )


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body,.bk,.bk * { font-family:'Inter','Helvetica Neue',Arial,sans-serif !important; }
.bk-tabs-header .bk-tab {
  font-size:13px !important; font-weight:500 !important; color:#8A95A3 !important;
  padding:10px 18px !important; border:none !important; background:transparent !important;
  border-bottom:3px solid transparent !important; border-radius:0 !important;
}
.bk-tabs-header .bk-tab:hover { color:#1A6FBF !important; }
.bk-tabs-header .bk-tab.bk-active {
  color:#0F2040 !important; font-weight:600 !important;
  border-bottom:3px solid #1A6FBF !important; background:transparent !important;
}
.bk-tabs-header { border-bottom:1px solid #E2E6ED !important; background:white !important; }
.bk-input { border:1px solid #E2E6ED !important; border-radius:6px !important; font-size:13px !important; }
.noUi-connect { background:#1A6FBF !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-thumb { background:#C5CDD8; border-radius:3px; }
</style>
"""


def _section_header(title, subtitle):
    return pn.pane.HTML(
        f'<div style="padding:4px 0 20px 0;border-bottom:1px solid {BORDER};margin-bottom:20px;">'
        f'<h2 style="margin:0 0 4px 0;font-size:20px;font-weight:700;color:{NAVY};">{title}</h2>'
        f'<p style="margin:0;font-size:13px;color:{MUTED};">{subtitle}</p></div>',
        sizing_mode="stretch_width",
    )


def _sidebar(widgets, width=260):
    return pn.Column(*widgets, width=width, margin=(0, 24, 0, 0))


# ---------------------------------------------------------------------------
# Tab 1 — National Overview
# ---------------------------------------------------------------------------

def _kpi_row(nat):
    def _v(col):
        s = nat.dropna(subset=[col]).sort_values("date")
        return s.iloc[-1][col] if not s.empty else float("nan")

    def _yoy(col):
        s = nat.dropna(subset=[col]).sort_values("date")
        if len(s) < 14:
            return float("nan")
        return (s.iloc[-1][col] - s.iloc[-13][col]) / s.iloc[-13][col] * 100

    zhvi = _v("zhvi"); zhvi_y = _yoy("zhvi")
    zori = _v("zori"); zori_y = _yoy("zori")
    mort = _v("MORTGAGE30US")
    fed  = _v("FEDFUNDS")

    def _delta(val, pos_good=True):
        if np.isnan(val):
            return ""
        c = TEAL if (val > 0) == pos_good else RED
        a = "▲" if val > 0 else "▼"
        return f'<span style="font-size:12px;color:{c};font-weight:500;">{a} {abs(val):.1f}%</span>'

    def _kpi(label, value, delta=""):
        return (
            f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;'
            f'padding:16px 20px;flex:1;min-width:130px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
            f'<div style="font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:8px;">{label}</div>'
            f'<div style="font-size:24px;font-weight:700;color:{NAVY};line-height:1;">{value}</div>'
            f'<div style="margin-top:6px;min-height:16px;">{delta}</div></div>'
        )

    cards = "".join([
        _kpi("Median Home Value",  f"${zhvi/1000:.0f}K" if not np.isnan(zhvi) else "—", _delta(zhvi_y, False)),
        _kpi("Median Rent",        f"${zori:,.0f}/mo"   if not np.isnan(zori) else "—", _delta(zori_y, False)),
        _kpi("30-Yr Mortgage",     f"{mort:.2f}%"       if not np.isnan(mort) else "—"),
        _kpi("Fed Funds Rate",     f"{fed:.2f}%"        if not np.isnan(fed)  else "—"),
    ])
    return pn.pane.HTML(
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px;">{cards}</div>',
        sizing_mode="stretch_width",
    )


def build_national_overview(nat):
    nat = nat.sort_values("date")

    fig1 = go.Figure()
    zhvi_d = nat.dropna(subset=["zhvi"])
    fig1.add_trace(go.Scatter(x=zhvi_d["date"], y=zhvi_d["zhvi"],
        name="Home Value (ZHVI)", line=dict(color=NAVY, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>"))
    if "MSPUS" in nat.columns:
        ms = nat.dropna(subset=["MSPUS"])
        fig1.add_trace(go.Scatter(x=ms["date"], y=ms["MSPUS"],
            name="Median Sale Price", line=dict(color=BLUE, width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>"))
    if "MORTGAGE30US" in nat.columns:
        mo = nat.dropna(subset=["MORTGAGE30US"])
        fig1.add_trace(go.Scatter(x=mo["date"], y=mo["MORTGAGE30US"],
            name="30-Yr Mortgage Rate", yaxis="y2", line=dict(color=RED, width=1.8),
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.2f}%<extra></extra>"))
    fig1.update_layout(**_fig_base(400),
        title=dict(text="Home Values & Mortgage Rates", font=dict(size=14, color=TEXT), x=0),
        yaxis2=dict(title="Mortgage Rate", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False, linecolor=BORDER))
    fig1.update_layout(yaxis=dict(title="Home Value (USD)", tickprefix="$", tickformat=",.0f",
                                   gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False))
    fig1.update_xaxes(**_AXIS)

    fig2 = go.Figure()
    if "CUSR0000SAH1" in nat.columns:
        cpi = nat.dropna(subset=["CUSR0000SAH1"]).copy()
        br = cpi[cpi["date"] >= "2015-01-01"]
        if not br.empty:
            base = br.iloc[0]["CUSR0000SAH1"]
            cpi["idx"] = cpi["CUSR0000SAH1"] / base * 100
            fig2.add_trace(go.Scatter(
                x=cpi[cpi["date"] >= "2015-01-01"]["date"],
                y=cpi[cpi["date"] >= "2015-01-01"]["idx"],
                name="CPI Shelter", line=dict(color=AMBER, width=2.5),
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}<extra></extra>"))
    if "zori" in nat.columns:
        zd = nat.dropna(subset=["zori"]).copy()
        if not zd.empty:
            zd["idx"] = zd["zori"] / zd.iloc[0]["zori"] * 100
            fig2.add_trace(go.Scatter(x=zd["date"], y=zd["idx"],
                name="Zillow Rent Index", line=dict(color=TEAL, width=2.5),
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}<extra></extra>"))
    fig2.update_layout(**_fig_base(360),
        title=dict(text="CPI Shelter vs. Zillow Rent Index (Jan 2015 = 100)",
                   font=dict(size=14, color=TEXT), x=0))
    fig2.update_xaxes(**_AXIS)
    fig2.update_yaxes(title="Index", gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False)

    return pn.Column(
        pn.pane.HTML(GLOBAL_CSS, width=0, height=0, margin=0),
        _section_header("National Overview",
                        "Macro trends in home values, mortgage rates, and rent inflation"),
        _kpi_row(nat),
        pn.pane.Plotly(fig1, sizing_mode="stretch_width"),
        pn.pane.HTML('<div style="height:12px"></div>'),
        pn.pane.Plotly(fig2, sizing_mode="stretch_width"),
        sizing_mode="stretch_width", margin=(20, 24),
    )


# ---------------------------------------------------------------------------
# Tab 2 — Metro Drill-Down
# ---------------------------------------------------------------------------

metro_select = pn.widgets.MultiSelect(
    name="metros", value=["New York, NY", "Los Angeles, CA", "Chicago, IL", "Austin, TX", "Miami, FL"],
    options=_top_metros, size=9, width=240)
year_range = pn.widgets.RangeSlider(
    name="year range", start=2000, end=2026, value=(2015, 2026), step=1, width=240)
metric_select = pn.widgets.Select(
    name="metric",
    options={"Home Value (ZHVI)": "zhvi", "Rent (ZORI)": "zori",
             "Home Value YoY %": "zhvi_yoy_pct", "Rent YoY %": "zori_yoy_pct"},
    value="zhvi", width=240)

_ML = {"zhvi": "Median Home Value ($)", "zori": "Median Rent ($/mo)",
       "zhvi_yoy_pct": "Home Value YoY (%)", "zori_yoy_pct": "Rent YoY (%)"}


@pn.depends(metro_select, year_range, metric_select)
def _metro_charts(metros, yr, metric):
    if not metros:
        return pn.pane.HTML(f"<p style='color:{MUTED};padding:20px;'>Select at least one metro.</p>")
    df = metro_df[metro_df["RegionName"].isin(metros) &
                  metro_df["date"].dt.year.between(yr[0], yr[1])].copy()
    if metric not in df.columns or df[metric].dropna().empty:
        return pn.pane.HTML(f"<p style='color:{MUTED};padding:20px;'>No {metric} data.</p>")

    is_pct = "pct" in metric
    label = _ML.get(metric, metric)

    fig = go.Figure()
    for i, m in enumerate(metros):
        mdf = df[df["RegionName"] == m].sort_values("date")
        if mdf[metric].dropna().empty:
            continue
        fig.add_trace(go.Scatter(x=mdf["date"], y=mdf[metric], name=m,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{m}</b><br>%{{x|%b %Y}}<br>{'%{y:.1f}%' if is_pct else '$%{y:,.0f}'}<extra></extra>"))
    fig.update_layout(**_fig_base(400),
        title=dict(text=f"{label} — Selected Metros", font=dict(size=14, color=TEXT), x=0))
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(title=label, tickprefix="" if is_pct else "$",
                     ticksuffix="%" if is_pct else "", tickformat=".1f" if is_pct else ",.0f",
                     gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False)

    latest = [{"Metro": m, "val": df[df["RegionName"] == m].dropna(subset=[metric]).sort_values("date").iloc[-1][metric]}
              for m in metros if not df[df["RegionName"] == m].dropna(subset=[metric]).empty]
    out = [pn.pane.Plotly(fig, sizing_mode="stretch_width")]
    if latest:
        ldf = pd.DataFrame(latest).sort_values("val")
        fig_b = go.Figure(go.Bar(
            x=ldf["val"], y=ldf["Metro"], orientation="h",
            marker=dict(color=ldf["val"].tolist(),
                        colorscale=[[0, "#B8D0E8"], [1, NAVY]], showscale=False),
            hovertemplate=f"<b>%{{y}}</b><br>{'%{x:.1f}%' if is_pct else '$%{x:,.0f}'}<extra></extra>"))
        fig_b.update_layout(**_fig_base(max(220, len(metros)*44+60), margin=dict(l=160, r=24, t=48, b=44)),
            title=dict(text=f"Latest {label}", font=dict(size=14, color=TEXT), x=0))
        fig_b.update_xaxes(title=label, tickprefix="" if is_pct else "$",
                            ticksuffix="%" if is_pct else "", tickformat=".1f" if is_pct else ",.0f",
                            gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False)
        fig_b.update_yaxes(linecolor=BORDER, zeroline=False)
        out += [pn.pane.HTML('<div style="height:12px"></div>'),
                pn.pane.Plotly(fig_b, sizing_mode="stretch_width")]
    return pn.Column(*out, sizing_mode="stretch_width")


def build_metro_tab():
    lbl = pn.pane.HTML(
        f'<div style="font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:16px;">Filters</div>', sizing_mode="stretch_width")
    return pn.Column(
        _section_header("Metro Drill-Down", "Compare home values and rents across up to 50 metro areas"),
        pn.Row(_sidebar([lbl, metro_select,
                         pn.pane.HTML('<div style="height:16px"></div>'), year_range,
                         pn.pane.HTML('<div style="height:16px"></div>'), metric_select]),
               pn.Column(_metro_charts, sizing_mode="stretch_width"), sizing_mode="stretch_width"),
        sizing_mode="stretch_width", margin=(20, 24))


# ---------------------------------------------------------------------------
# Tab 3 — Home Value Growth (Panel DiscreteSlider — no Plotly animation)
# ---------------------------------------------------------------------------

_race_raw = metro_df[
    (metro_df["RegionName"] != "United States") & (metro_df["date"].dt.month == 12)
].copy()
_race_raw["year"] = _race_raw["date"].dt.year

_top15 = (
    _race_raw.groupby("RegionName")["zhvi"].mean()
    .nlargest(15).index.tolist()
)

_race_data = {}
for _yr, _grp in _race_raw[_race_raw["RegionName"].isin(_top15)].groupby("year"):
    _snap = _grp.dropna(subset=["zhvi"]).sort_values("zhvi", ascending=True)
    if not _snap.empty:
        _race_data[int(_yr)] = _snap

_race_years = sorted(_race_data.keys())

race_slider = pn.widgets.DiscreteSlider(
    name="year", options=_race_years, value=_race_years[-1], width=500)


@pn.depends(race_slider)
def _race_chart(year):
    snap = _race_data.get(year)
    if snap is None:
        return pn.pane.HTML(f"<p style='color:{MUTED};padding:20px;'>No data for {year}.</p>")

    max_val = max(d["zhvi"].max() for d in _race_data.values()) * 1.08
    vals = snap["zhvi"].values
    norm = (vals - vals.min()) / max(vals.max() - vals.min(), 1)
    colors = [f"rgb({int(15+n*10)},{int(80+n*31)},{int(150+n*45)})" for n in norm]

    fig = go.Figure(go.Bar(
        x=snap["zhvi"], y=snap["RegionName"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"${v/1000:.0f}K" for v in snap["zhvi"]],
        textposition="outside", textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(**_fig_base(480, margin=dict(l=160, r=80, t=48, b=44)),
        title=dict(text=f"Top 15 Most Expensive Metros — {year}",
                   font=dict(size=14, color=TEXT), x=0),
        bargap=0.28)
    fig.update_xaxes(title="Median Home Value", tickprefix="$", tickformat=",.0f",
                     range=[0, max_val], gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False)
    fig.update_yaxes(linecolor=BORDER, tickfont=dict(size=11), zeroline=False)
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def build_animated_tab():
    extras = []
    if "zhvi_to_income_ratio" in metro_df.columns:
        rdf = metro_df[metro_df["RegionName"].isin(_top15[:8])].dropna(subset=["zhvi_to_income_ratio"])
        if not rdf.empty:
            fig2 = go.Figure()
            for i, m in enumerate(_top15[:8]):
                mdf = rdf[rdf["RegionName"] == m].sort_values("date")
                if not mdf.empty:
                    fig2.add_trace(go.Scatter(
                        x=mdf["date"], y=mdf["zhvi_to_income_ratio"], name=m,
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.8),
                        hovertemplate=f"<b>{m}</b><br>%{{x|%Y}}<br>%{{y:.1f}}x income<extra></extra>"))
            fig2.update_layout(**_fig_base(340),
                title=dict(text="Home Price-to-Income Ratio — Top 8 Metros",
                           font=dict(size=14, color=TEXT), x=0))
            fig2.update_xaxes(**_AXIS)
            fig2.update_yaxes(title="Years of Income to Buy", gridcolor="#E8ECF0",
                              linecolor=BORDER, zeroline=False)
            extras = [pn.pane.HTML('<div style="height:16px"></div>'),
                      pn.pane.Plotly(fig2, sizing_mode="stretch_width")]

    return pn.Column(
        _section_header("Top 15 Most Expensive Metros — Home Value Growth",
                        "Drag the year slider to see how rankings and prices shifted from 2009 to 2023"),
        pn.pane.HTML(
            f'<div style="font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;'
            f'letter-spacing:0.06em;margin-bottom:8px;">Year</div>'),
        race_slider,
        pn.pane.HTML('<div style="height:4px"></div>'),
        _race_chart,
        *extras,
        sizing_mode="stretch_width", margin=(20, 24))


# ---------------------------------------------------------------------------
# Tab 4 — Rent vs. Buy Calculator
# ---------------------------------------------------------------------------

calc_metro = pn.widgets.Select(
    name="metro area",
    options=_top_metros,
    value="Boston, MA" if "Boston, MA" in _top_metros else _top_metros[0],
    width=240)
calc_income = pn.widgets.IntSlider(
    name="annual income ($)", start=30000, end=400000, step=5000, value=85000, width=240)
calc_down   = pn.widgets.IntSlider(
    name="down payment (%)", start=3, end=30, step=1, value=20, width=240)
calc_rate   = pn.widgets.FloatSlider(
    name="mortgage rate (%)", start=3.0, end=10.0, step=0.1, value=6.8, width=240)


def _monthly_pmt(price, down_pct, rate_pct, years=30):
    loan = price * (1 - down_pct / 100)
    r = rate_pct / 100 / 12
    n = years * 12
    return loan * r * (1 + r) ** n / ((1 + r) ** n - 1) if r else loan / n


@pn.depends(calc_metro, calc_income, calc_down, calc_rate)
def _calc_output(metro, income, down_pct, rate):
    mdf = metro_df[metro_df["RegionName"] == metro].dropna(subset=["zhvi"]).sort_values("date")
    if mdf.empty:
        return pn.pane.HTML(f"<p style='color:{MUTED};padding:20px;'>No home value data.</p>")

    zhvi = mdf.iloc[-1]["zhvi"]
    pmt = _monthly_pmt(zhvi, down_pct, rate)
    mi = income / 12
    buy_pct = pmt / mi * 100

    rdf = metro_df[metro_df["RegionName"] == metro].dropna(subset=["zori"]).sort_values("date")
    zori = rdf.iloc[-1]["zori"] if not rdf.empty else None
    rent_pct = (zori / mi * 100) if zori else None

    if rent_pct:
        if buy_pct < rent_pct:
            rb, rbo = f"{TEAL}15", TEAL
            rec = f"Buying looks more affordable — mortgage is <b>{buy_pct:.0f}%</b> of income vs renting at <b>{rent_pct:.0f}%</b>."
        elif buy_pct > rent_pct + 10:
            rb, rbo = f"{RED}15", RED
            rec = f"Renting is significantly cheaper — buying costs <b>{buy_pct:.0f}%</b> vs renting at <b>{rent_pct:.0f}%</b>."
        else:
            rb, rbo = f"{AMBER}18", AMBER
            rec = f"It's close — buying at <b>{buy_pct:.0f}%</b> of income, renting at <b>{rent_pct:.0f}%</b>."
    else:
        rb, rbo = "#E2E6ED15", BORDER
        rec = f"Monthly mortgage is <b>{buy_pct:.0f}%</b> of your income."

    kpis = "".join([
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;'
        f'padding:14px 18px;flex:1;min-width:120px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
        f'<div style="font-size:10px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:6px;">{lbl}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{NAVY};">{val}</div></div>'
        for lbl, val in [
            ("Home Value",      f"${zhvi/1000:.0f}K"),
            ("Monthly Payment", f"${pmt:,.0f}"),
            ("Down Payment",    f"${zhvi*down_pct/100/1000:.0f}K"),
            ("Current Rent",    f"${zori:,.0f}/mo" if zori else "—"),
        ]
    ])

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=round(buy_pct, 1),
        number={"suffix": "%", "valueformat": ".1f",
                "font": {"size": 28, "color": NAVY, "family": "Inter, Arial"}},
        title={"text": "Mortgage as % of Income",
               "font": {"size": 12, "color": SLATE, "family": "Inter, Arial"}},
        gauge={"axis": {"range": [0, 80], "ticksuffix": "%", "nticks": 9},
               "bar": {"color": NAVY, "thickness": 0.28}, "bgcolor": OFFWHITE, "borderwidth": 0,
               "steps": [{"range": [0,  28], "color": "#C8EDD8"},
                          {"range": [28, 36], "color": "#FDE8B0"},
                          {"range": [36, 80], "color": "#FAD0D0"}],
               "threshold": {"line": {"color": RED, "width": 2}, "value": 36}}))
    fig_g.update_layout(height=240, margin=dict(l=20, r=20, t=56, b=10),
                        paper_bgcolor=CARD_BG, font_family="Inter, Arial")

    hist = mdf.copy()
    hist["pmt_pct"] = hist["zhvi"].apply(lambda v: _monthly_pmt(v, down_pct, rate)) / mi * 100
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=hist["date"], y=hist["pmt_pct"],
        name="Buy cost % income", fill="tozeroy",
        fillcolor="rgba(26,111,191,0.08)", line=dict(color=BLUE, width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}%<extra></extra>"))
    fig_h.add_hline(y=28, line_dash="dash", line_color=TEAL, line_width=1.5,
                    annotation_text="Affordable (28%)", annotation_position="top right",
                    annotation_font=dict(size=11, color=TEAL))
    fig_h.add_hline(y=36, line_dash="dash", line_color=RED, line_width=1.5,
                    annotation_text="Stressed (36%)", annotation_position="top right",
                    annotation_font=dict(size=11, color=RED))
    if rent_pct:
        fig_h.add_hline(y=rent_pct, line_dash="dot", line_color=AMBER, line_width=1.5,
                        annotation_text=f"Rent ({rent_pct:.0f}%)", annotation_position="bottom right",
                        annotation_font=dict(size=11, color=AMBER))
    fig_h.update_layout(**_fig_base(320),
        title=dict(text=f"Historical Buy Affordability — {metro}",
                   font=dict(size=14, color=TEXT), x=0),
        showlegend=False)
    fig_h.update_xaxes(**_AXIS)
    fig_h.update_yaxes(title="Monthly Cost as % of Income", ticksuffix="%",
                       gridcolor="#E8ECF0", linecolor=BORDER, zeroline=False)

    return pn.Column(
        pn.pane.HTML(
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">{kpis}</div>'
            f'<div style="background:#{rb};border-left:3px solid {rbo};'
            f'border-radius:0 8px 8px 0;padding:10px 14px;font-size:13px;'
            f'color:{TEXT};margin-bottom:16px;">{rec}</div>',
            sizing_mode="stretch_width"),
        pn.Row(pn.pane.Plotly(fig_g, width=280, height=240),
               pn.pane.Plotly(fig_h, sizing_mode="stretch_width"),
               sizing_mode="stretch_width"),
        sizing_mode="stretch_width")


def build_calculator_tab():
    legend = pn.pane.HTML(
        f'<div style="font-size:12px;color:{MUTED};line-height:1.8;margin-top:16px;">'
        f'<span style="color:{TEAL};font-weight:600;">■</span> &lt;28% — affordable<br>'
        f'<span style="color:{AMBER};font-weight:600;">■</span> 28–36% — caution<br>'
        f'<span style="color:{RED};font-weight:600;">■</span> &gt;36% — stressed</div>')
    lbl = pn.pane.HTML(
        f'<div style="font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:0.06em;margin-bottom:16px;">Your Inputs</div>',
        sizing_mode="stretch_width")
    return pn.Column(
        _section_header("Rent vs. Buy Calculator",
                        "Estimate affordability based on your income and local market"),
        pn.Row(_sidebar([lbl, calc_metro,
                         pn.pane.HTML('<div style="height:12px"></div>'), calc_income,
                         pn.pane.HTML('<div style="height:4px"></div>'), calc_down,
                         pn.pane.HTML('<div style="height:4px"></div>'), calc_rate, legend]),
               pn.Column(_calc_output, sizing_mode="stretch_width"), sizing_mode="stretch_width"),
        sizing_mode="stretch_width", margin=(20, 24))


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def build_dashboard():
    tabs = pn.Tabs(
        ("National Overview", build_national_overview(national_df)),
        ("Metro Drill-Down",  build_metro_tab()),
        ("Home Value Growth", build_animated_tab()),
        ("Rent vs. Buy",      build_calculator_tab()),
        dynamic=True,
    )
    title_bar = pn.pane.HTML(
        f'<div style="background:{NAVY};padding:18px 28px 16px 28px;">'
        f'<div style="display:flex;align-items:baseline;gap:12px;">'
        f'<span style="font-size:22px;font-weight:700;color:#FFFFFF;letter-spacing:-0.3px;">'
        f'Housing Affordability</span>'
        f'<span style="font-size:13px;color:rgba(255,255,255,0.5);">'
        f'FRED · Zillow · Census ACS</span></div></div>',
        sizing_mode="stretch_width")
    tmpl = pn.template.BootstrapTemplate(
        title="Housing Affordability Dashboard",
        header_background=NAVY,
    )
    tmpl.main.append(pn.Column(title_bar, tabs, sizing_mode="stretch_width"))
    return tmpl


dashboard = build_dashboard()
dashboard.servable()
