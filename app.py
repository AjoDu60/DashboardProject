"""MacroLens Asia | Eight Asian economies and major economic shocks, 1990-2025.

Run: streamlit run app.py
Data: World Bank World Development Indicators; no synthetic observations are bundled.
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START_YEAR, END_YEAR = 1990, 2025
COUNTRIES: Dict[str, str] = {
    "Indonesia": "IDN", "Japan": "JPN", "Malaysia": "MYS", "Thailand": "THA",
    "Philippines": "PHL", "South Korea": "KOR", "Singapore": "SGP", "Hong Kong SAR": "HKG",
}
CORE_CRISIS = ("Indonesia", "Malaysia", "Thailand", "Philippines", "South Korea")
# Each crisis uses a pre-shock average, a shock window, and a post-shock average.
# Recovery compares constant-price GDP with its own pre-shock reference year.
CRISES = {
    "asian": {
        "short": "1997–98 Asian Crisis",
        "title": "Asian Financial Crisis",
        "baseline": 1996, "pre": (1994, 1996), "shock": (1997, 1999),
        "post": (2000, 2002), "view": (1994, 2005), "recovery_end": 2005,
        "landmark": 1998,
        "context": (
            "On 2 July 1997, Thailand allowed the baht to float after sustained market "
            "pressure. Currency and banking strains then spread across parts of East Asia, "
            "including Indonesia and South Korea. The downturn became especially visible "
            "in many economies' 1998 output data (Federal Reserve History, 2013; IMF, 1999)."
        ),
        "mechanism": "Currency pressure, external financing, banking-sector stress and real output",
        "question": "How did exchange-rate movements and GDP contractions differ across the eight economies?",
        "source": "Federal Reserve History (2013) — Asian Financial Crisis",
        "background_sources": (
            "Federal Reserve History (2013) — Asian Financial Crisis",
            "IMF (1999) — The Asia Crisis: Causes, Policy Responses, and Outcomes",
        ),
        "milestones": (
            ("Jul 1997", "Thailand floats the baht; regional currency pressure follows."),
            ("1998", "Several affected economies record sharp output contractions."),
            ("1999 onward", "Stabilization and recovery proceed at different speeds."),
        ),
    },
    "global": {
        "short": "2008–09 Global Crisis",
        "title": "Global Financial Crisis",
        "baseline": 2007, "pre": (2005, 2007), "shock": (2008, 2010),
        "post": (2011, 2013), "view": (2005, 2015), "recovery_end": 2015,
        "landmark": 2009,
        "context": (
            "Financial stress linked to the US housing market escalated in September "
            "2008, disrupting international credit and trade. For export-oriented Asian "
            "economies, weaker external demand contributed to sharp production and "
            "trade declines into 2009 (IMF, 2009; World Bank, 2010)."
        ),
        "mechanism": "Global demand, trade, cross-border finance and manufacturing output",
        "question": "Which Asian economies experienced the steepest 2009 slowdown, and how did their output recover?",
        "source": "IMF (2009) — World Economic Outlook: Crisis and Recovery",
        "background_sources": (
            "IMF (2009) — World Economic Outlook: Crisis and Recovery",
            "World Bank (2010) — Economic impact of the 2008 financial crisis",
        ),
        "milestones": (
            ("2007", "US mortgage-related financial strains intensify."),
            ("Sep 2008", "Global financial stress escalates after Lehman Brothers' failure."),
            ("2009", "Global demand and trade contract; Asian exporters face severe pressure."),
        ),
    },
    "covid": {
        "short": "2020 COVID-19 Shock",
        "title": "COVID-19 Economic Shock",
        "baseline": 2019, "pre": (2017, 2019), "shock": (2020, 2021),
        "post": (2022, 2024), "view": (2017, 2025), "recovery_end": 2025,
        "landmark": 2020,
        "context": (
            "The COVID-19 pandemic disrupted business activity, travel and supply chains "
            "during 2020. Public-health measures and falling demand weighed particularly "
            "on contact-intensive sectors. Recovery was uneven as restrictions and "
            "international travel changed over time (IMF, 2020; World Bank, 2023)."
        ),
        "mechanism": "Mobility restrictions, domestic services, trade, employment and real GDP recovery",
        "question": "Which economies contracted in 2020, and when did real GDP regain its 2019 level?",
        "source": "IMF (2020) — World Economic Outlook: The Great Lockdown",
        "background_sources": (
            "IMF (2020) — World Economic Outlook: The Great Lockdown",
            "World Bank (2023) — COVID-19 recovery in East Asia and Pacific",
        ),
        "milestones": (
            ("2020", "The pandemic and containment measures disrupt economic activity."),
            ("2021", "Economic activity rebounds in many places, unevenly."),
            ("2022 onward", "Recoveries diverge as travel and supply conditions change."),
        ),
    },
}
# "All Time" is an overview, not a fourth crisis.
ALL_TIME_KEY = "all"
FULL_PERIOD_NAME = "Full period (1990–2025)"
CRISIS_NAMES = {key: value["short"] for key, value in CRISES.items()}
CRISIS_NAMES[ALL_TIME_KEY] = "All Time"
PERIOD_PRESETS = {
    "Full period (1990–2025)": (1990, 2025),
    "Asian Financial Crisis (1994–2005)": CRISES["asian"]["view"],
    "Global Financial Crisis (2005–2015)": CRISES["global"]["view"],
    "COVID-19 shock (2017–2025)": CRISES["covid"]["view"],
    "Recent years (2020–2025)": (2020, 2025),
    "Custom range": None,
}
PRESET_FOR_CRISIS = {
    "asian": "Asian Financial Crisis (1994–2005)",
    "global": "Global Financial Crisis (2005–2015)",
    "covid": "COVID-19 shock (2017–2025)",
}

# The first three series are necessary for the baseline dashboard. The last two
# enhance the crisis analysis, and can fail without hiding the other indicators.
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "SL.UEM.TOTL.ZS": "unemployment",
    "FP.CPI.TOTL.ZG": "inflation",
    "NY.GDP.MKTP.KD": "real_gdp",
    "PA.NUS.FCRF": "fx_rate",
}
CORE_FIELDS = ("gdp_growth", "unemployment", "inflation")
SOURCE_FIELDS = tuple(INDICATORS.values())
DATA_COLUMNS = ["country", "country_code", "year", *SOURCE_FIELDS, "fx_depreciation", "unemployment_change",
                "growth_from_levels", "growth_difference_pp"]
COLORS = {
    "Indonesia": "#0072B2", "Japan": "#D55E00", "Malaysia": "#009E73",
    "Thailand": "#CC79A7", "Philippines": "#A97400", "South Korea": "#7055AA",
    "Singapore": "#3797BB", "Hong Kong SAR": "#9B623B",
}
METRIC_LABELS = {
    "gdp_growth": "Real GDP growth (%)",
    "unemployment": "Unemployment rate (%)",
    "inflation": "Consumer price inflation (%)",
    "fx_depreciation": "Exchange-rate change (% YoY)",
}
SOURCES = {
    "World Bank — GDP growth (annual %)": "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG",
    "World Bank — Unemployment (ILO modeled estimate)": "https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS",
    "World Bank — Consumer price inflation": "https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG",
    "World Bank — GDP (constant-price US$)": "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD",
    "World Bank — Official exchange rate (LCU per US$)": "https://data.worldbank.org/indicator/PA.NUS.FCRF",
    "IMF (1999) — The Asia Crisis: Causes, Policy Responses, and Outcomes": "https://www.imf.org/external/pubs/ft/wp/1999/wp99138.pdf",
    "Federal Reserve History (2013) — Asian Financial Crisis": "https://www.federalreservehistory.org/essays/asian-financial-crisis",
    "IMF (2009) — World Economic Outlook: Crisis and Recovery": "https://www.imf.org/en/publications/weo/issues/2016/12/31/world-economic-outlook-april-2009-crisis-and-recovery-22575",
    "World Bank (2010) — Economic impact of the 2008 financial crisis": "https://blogs.worldbank.org/en/opendata/data-show-mdg-target-challenges008-financial-crisis",
    "World Bank (1998) — Indonesia in Crisis": "https://timeline.worldbank.org/en/timeline/eventdetail/2381",
    "World Bank (2023) — COVID-19 recovery in East Asia and Pacific": "https://www.worldbank.org/en/results/2023/12/08/putting-east-asia-and-the-pacific-on-a-path-to-recovery-from-the-covid-19-crisis",
    "IMF (2020) — World Economic Outlook: The Great Lockdown": "https://www.imf.org/en/Publications/WEO/Issues/2020/04/14/weo-april-2020",
    "Federal Reserve Bank of St. Louis (2013) — Output and Unemployment": "https://www.stlouisfed.org/publications/regional-economist/october-2013/output-and-unemployment-how-do-they-relate-today",
    "World Bank — WDI GDP growth definition": "https://databank.worldbank.org/metadataglossary/world-development-indicators/series/NY.GDP.MKTP.KD.ZG",
    "World Bank — ILO unemployment definition": "https://databank.worldbank.org/metadataglossary/health-nutrition-and-population-statistics/series/SL.UEM.TOTL.ZS",
}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_indicator(series: str) -> list[dict]:
    """Fetch/cache each WDI series separately, so one timeout does not waste earlier work."""
    url = "https://api.worldbank.org/v2/country/" + ";".join(COUNTRIES.values()) + f"/indicator/{series}"
    retry = Retry(total=2, connect=2, read=2, status=2, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        try:
            result = session.get(url, params={"format": "json", "date": f"{START_YEAR}:{END_YEAR}",
                                              "per_page": 2000, "source": 2},
                                 timeout=(10, 35), headers={"User-Agent": "MacroLensAsia/4.0"})
            result.raise_for_status()
            payload = result.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"World Bank download failed for {series}: {exc}") from exc
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise RuntimeError(f"World Bank did not return a data table for {series}.")
    if payload[0].get("pages", 1) > 1:
        raise RuntimeError(f"The API response for {series} was paginated; refusing partial results.")
    field = INDICATORS[series]
    return [
        {"country_code": row["countryiso3code"], "year": int(row["date"]),
         "field": field, "value": row.get("value")}
        for row in payload[1]
        if isinstance(row, dict) and row.get("countryiso3code") in COUNTRIES.values()
    ]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep WDI source values unchanged; derive diagnostics before filtering.

    Δ unemployment is in percentage points. FX change is the annual percent
    movement in LCU per USD (positive = depreciation). Real-GDP growth is a
    *validation* series only and never replaces reported WDI GDP growth.
    """
    frame = df.copy()
    for field in SOURCE_FIELDS:
        if field not in frame.columns:
            frame[field] = np.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if frame.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate economy-year identifiers; cannot safely calculate changes.")
    frame = frame.sort_values(["country_code", "year"]).reset_index(drop=True)
    frame["fx_depreciation"] = np.nan
    frame["unemployment_change"] = np.nan
    frame["growth_from_levels"] = np.nan
    frame["growth_difference_pp"] = np.nan
    for _, group in frame.groupby("country_code", sort=False):
        idx = group.index
        consecutive = group["year"].diff().eq(1)
        previous_fx = group["fx_rate"].shift(1)
        fx_change = 100 * (group["fx_rate"] / previous_fx - 1)
        fx_change = fx_change.where(consecutive & previous_fx.gt(0) & group["fx_rate"].gt(0))
        frame.loc[idx, "fx_depreciation"] = fx_change.to_numpy()
        frame.loc[idx, "unemployment_change"] = group["unemployment"].diff().where(consecutive).to_numpy()
        prev_gdp = group["real_gdp"].shift(1)
        calculated = 100 * (group["real_gdp"] / prev_gdp - 1)
        calculated = calculated.where(consecutive & prev_gdp.gt(0) & group["real_gdp"].gt(0))
        frame.loc[idx, "growth_from_levels"] = calculated.to_numpy()
        frame.loc[idx, "growth_difference_pp"] = (group["gdp_growth"] - calculated).to_numpy()
    return frame[DATA_COLUMNS].sort_values(["country", "year"]).reset_index(drop=True)

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_all_data() -> tuple[pd.DataFrame, list[str]]:
    rows, optional_failed = [], []
    for series, field in INDICATORS.items():
        try:
            rows.extend(fetch_indicator(series))
        except RuntimeError:
            if field in CORE_FIELDS:
                raise
            optional_failed.append(field)
    if not rows:
        raise RuntimeError("The World Bank API returned no observations.")
    long = pd.DataFrame(rows)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    grid = pd.MultiIndex.from_product(
        [COUNTRIES.values(), range(START_YEAR, END_YEAR + 1)],
        names=["country_code", "year"],
    ).to_frame(index=False)
    wide = long.pivot_table(index=["country_code", "year"], columns="field",
                            values="value", aggfunc="first").reset_index()
    frame = grid.merge(wide, on=["country_code", "year"], how="left")
    frame["country"] = frame["country_code"].map({v: k for k, v in COUNTRIES.items()})
    return prepare_data(frame), optional_failed


def read_backup(uploaded_file) -> pd.DataFrame:
    """Load an export of THIS version; do not silently fill new columns with invented data."""
    raw = pd.read_csv(uploaded_file)
    necessary = ["country", "country_code", "year", *SOURCE_FIELDS]
    missing = [name for name in necessary if name not in raw.columns]
    if missing:
        raise ValueError("The CSV is from an older dashboard or has missing columns: " + ", ".join(missing))
    if raw.empty or raw[necessary[:3]].isna().any().any():
        raise ValueError("The CSV has missing economy or year identifiers.")
    raw["year"] = pd.to_numeric(raw["year"], errors="raise").astype(int)
    if raw["year"].lt(START_YEAR).any() or raw["year"].gt(END_YEAR).any():
        raise ValueError(f"The backup must use years {START_YEAR}-{END_YEAR}.")
    mapping = {v: k for k, v in COUNTRIES.items()}
    if not raw["country_code"].isin(mapping).all():
        raise ValueError("The backup includes an unexpected economy code.")
    if not raw["country"].eq(raw["country_code"].map(mapping)).all():
        raise ValueError("Some economy names do not match their ISO codes.")
    if raw.duplicated(["country_code", "year"]).any():
        raise ValueError("The backup includes duplicate economy-year records.")
    expected = len(COUNTRIES) * (END_YEAR - START_YEAR + 1)
    if len(raw) != expected:
        raise ValueError(f"The full backup should have {expected} economy-year rows; export a new CSV.")
    return prepare_data(raw[necessary])


def okun_statistics(df: pd.DataFrame) -> dict:
    """Descriptive GDP-growth / Δ-unemployment comparisons, never causal.

    Pooled: raw annual country-years.
    Country adjusted: one-way within-country demeaning.
    Country + year adjusted: residualization on country and calendar-year
    dummies, then OLS of residual unemployment change on residual growth.
    No p-values or causal interpretation are offered for this small panel.
    """
    pairs = df[["country", "year", "gdp_growth", "unemployment_change"]].dropna().copy()
    result = dict(pooled_r=None, within_r=None, slope=None,
                  two_way_r=None, two_way_slope=None, two_way_n=0,
                  n=len(pairs), economies=int(pairs["country"].nunique()),
                  pairs=pairs, within_pairs=pd.DataFrame(),
                  two_way_pairs=pd.DataFrame(), country_table=pd.DataFrame())
    if len(pairs) < 4 or pairs["gdp_growth"].nunique() < 2 or pairs["unemployment_change"].nunique() < 2:
        return result
    raw_corr = pairs["gdp_growth"].corr(pairs["unemployment_change"])
    if pd.notna(raw_corr):
        result["pooled_r"] = float(raw_corr)
    pairs["growth_deviation"] = pairs["gdp_growth"] - pairs.groupby("country")["gdp_growth"].transform("mean")
    pairs["unemployment_deviation"] = (pairs["unemployment_change"] -
                                      pairs.groupby("country")["unemployment_change"].transform("mean"))
    result["pairs"] = pairs
    eligible = pairs[pairs.groupby("country")["country"].transform("size").ge(2)]
    result["within_pairs"] = eligible
    if len(eligible) >= 4:
        x = eligible["growth_deviation"].to_numpy(dtype=float)
        y = eligible["unemployment_deviation"].to_numpy(dtype=float)
        variance = float(x @ x)
        if variance > 1e-10 and float(y @ y) > 1e-10:
            within = eligible["growth_deviation"].corr(eligible["unemployment_deviation"])
            result["within_r"] = float(within) if pd.notna(within) else None
            result["slope"] = float((x @ y) / variance)

    # Two-way adjustment by least squares: FWL partialling-out country and year.
    # Restrict to estimable panels; a single economy or year does not identify it.
    if pairs["country"].nunique() >= 2 and pairs["year"].nunique() >= 3:
        dummy_country = pd.get_dummies(pairs["country"], prefix="c", drop_first=True, dtype=float)
        dummy_year = pd.get_dummies(pairs["year"].astype(str), prefix="y", drop_first=True, dtype=float)
        controls = np.column_stack([np.ones(len(pairs)), dummy_country.to_numpy(), dummy_year.to_numpy()])
        rank = int(np.linalg.matrix_rank(controls))
        dof = len(pairs) - rank - 1
        if dof >= 5:
            x = pairs["gdp_growth"].to_numpy(dtype=float)
            y = pairs["unemployment_change"].to_numpy(dtype=float)
            x_resid = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
            y_resid = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
            if float(x_resid @ x_resid) > 1e-8 and float(y_resid @ y_resid) > 1e-8:
                result["two_way_slope"] = float((x_resid @ y_resid) / (x_resid @ x_resid))
                result["two_way_r"] = float(np.corrcoef(x_resid, y_resid)[0, 1])
                result["two_way_n"] = len(pairs)
                partial = pairs.copy()
                partial["growth_residual"] = x_resid
                partial["unemployment_residual"] = y_resid
                result["two_way_pairs"] = partial

    country_rows = []
    for country, group in pairs.groupby("country"):
        x, y = group["gdp_growth"], group["unemployment_change"]
        if len(group) < 6 or x.nunique() < 2 or y.nunique() < 2:
            continue
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0))
        country_rows.append({"Economy": country, "Paired years": len(group),
                             "Correlation (r)": float(x.corr(y)),
                             "Slope (pp per 1pp GDP)": slope})
    result["country_table"] = pd.DataFrame(country_rows)
    return result

def safe_corr(df: pd.DataFrame) -> tuple[float | None, int]:
    """Backward-compatible interface: correlation using Δ unemployment, not its level."""
    result = okun_statistics(df)
    return result["pooled_r"], result["n"]


def check_data_quality(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Check coverage and numerical consistency without deleting source observations.

    Derived GDP growth should have been calculated BEFORE the year filter;
    otherwise the first year in any selected range is spuriously uncheckable.
    The 0.5pp threshold is a review flag, NOT a declaration of an error.
    """
    frame = df.copy()
    duplicate_pairs = int(frame.duplicated(["country_code", "year"]).sum())
    absent_ids = int(frame[["country_code", "year"]].isna().any(axis=1).sum())
    comparison = frame[["country", "year", "gdp_growth", "growth_from_levels", "growth_difference_pp"]].copy()
    comparison = comparison.rename(columns={"growth_from_levels": "Growth from GDP levels (%)",
                                            "growth_difference_pp": "Difference (pp)"})
    comparison = comparison.dropna(subset=["gdp_growth", "Growth from GDP levels (%)"])
    gaps = comparison["Difference (pp)"].abs()
    invalid = {
        "Unemployment outside 0-100%": int(frame["unemployment"].notna().sum() -
                                          frame["unemployment"].dropna().between(0, 100).sum()),
        "Nonpositive real GDP": int(frame["real_gdp"].dropna().le(0).sum()),
        "Nonpositive exchange rate": int(frame["fx_rate"].dropna().le(0).sum()),
        "Unusually large CPI inflation (>100% absolute)": int(frame["inflation"].dropna().abs().gt(100).sum()),
    }
    checks = {
        "missing_identifiers": absent_ids,
        "duplicates": duplicate_pairs,
        "observations": len(df),
        "growth_comparisons": len(comparison),
        "median_growth_difference_pp": float(gaps.median()) if len(gaps) else None,
        "large_growth_differences": int(gaps.gt(0.5).sum()),
        "complete_three": int(frame[list(CORE_FIELDS)].notna().all(axis=1).sum()),
        "plausibility_flags": invalid,
        "last_gdp_year": (int(frame.loc[frame["gdp_growth"].notna(), "year"].max())
                          if frame["gdp_growth"].notna().any() else None),
    }
    return checks, comparison


def indicator_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of country-years with each *source* indicator observed."""
    labels = {"gdp_growth": "GDP growth", "unemployment": "Unemployment",
              "inflation": "Inflation", "real_gdp": "Real GDP level", "fx_rate": "Official exchange rate"}
    if df.empty:
        return pd.DataFrame(columns=["Economy", "Years selected", *labels.values()])
    result = df.groupby("country", sort=True).agg(**{
        labels[field]: (field, lambda series: 100 * series.notna().mean()) for field in SOURCE_FIELDS
    })
    result.insert(0, "Years selected", df.groupby("country").size())
    return result.reset_index().rename(columns={"country": "Economy"})

def format_pct(value, digits=2):
    return "N/A" if pd.isna(value) else f"{value:,.{digits}f}%"


def year_span(bounds: tuple[int, int]) -> str:
    return f"{bounds[0]}–{bounds[1]}"


def crisis_comparison(df: pd.DataFrame, crisis_key: str = "asian") -> pd.DataFrame:
    """Comparable pre-crisis mean, shock trough, and post-crisis mean.

    Only the selected country-year rows are used. Partial windows are made
    visible through observation counts; missing years are not imputed.
    """
    spec = CRISES[crisis_key]
    results = []
    for country, group in df.groupby("country", sort=False):
        pre = group[group["year"].between(*spec["pre"])]["gdp_growth"].dropna()
        shock = group[group["year"].between(*spec["shock"])].dropna(subset=["gdp_growth"])
        post = group[group["year"].between(*spec["post"])]["gdp_growth"].dropna()
        trough = shock.loc[shock["gdp_growth"].idxmin()] if len(shock) else None
        results.append({
            "Economy": country,
            "Before mean (%)": pre.mean() if len(pre) else np.nan,
            "Before years": len(pre),
            "Crisis trough (%)": float(trough["gdp_growth"]) if trough is not None else np.nan,
            "Trough year": int(trough["year"]) if trough is not None else np.nan,
            "Crisis years": len(shock),
            "After mean (%)": post.mean() if len(post) else np.nan,
            "After years": len(post),
        })
    return pd.DataFrame(results)


def recovery_data(full: pd.DataFrame, selected: pd.DataFrame,
                  crisis_key: str = "asian") -> tuple[pd.DataFrame, pd.DataFrame]:
    """First *observed* return to pre-shock real GDP, not sustained recovery.

    Baselines come from the full source dataset, even if excluded by filters.
    Missing years are never treated as observed or interpolated. A full shock
    window is required before determining the trough.
    """
    spec = CRISES[crisis_key]
    base_year = spec["baseline"]
    base = full[full["year"] == base_year].set_index("country")["real_gdp"]
    indexed = selected[selected["year"].between(base_year, spec["recovery_end"])].copy()
    indexed["reference_gdp"] = indexed["country"].map(base)
    indexed["gdp_index"] = np.where(indexed["reference_gdp"].gt(0),
                                    indexed["real_gdp"] / indexed["reference_gdp"] * 100, np.nan)
    table = []
    for country in selected["country"].drop_duplicates():
        all_country = indexed[indexed["country"] == country]
        observed = all_country.dropna(subset=["gdp_index"])
        crisis = observed[observed["year"].between(*spec["shock"])]
        baseline = base.get(country, np.nan)
        first_year, years_from_trough, trough_year, trough_index = np.nan, np.nan, np.nan, np.nan
        gap_warning = False
        if pd.isna(baseline) or float(baseline) <= 0:
            status = "Reference GDP unavailable"
        elif crisis["year"].nunique() != spec["shock"][1] - spec["shock"][0] + 1:
            status = "Incomplete shock-window GDP data"
        else:
            trough = crisis.loc[crisis["gdp_index"].idxmin()]
            trough_index = float(trough["gdp_index"])
            trough_year = int(trough["year"])
            if trough_index >= 100:
                status = "No fall below baseline in shock window"
            else:
                later = observed[observed["year"] > trough_year].sort_values("year")
                if later.empty:
                    status = "No subsequent GDP data to assess recovery"
                else:
                    return_years = later[later["gdp_index"] >= 100]
                    if return_years.empty:
                        status = "Not observed as recovered"
                    else:
                        first_year = int(return_years.iloc[0]["year"])
                        years_from_trough = first_year - trough_year
                        years_observed = set(later.loc[later["year"].le(first_year), "year"].astype(int))
                        gap_warning = len(years_observed) < first_year - trough_year
                        status = f"First observed at/above baseline in {first_year}"
        table.append({"Economy": country, "Recovery status": status,
                      "Trough year": trough_year, "Trough GDP index": trough_index,
                      "Shortfall vs baseline (%)": (max(0, 100 - trough_index)
                                                      if pd.notna(trough_index) else np.nan),
                      "First observed year": first_year,
                      "Years after trough": years_from_trough,
                      "Missing years before recovery": "Yes" if gap_warning else "No"})
    return indexed, pd.DataFrame(table)

def crisis_dashboard_stats(df: pd.DataFrame, full: pd.DataFrame,
                           crisis_key: str) -> dict:
    spec = CRISES[crisis_key]
    shock = df[df["year"].between(*spec["shock"])].dropna(subset=["gdp_growth"])
    affected = shock.groupby("country")["gdp_growth"].min() if not shock.empty else pd.Series(dtype=float)
    worst = shock.loc[shock["gdp_growth"].idxmin()] if not shock.empty else None
    _, recovery = recovery_data(full, df, crisis_key)
    recovered_count = int(recovery["First observed year"].notna().sum()) if not recovery.empty else 0
    eligible_count = int(recovery["Recovery status"].isin(
        ["Not observed as recovered"]).sum()) + recovered_count if not recovery.empty else 0
    return {"shock": shock, "affected": affected,
            "worst": worst, "recovered_count": recovered_count,
            "recovery_eligible": eligible_count,
            "recovery_table": recovery}


def three_crisis_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country by crisis: lowest annual growth within each shock window."""
    rows = []
    for key, spec in CRISES.items():
        for country, group in df.groupby("country", sort=False):
            d = group[group["year"].between(*spec["shock"])].dropna(subset=["gdp_growth"])
            if d.empty:
                rows.append({"Economy": country, "Crisis": spec["short"], "GDP growth": np.nan,
                             "Year": np.nan, "Observations": 0})
            else:
                r = d.loc[d["gdp_growth"].idxmin()]
                rows.append({"Economy": country, "Crisis": spec["short"],
                             "GDP growth": float(r["gdp_growth"]), "Year": int(r["year"]),
                             "Observations": len(d)})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="Economy", columns="Crisis", values="GDP growth")
    return pivot, table


RANK_OPTIONS = {
    "Deepest GDP contraction": ("gdp_growth", "min", True, "%"),
    "Highest annual inflation": ("inflation", "max", False, "%"),
    "Lowest average GDP growth": ("gdp_growth", "mean_min", False, "%"),
    "Largest unemployment increase": ("unemployment_change", "max", True, "pp"),
    "Largest currency depreciation": ("fx_depreciation", "max", True, "%"),
}


def top_five(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One economy per row; clear unit and year, no duplicate country ranks."""
    field, method, only_significant, unit = RANK_OPTIONS[metric]
    available = df.dropna(subset=[field]).copy()
    if method == "mean_min":
        mean = available.groupby("country", as_index=False)[field].mean()
        mean["year_label"] = "Mean"
        rank = mean.nsmallest(5, field)
    else:
        if only_significant:
            available = available[available[field] < 0] if method == "min" else available[available[field] > 0]
        if available.empty:
            return pd.DataFrame(columns=["rank", "country", "year_label", "score", "unit"])
        available = available.sort_values([field, "country", "year"], ascending=[method == "min", True, True])
        # For a minimum, sort ascending. For a maximum, sort descending.
        rank = available.drop_duplicates("country").head(5).copy()
        rank["year_label"] = rank["year"].astype(int).astype(str)
    rank = rank.reset_index(drop=True)
    rank["rank"] = range(1, len(rank) + 1)
    rank["unit"] = unit
    return rank.rename(columns={field: "score"})[["rank", "country", "year_label", "score", "unit"]]


def economic_insights(df: pd.DataFrame, crisis_key: str = ALL_TIME_KEY) -> list[tuple[str, str]]:
    """Three evidence-first, filter-sensitive descriptive findings."""
    growth = df.dropna(subset=["gdp_growth"])
    if growth.empty:
        return [("Insufficient GDP observations", "No GDP growth data are available for this selection.")]
    if crisis_key == ALL_TIME_KEY:
        worst = growth.loc[growth["gdp_growth"].idxmin()]
        heading = "Largest observed annual contraction" if worst["gdp_growth"] < 0 else "Lowest observed annual growth"
        first = (heading,
                 f"{worst['country']} recorded {worst['gdp_growth']:.2f}% GDP growth in {int(worst['year'])}, "
                 "the lowest annual rate in the current selection. This is a one-year growth rate, "
                 "not the cumulative decline in output.")
        windows = []
        for key, spec in CRISES.items():
            sample = growth[growth["year"].between(*spec["shock"])]
            if not sample.empty:
                low = sample.loc[sample["gdp_growth"].idxmin()]
                windows.append(f"{spec['title']}: {low['country']} ({low['gdp_growth']:.2f}% in {int(low['year'])})")
        second = ("Economic shocks affected economies differently",
                  "Lowest observed annual GDP growth in each shock window: " + "; ".join(windows)
                  + ". These values help compare timing and severity, but alone cannot identify causes."
                  if windows else "The current selection has no observations in the three crisis windows.")
    else:
        spec = CRISES[crisis_key]
        shock = growth[growth["year"].between(*spec["shock"])]
        if shock.empty:
            first = ("Depth of the downturn", "No GDP observations from the selected shock window are available.")
        else:
            worst = shock.loc[shock["gdp_growth"].idxmin()]
            affected = int(shock.groupby("country")["gdp_growth"].min().lt(0).sum())
            first = ("Depth of the downturn",
                     f"In {year_span(spec['shock'])}, {affected} of {shock['country'].nunique()} economies with "
                     f"available observations had a year of negative GDP growth. {worst['country']} "
                     f"had the lowest rate ({worst['gdp_growth']:.2f}% in {int(worst['year'])}). "
                     f"The crisis context is described by {spec['source'].split(' — ')[0]}; the measured "
                     "contraction is not an estimate of any policy's causal effect.")
        comparison = crisis_comparison(df, crisis_key)
        complete = comparison[(comparison["Before years"] == 3) & (comparison["After years"] == 3)].copy()
        if complete.empty:
            second = ("Growth before and after the shock",
                      "A full three-year comparison is not possible with the current filters or missing values. "
                      "Expand the period for a like-for-like comparison.")
        else:
            complete["change"] = complete["After mean (%)"] - complete["Before mean (%)"]
            low = complete.loc[complete["change"].idxmin()]
            second = ("Growth before and after the shock",
                      f"Among economies with complete windows, {low['Economy']} had the largest decline in "
                      f"average annual growth: {low['Before mean (%)']:.2f}% in {year_span(spec['pre'])} "
                      f"versus {low['After mean (%)']:.2f}% in {year_span(spec['post'])} "
                      f"({low['change']:+.2f} percentage points). Differences in averages are descriptive "
                      "and should not be read as the shock's causal impact.")
    stats = okun_statistics(df)
    if stats["two_way_slope"] is not None:
        beta = stats["two_way_slope"]
        sign = ("consistent in sign with the usual inverse Okun-style association"
                if beta < 0 else "not inverse in this selected sample")
        third = ("Growth and labor-market adjustment",
                 f"After accounting for average economy and year differences, 1 percentage point "
                 f"higher annual GDP growth is associated with {beta:+.3f} percentage points of "
                 f"annual unemployment-rate change across {stats['two_way_n']} paired observations. "
                 f"The sign is {sign}. The result is descriptive, sensitive to the sample, and not causal.")
    elif stats["within_r"] is not None:
        third = ("Growth and labor-market adjustment",
                 f"The within-economy correlation of GDP growth and annual unemployment change is "
                 f"{stats['within_r']:+.2f} across {stats['n']} complete economy-years. "
                 "This removes economy averages, not calendar-year shocks or other confounders.")
    else:
        inflation = df.groupby("country")["inflation"].mean().dropna().sort_values()
        if len(inflation) >= 2:
            third = ("Inflation differences",
                     f"Mean annual inflation ranged from {inflation.iloc[0]:.2f}% ({inflation.index[0]}) "
                     f"to {inflation.iloc[-1]:.2f}% ({inflation.index[-1]}). These are average annual rates, "
                     "not the total change in consumer prices over the period.")
        else:
            third = ("Data limitations", "The selected years do not provide enough paired labor and GDP data "
                     "for an interpretable cross-economy comparison.")
    return [first, second, third]

def create_all_time_summary(df: pd.DataFrame, countries: list[str],
                            years: tuple[int, int], crisis_key: str = ALL_TIME_KEY) -> bytes:
    """PDF with three data-grounded findings, method details and author-year references."""
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=(210 * mm, 297 * mm),
                            leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=15 * mm, bottomMargin=14 * mm,
                            title="MacroLens Asia | Executive Summary", author="NV-215")
    styles = getSampleStyleSheet()
    navy, ink, muted = (colors.HexColor(c) for c in ("#17334E", "#293949", "#526579"))
    for name, parent, size, leading, color, before, after, font in [
        ("MLTitle", "Title", 16.2, 19, navy, 0, 5, "Helvetica-Bold"),
        ("MLSubtitle", "Normal", 10, 13.5, navy, 0, 5, "Helvetica-Bold"),
        ("MLSection", "Heading2", 9.8, 13, navy, 8, 3, "Helvetica-Bold"),
        ("MLBody", "BodyText", 8.9, 12.6, ink, 0, 5, "Helvetica"),
        ("MLFinding", "Normal", 9, 12, navy, 4, 2, "Helvetica-Bold"),
        ("MLSmall", "BodyText", 7.25, 10.2, muted, 0, 2.5, "Helvetica"),
    ]:
        styles.add(ParagraphStyle(name=name, parent=styles[parent], fontName=font,
                                  fontSize=size, leading=leading, textColor=color,
                                  alignment=TA_LEFT, spaceBefore=before, spaceAfter=after))
    active = CRISES.get(crisis_key)
    full_view = crisis_key == ALL_TIME_KEY
    label = "Long-run comparison of three major shocks" if full_view else active["title"]
    names = ", ".join(countries)
    today = datetime.now(timezone.utc).strftime("%d %B %Y (UTC)")
    narrative = (
        "The three events are different: the Asian Financial Crisis combined currency and banking stress "
        "(IMF, 1999); the Global Financial Crisis affected Asian exporters through a sharp fall in world demand "
        "(IMF, 2009); and COVID-19 disrupted domestic activity, travel and trade (IMF, 2020; World Bank, 2023). "
        if full_view else
        active["context"] + " "
    )
    story = [
        Paragraph("EXECUTIVE SUMMARY", styles["MLTitle"]),
        Paragraph("MacroLens Asia | Economic Growth, Employment and Crisis Recovery", styles["MLSubtitle"]),
        Paragraph(f"<b>Economies:</b> {html.escape(names)}<br/>"
                  f"<b>Period:</b> {years[0]}–{years[1]} | <b>Focus:</b> {html.escape(label)}<br/>"
                  f"<b>Prepared by:</b> NV-215 | <b>Report date:</b> {today}", styles["MLSmall"]),
        Spacer(1, 3), HRFlowable(width="100%", thickness=.65, color=colors.HexColor("#D1DCE5")),
        Paragraph("1. Purpose and economic context", styles["MLSection"]),
        Paragraph("I use real GDP growth, the unemployment rate and CPI inflation to compare output, "
                  "labor-market conditions and price developments. Constant-price GDP levels and exchange rates "
                  "help examine how severe downturns were and how long recovery took (World Bank, WDI). "
                  + html.escape(narrative), styles["MLBody"]),
        Paragraph("2. Data and method", styles["MLSection"]),
        Paragraph(f"I used annual World Bank World Development Indicators for {html.escape(names)} "
                  f"in {years[0]}–{years[1]}. The five indicator series are GDP growth, unemployment "
                  "(ILO modeled estimate), CPI inflation, real GDP levels and the official exchange rate. "
                  "Observations were joined by economy and year; missing values remain missing. "
                  "I calculate unemployment changes in percentage points from consecutive years, "
                  "and currency depreciation as the percent change in local currency per US dollar. "
                  "Recovery is the first observed return to the real GDP level of 1996, 2007 or 2019 "
                  "for the relevant crisis, not simply the first positive growth rate. "
                  "Data checks compare official GDP growth with changes in real GDP levels, without "
                  "replacing the original series.", styles["MLBody"]),
        Paragraph("3. Main findings", styles["MLSection"]),
    ]
    for number, (heading, detail) in enumerate(economic_insights(df, crisis_key), 1):
        story.append(KeepTogether([
            Paragraph(f"{number}. {html.escape(heading)}", styles["MLFinding"]),
            Paragraph(html.escape(detail), styles["MLBody"]),
        ]))
    checks, _ = check_data_quality(df)
    stats = okun_statistics(df)
    missing = {k: int(df[k].isna().sum()) for k in SOURCE_FIELDS}
    within = (f"The within-economy GDP-growth/unemployment-change correlation was "
              f"{stats['within_r']:+.2f} across {stats['n']} matched observations. "
              if stats["within_r"] is not None else
              "The selected period does not provide enough usable annual changes to calculate a meaningful "
              "within-economy growth/unemployment association. ")
    adjusted_note = (f"The country-and-year-adjusted descriptive slope was "
                     f"{stats['two_way_slope']:+.3f} pp of unemployment change per 1 pp growth. "
                     if stats["two_way_slope"] is not None else "")
    match_note = (f"A GDP growth/real-GDP-level consistency comparison was possible for "
                  f"{checks['growth_comparisons']} observations; {checks['large_growth_differences']} "
                  "had an absolute difference above 0.5 percentage points. "
                  if checks["growth_comparisons"] else
                  "A direct growth/real-GDP-level consistency check was unavailable for this selection. ")
    story += [
        Paragraph("4. Data quality and limitations", styles["MLSection"]),
        Paragraph(f"The selection covers {len(df)} economy-year rows; {checks['complete_three']} have "
                  "all three primary indicators. Missing: GDP growth "
                  f"{missing['gdp_growth']}, unemployment {missing['unemployment']}, "
                  f"inflation {missing['inflation']}, constant-price GDP {missing['real_gdp']}, "
                  f"exchange rate {missing['fx_rate']}. "
                  + within + adjusted_note + match_note +
                  "Reported GDP growth is not overwritten. All estimates are descriptive, not causal. "
                  "Differences in economic structure, "
                  "exchange-rate regimes, differing source coverage and ILO modeled estimates remain important, "
                  "and source values may be revised.",
                  styles["MLBody"]),
        Paragraph("References", styles["MLSection"]),
    ]
    references = [
        ("World Bank. World Development Indicators — GDP growth (annual %).",
         SOURCES["World Bank — GDP growth (annual %)"]),
        ("World Bank / ILO. WDI — Unemployment (modeled ILO estimate).",
         SOURCES["World Bank — Unemployment (ILO modeled estimate)"]),
        ("World Bank. WDI — Consumer price inflation.",
         SOURCES["World Bank — Consumer price inflation"]),
        ("World Bank. WDI — GDP at constant prices.",
         SOURCES["World Bank — GDP (constant-price US$)"]),
        ("World Bank. WDI — Official exchange rate.",
         SOURCES["World Bank — Official exchange rate (LCU per US$)"]),
        ("Federal Reserve Bank of St. Louis (2013). Output and Unemployment.",
         SOURCES["Federal Reserve Bank of St. Louis (2013) — Output and Unemployment"]),
    ]
    background = (
        [("International Monetary Fund (1999). The Asia Crisis: Causes, Policy Responses, and Outcomes.",
          SOURCES["IMF (1999) — The Asia Crisis: Causes, Policy Responses, and Outcomes"])] if crisis_key == "asian" else
        [("International Monetary Fund (2009). World Economic Outlook: Crisis and Recovery.",
          SOURCES["IMF (2009) — World Economic Outlook: Crisis and Recovery"])] if crisis_key == "global" else
        [("International Monetary Fund (2020). World Economic Outlook: The Great Lockdown.",
          SOURCES["IMF (2020) — World Economic Outlook: The Great Lockdown"]),
         ("World Bank (2023). COVID-19 recovery in East Asia and Pacific.",
          SOURCES["World Bank (2023) — COVID-19 recovery in East Asia and Pacific"])] if crisis_key == "covid" else
        [("IMF (1999). Asian Financial Crisis: policy responses and outcomes.",
          SOURCES["IMF (1999) — The Asia Crisis: Causes, Policy Responses, and Outcomes"]),
         ("IMF (2009). World Economic Outlook: Crisis and Recovery.",
          SOURCES["IMF (2009) — World Economic Outlook: Crisis and Recovery"]),
         ("IMF (2020). World Economic Outlook: The Great Lockdown.",
          SOURCES["IMF (2020) — World Economic Outlook: The Great Lockdown"]),
         ("World Bank (2023). Recovery from COVID-19 in East Asia and Pacific.",
          SOURCES["World Bank (2023) — COVID-19 recovery in East Asia and Pacific"]) ]
    )
    for title, uri in references + background:
        story.append(Paragraph(f'<link href="{html.escape(uri, quote=True)}" color="#225B86">'
                               f'{html.escape(title)}</link>', styles["MLSmall"]))
    doc.build(story)
    return out.getvalue()



def export_scope(full: pd.DataFrame, selected: pd.DataFrame,
                 economies: list[str], years: tuple[int, int],
                 crisis_key: str) -> tuple[pd.DataFrame, tuple[int, int], str]:
    """A named crisis has a fixed comparison window; All Time keeps slider-selected data.

    Named reports must include the full baseline, shock, and post-shock years even
    if the user is temporarily zoomed into a narrower chart range.
    """
    if crisis_key == ALL_TIME_KEY:
        return selected.copy(), years, "all_time"
    spec = CRISES[crisis_key]
    report_years = tuple(spec["view"])
    frame = full[
        full["country"].isin(economies)
        & full["year"].between(*report_years)
    ].copy()
    return frame, report_years, crisis_key


def create_analysis_summary(df: pd.DataFrame, countries: list[str],
                            years: tuple[int, int], crisis_key: str = ALL_TIME_KEY) -> bytes:
    """PDF with three data-grounded findings, method details and author-year references."""

    if crisis_key == ALL_TIME_KEY:
        return create_all_time_summary(df, countries, years, crisis_key)

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=(210 * mm, 297 * mm),
                            leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=15 * mm, bottomMargin=14 * mm,
                            title="MacroLens Asia | Executive Summary", author="NV-215")
    styles = getSampleStyleSheet()
    navy, ink, muted = (colors.HexColor(c) for c in ("#17334E", "#293949", "#526579"))
    for name, parent, size, leading, color, before, after, font in [
        ("MLTitle", "Title", 16.2, 19, navy, 0, 5, "Helvetica-Bold"),
        ("MLSubtitle", "Normal", 10, 13.5, navy, 0, 5, "Helvetica-Bold"),
        ("MLSection", "Heading2", 9.8, 13, navy, 8, 3, "Helvetica-Bold"),
        ("MLBody", "BodyText", 8.9, 12.6, ink, 0, 5, "Helvetica"),
        ("MLFinding", "Normal", 9, 12, navy, 4, 2, "Helvetica-Bold"),
        ("MLSmall", "BodyText", 7.25, 10.2, muted, 0, 2.5, "Helvetica"),
    ]:
        styles.add(ParagraphStyle(name=name, parent=styles[parent], fontName=font,
                                  fontSize=size, leading=leading, textColor=color,
                                  alignment=TA_LEFT, spaceBefore=before, spaceAfter=after))
    active = CRISES.get(crisis_key)
    full_view = crisis_key == ALL_TIME_KEY
    label = "Long-run comparison of three major shocks" if full_view else active["title"]
    names = ", ".join(countries)
    today = datetime.now(timezone.utc).strftime("%d %B %Y (UTC)")
    narrative = (
        "The events had different mechanisms. The 1997 crisis began with the Thai baht float and "
        "regional currency and banking stress (Federal Reserve History, 2013). The 2008-09 crisis "
        "spread to Asian economies through international finance and trade (IMF, 2009; "
        "World Bank, 2010). The 2020 pandemic disrupted mobility, production and demand "
        "(IMF, 2020). "
        if full_view else active["context"] + " "
    )
    story = [
        Paragraph("EXECUTIVE SUMMARY", styles["MLTitle"]),
        Paragraph("MacroLens Asia | Economic Growth, Employment and Crisis Recovery", styles["MLSubtitle"]),
        Paragraph(f"<b>Economies:</b> {html.escape(names)}<br/>"
                  f"<b>Period:</b> {years[0]}–{years[1]} | <b>Focus:</b> {html.escape(label)}<br/>"
                  f"<b>Prepared by:</b> NV-215 | <b>Report date:</b> {today}", styles["MLSmall"]),
        Spacer(1, 3), HRFlowable(width="100%", thickness=.65, color=colors.HexColor("#D1DCE5")),
        Paragraph("1. Purpose and economic context", styles["MLSection"]),
        Paragraph("I use real GDP growth, the unemployment rate and CPI inflation to compare output, "
                  "labor-market conditions and price developments. Constant-price GDP levels and exchange rates "
                  "help examine how severe downturns were and how long recovery took (World Bank, WDI). "
                  + html.escape(narrative), styles["MLBody"]),
        Paragraph("2. Data and method", styles["MLSection"]),
        Paragraph(f"I used annual World Bank World Development Indicators for {html.escape(names)} "
                  f"in {years[0]}–{years[1]}. The five indicator series are GDP growth, unemployment "
                  "(ILO modeled estimate), CPI inflation, real GDP levels and the official exchange rate. "
                  "Observations were joined by economy and year; missing values remain missing. "
                  "I calculate unemployment changes in percentage points from consecutive years, "
                  "and currency depreciation as the percent change in local currency per US dollar. "
                  "Recovery is the first observed return to the real GDP level of 1996, 2007 or 2019 "
                  "for the relevant crisis, not simply the first positive growth rate. "
                  "Data checks compare official GDP growth with changes in real GDP levels, without "
                  "replacing the original series. Historical events come from cited institutional "
                  "accounts; all numerical findings come from the selected WDI observations. "
                  "Annual values cannot identify the exact timing or causal impact of events.", styles["MLBody"]),
        Paragraph("3. Main findings", styles["MLSection"]),
    ]
    for number, (heading, detail) in enumerate(economic_insights(df, crisis_key), 1):
        story.append(KeepTogether([
            Paragraph(f"{number}. {html.escape(heading)}", styles["MLFinding"]),
            Paragraph(html.escape(detail), styles["MLBody"]),
        ]))
    checks, _ = check_data_quality(df)
    stats = okun_statistics(df)
    missing = {k: int(df[k].isna().sum()) for k in SOURCE_FIELDS}
    within = (f"The within-economy GDP-growth/unemployment-change correlation was "
              f"{stats['within_r']:+.2f} across {stats['n']} matched observations. "
              if stats["within_r"] is not None else
              "The selected period does not provide enough usable annual changes to calculate a meaningful "
              "within-economy growth/unemployment association. ")
    adjusted_note = (f"The country-and-year-adjusted descriptive slope was "
                     f"{stats['two_way_slope']:+.3f} pp of unemployment change per 1 pp growth. "
                     if stats["two_way_slope"] is not None else "")
    match_note = (f"A GDP growth/real-GDP-level consistency comparison was possible for "
                  f"{checks['growth_comparisons']} observations; {checks['large_growth_differences']} "
                  "had an absolute difference above 0.5 percentage points. "
                  if checks["growth_comparisons"] else
                  "A direct growth/real-GDP-level consistency check was unavailable for this selection. ")
    story += [
        Paragraph("4. Data quality and limitations", styles["MLSection"]),
        Paragraph(f"The selection covers {len(df)} economy-year rows; {checks['complete_three']} have "
                  "all three primary indicators. Missing: GDP growth "
                  f"{missing['gdp_growth']}, unemployment {missing['unemployment']}, "
                  f"inflation {missing['inflation']}, constant-price GDP {missing['real_gdp']}, "
                  f"exchange rate {missing['fx_rate']}. "
                  + within + adjusted_note + match_note +
                  "Reported GDP growth is not overwritten. All estimates are descriptive, not causal. "
                  "Differences in economic structure, "
                  "exchange-rate regimes, differing source coverage and ILO modeled estimates remain important, "
                  "and source values may be revised.",
                  styles["MLBody"]),
        Paragraph("References", styles["MLSection"]),
    ]
    references = [
        ("World Bank. World Development Indicators — GDP growth (annual %).",
         SOURCES["World Bank — GDP growth (annual %)"]),
        ("World Bank / ILO. WDI — Unemployment (modeled ILO estimate).",
         SOURCES["World Bank — Unemployment (ILO modeled estimate)"]),
        ("World Bank. WDI — Consumer price inflation.",
         SOURCES["World Bank — Consumer price inflation"]),
        ("World Bank. WDI — GDP at constant prices.",
         SOURCES["World Bank — GDP (constant-price US$)"]),
        ("World Bank. WDI — Official exchange rate.",
         SOURCES["World Bank — Official exchange rate (LCU per US$)"]),
        ("Federal Reserve Bank of St. Louis (2013). Output and Unemployment.",
         SOURCES["Federal Reserve Bank of St. Louis (2013) — Output and Unemployment"]),
    ]
    background = (
        [("International Monetary Fund (1999). The Asia Crisis: Causes, Policy Responses, and Outcomes.",
          SOURCES["IMF (1999) — The Asia Crisis: Causes, Policy Responses, and Outcomes"])] if crisis_key == "asian" else
        [("International Monetary Fund (2009). World Economic Outlook: Crisis and Recovery.",
          SOURCES["IMF (2009) — World Economic Outlook: Crisis and Recovery"]),
         ("World Bank (2010). Economic impact of the 2008 financial crisis.",
          SOURCES["World Bank (2010) — Economic impact of the 2008 financial crisis"])] if crisis_key == "global" else
        [("International Monetary Fund (2020). World Economic Outlook: The Great Lockdown.",
          SOURCES["IMF (2020) — World Economic Outlook: The Great Lockdown"]),
         ("World Bank (2023). COVID-19 recovery in East Asia and Pacific.",
          SOURCES["World Bank (2023) — COVID-19 recovery in East Asia and Pacific"])] if crisis_key == "covid" else
        [("Federal Reserve History (2013). Asian Financial Crisis.",
          SOURCES["Federal Reserve History (2013) — Asian Financial Crisis"]),
         ("IMF (2009). World Economic Outlook: Crisis and Recovery.",
          SOURCES["IMF (2009) — World Economic Outlook: Crisis and Recovery"]),
         ("World Bank (2010). Financial crisis and East Asian trade.",
          SOURCES["World Bank (2010) — Economic impact of the 2008 financial crisis"]),
         ("IMF (2020). World Economic Outlook: The Great Lockdown.",
          SOURCES["IMF (2020) — World Economic Outlook: The Great Lockdown"]),
]    )
    for title, uri in references + background:
        story.append(Paragraph(f'<link href="{html.escape(uri, quote=True)}" color="#225B86">'
                               f'{html.escape(title)}</link>', styles["MLSmall"]))
    doc.build(story)
    return out.getvalue()


def tune_fig(fig, y_title=None, height=385):
    """Allow Streamlit's active light/dark theme to style Plotly figures.

    Avoid fixed white templates, dark font colors, and pale grids; those
    produced unreadable charts whenever the browser theme changed.
    """
    fig.update_layout(
        template=None,
        height=height,
        margin=dict(l=16, r=18, t=54, b=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12),
        title=dict(font=dict(size=16)),
        legend_title_text="Economy",
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def _valid_crisis_key(value) -> str:
    """Keep a real crisis selection when Streamlit sends None on deselection."""
    valid = (*CRISES, ALL_TIME_KEY)
    if value in valid:
        return value
    previous = st.session_state.get("last_valid_crisis")
    return previous if previous in valid else ALL_TIME_KEY


def period_changed():
    preset = st.session_state.get("period_preset", FULL_PERIOD_NAME)
    bounds = PERIOD_PRESETS.get(preset)
    if bounds is None:
        return
    st.session_state["selected_years"] = bounds
    if preset == FULL_PERIOD_NAME:
        st.session_state["active_crisis"] = ALL_TIME_KEY
        st.session_state["last_valid_crisis"] = ALL_TIME_KEY
        st.session_state["primary_crisis"] = ALL_TIME_KEY
    else:
        for key, name in PRESET_FOR_CRISIS.items():
            if preset == name:
                st.session_state["active_crisis"] = key
                st.session_state["last_valid_crisis"] = key
                st.session_state["primary_crisis"] = key
                break


def years_changed():
    years = tuple(st.session_state.get("selected_years", (START_YEAR, END_YEAR)))
    st.session_state["period_preset"] = next(
        (name for name, bounds in PERIOD_PRESETS.items() if bounds == years), "Custom range"
    )


def primary_crisis_changed():
    """The four crisis buttons select the analysis period and the graph period."""
    selection = st.session_state.get("primary_crisis")
    if selection not in (*CRISES, ALL_TIME_KEY):
        # A Streamlit segmented control can become None if clicked twice.
        st.session_state["primary_crisis"] = _valid_crisis_key(None)
        return
    st.session_state["active_crisis"] = selection
    active_crisis_changed()


def active_crisis_changed():
    """Synchronize the selected historical period with the sidebar controls."""
    crisis_key = _valid_crisis_key(st.session_state.get("active_crisis"))
    st.session_state["active_crisis"] = crisis_key
    st.session_state["last_valid_crisis"] = crisis_key
    if crisis_key == ALL_TIME_KEY:
        st.session_state["period_preset"] = FULL_PERIOD_NAME
        st.session_state["selected_years"] = (START_YEAR, END_YEAR)
    else:
        st.session_state["period_preset"] = PRESET_FOR_CRISIS[crisis_key]
        st.session_state["selected_years"] = CRISES[crisis_key]["view"]


def plot_trend(data: pd.DataFrame, field: str, years: tuple[int, int]):
    trend = data.dropna(subset=[field])
    if trend.empty:
        st.info("No observations are available for this indicator and selection.")
        return
    fig = px.line(trend, x="year", y=field, color="country", color_discrete_map=COLORS,
                  markers=years[1] - years[0] <= 12,
                  title=f"{METRIC_LABELS[field]} | {years[0]}-{years[1]}",
                  labels={"year": "Year", field: METRIC_LABELS[field], "country": "Economy"})
    if field == "gdp_growth":
        if years[0] <= 1999 and years[1] >= 1997:
            fig.add_vrect(x0=1996.5, x1=1999.5, fillcolor="rgba(214,142,64,.10)", line_width=0)
        for shock in (2009, 2020):
            if years[0] <= shock <= years[1]:
                fig.add_vline(x=shock, line_dash="dot", line_color="#9BA7B4", line_width=1)
    tune_fig(fig, METRIC_LABELS[field])
    fig.update_xaxes(dtick=max(1, (years[1] - years[0]) // 7))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if field == "gdp_growth":
        st.caption("Historical markers: 1997-1999 Asian Financial Crisis, "
                   "2009 global financial crisis, and 2020 COVID-19 shock. "
                   "Markers provide context, not causal estimates.")


def render_heatmap(df: pd.DataFrame, countries: list[str], years: tuple[int, int]):
    st.subheader("Economic heatmap")
    field = st.selectbox("Choose an indicator", list(METRIC_LABELS),
                         format_func=lambda name: METRIC_LABELS[name], key="heat_metric")
    pivot = df.pivot(index="country", columns="year", values=field)
    pivot = pivot.reindex(index=countries,
                          columns=list(range(years[0], years[1] + 1)))
    vals = pivot.to_numpy(dtype=float)
    flat = vals[np.isfinite(vals)]
    if not len(flat):
        st.info("No usable observations for this heatmap.")
        return
    # Other values appear in the hover without changing the heatmap's chosen unit.
    hover_fields = ["gdp_growth", "unemployment", "inflation", "fx_depreciation"]
    hover_values = [
        df.pivot(index="country", columns="year", values=f)
          .reindex(index=countries, columns=pivot.columns)
          .to_numpy(dtype=float) for f in hover_fields
    ]
    hover_text = []
    for r, country in enumerate(countries):
        hover_text.append([])
        for c, year in enumerate(pivot.columns):
            parts = [f"<b>{html.escape(country)}</b> | {year}"]
            for name, array in zip(("GDP growth", "Unemployment", "Inflation", "FX change"), hover_values):
                number = array[r, c]
                parts.append(f"{name}: {number:.2f}%" if np.isfinite(number) else f"{name}: N/A")
            hover_text[-1].append("<br>".join(parts))
    if field in ("gdp_growth", "fx_depreciation"):
        extreme = max(abs(float(flat.min())), abs(float(flat.max())), .1)
        scale_args = dict(colorscale="RdBu", zmin=-extreme, zmax=extreme, zmid=0)
    else:
        scale_args = dict(colorscale="Blues", zmin=float(flat.min()),
                          zmax=float(flat.max()) if flat.max() > flat.min() else float(flat.min()) + 1)
    fig = go.Figure(go.Heatmap(
        z=vals, x=list(pivot.columns), y=list(pivot.index), text=hover_text,
        hoverongaps=False, xgap=1, ygap=2, **scale_args,
        colorbar=dict(title=METRIC_LABELS[field]),
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(template=None, height=max(330, 98 + 43 * len(countries)),
                      margin=dict(l=18, r=14, t=18, b=38),
                      paper_bgcolor="rgba(0,0,0,0)",
                      # Translucent neutral gray stays readable in either mode;
                      # unobserved values remain visibly distinct from zero.
                      plot_bgcolor="rgba(125,135,145,0.15)",
                      yaxis=dict(autorange="reversed", title="Economy"),
                      xaxis=dict(title="Year", tickmode="linear", tick0=years[0],
                                 dtick=1 if years[1] - years[0] < 13 else 5))
    if years[0] <= 1999 and years[1] >= 1997:
        fig.add_vrect(x0=1996.5, x1=1999.5,
                      fillcolor="rgba(255,165,0,.035)",
                      line_color="rgba(130,85,32,.8)", line_width=1)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption("Each cell represents one economy-year. Grey cells mean missing data; "
               "hover for the other indicators. The shaded outline marks 1997-1999. "
               "Positive FX changes mean depreciation against the US dollar; "
               "different exchange-rate regimes limit direct comparisons.")


def render_crisis_comparison(selected: pd.DataFrame, crisis_key: str):
    spec = CRISES[crisis_key]
    comparison = crisis_comparison(selected, crisis_key)
    display = comparison.dropna(subset=["Before mean (%)", "After mean (%)"], how="all")
    if display.empty:
        st.info("The selected years do not cover the pre- and post-crisis windows. "
                "Use the crisis preset or widen the year slider.")
        return
    st.caption(f"Mean growth before ({year_span(spec['pre'])}) and after ({year_span(spec['post'])}); "
               f"the trough is the lowest single-year rate during {year_span(spec['shock'])}. "
               "The table shows how many years are observed in each period.")
    complete = display[(display["Before years"] == 3) & (display["After years"] == 3)]
    if len(complete) != len(display):
        st.caption("Some economies have incomplete three-year windows. They remain visible in the table "
                   "but are omitted from the bar comparison to avoid misleading averages.")
    long = complete.melt(id_vars=["Economy"], value_vars=["Before mean (%)", "After mean (%)"],
                        var_name="Period", value_name="GDP growth (%)").dropna()
    if not long.empty:
        long["Period"] = long["Period"].replace({
            "Before mean (%)": f"Before ({year_span(spec['pre'])})",
            "After mean (%)": f"After ({year_span(spec['post'])})"})
        fig = px.bar(long, x="Economy", y="GDP growth (%)", color="Period", barmode="group",
                     color_discrete_sequence=["#638EAB", "#BA9270"],
                     title=f"GDP growth before and after: {spec['title']}")
        tune_fig(fig, "Average annual GDP growth (%)")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.dataframe(comparison, hide_index=True, use_container_width=True,
                 column_config={key: st.column_config.NumberColumn(format="%.2f%%")
                                for key in ("Before mean (%)", "Crisis trough (%)", "After mean (%)")})


def render_recovery(full: pd.DataFrame, filtered: pd.DataFrame, crisis_key: str):
    spec = CRISES[crisis_key]
    if full["real_gdp"].notna().sum() == 0:
        st.info("Constant-price GDP is unavailable. Refresh the data when the World Bank API is reachable.")
        return
    indexed, statuses = recovery_data(full, filtered, crisis_key)
    valid = indexed.dropna(subset=["gdp_index"])
    if valid.empty:
        st.info(f"No GDP index values in the selected part of {spec['baseline']}–{spec['recovery_end']}. "
                "Use this crisis's period preset or widen the year range.")
        return
    fig = px.line(valid, x="year", y="gdp_index", color="country", markers=True,
                  color_discrete_map=COLORS,
                  title=f"Real GDP index ({spec['baseline']} = 100)",
                  labels={"year": "Year", "gdp_index": "Real GDP index", "country": "Economy"})
    fig.add_hline(y=100, line_dash="dash", line_color="#8E9CA9", line_width=1)
    fig.add_vrect(x0=spec["shock"][0]-.5, x1=spec["shock"][1]+.5,
                  fillcolor="rgba(192,149,92,.12)", line_width=0)
    tune_fig(fig, "Index (baseline = 100)", height=405)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption(f"Each economy's {spec['baseline']} real GDP = 100. Recovery is the first observed "
               "post-trough year at or above that level, and only economies whose output fell "
               "below it during the shock window are assigned a recovery year. Missing years "
               "can delay the first observed recovery date. A return above baseline may not be sustained.")
    st.dataframe(statuses, hide_index=True, use_container_width=True)


def render_currency_pressure(df: pd.DataFrame, crisis_key: str):
    spec = CRISES[crisis_key]
    start, end = spec["pre"][0], spec["post"][1]
    fx = df[df["year"].between(start, end)].dropna(subset=["fx_depreciation"])
    if fx.empty:
        st.info(f"No exchange-rate observations within {start}–{end} for the current selection.")
        return
    fig = px.line(fx, x="year", y="fx_depreciation", color="country", markers=True,
                  color_discrete_map=COLORS,
                  title=f"Annual exchange-rate change around {spec['title']}",
                  labels={"year": "Year", "fx_depreciation": "LCU per US$ change (%)",
                          "country": "Economy"})
    fig.add_hline(y=0, line_width=1, line_color="#8F9DAA")
    fig.add_vrect(x0=spec["shock"][0]-.5, x1=spec["shock"][1]+.5,
                  fillcolor="rgba(192,149,92,.12)", line_width=0)
    tune_fig(fig, "Annual exchange-rate change (%)")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption("Positive annual changes in local currency per US dollar indicate depreciation. "
               "Annual averages can obscure sharp short-term movements; fixed and floating "
               "exchange-rate regimes are not directly comparable.")


def render_three_crisis_comparison(df: pd.DataFrame, countries: list[str]):
    st.caption("The color shows the lowest annual real GDP growth rate for each economy within "
               "each shock window. This is not a three-year average or an estimate of the crisis's causal effect.")
    pivot, rows = three_crisis_matrix(df)
    labels = [spec["short"] for spec in CRISES.values()]
    matrix = pivot.reindex(index=countries, columns=labels)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        st.info("No crisis-window observations fall inside the selected year range.")
        return
    extremes = max(2.0, abs(float(finite.min())), abs(float(finite.max())))
    hover = []
    for econ in countries:
        row = []
        for name in labels:
            record = rows[(rows["Economy"] == econ) & (rows["Crisis"] == name)].iloc[0]
            if pd.isna(record["GDP growth"]):
                row.append(f"{html.escape(econ)} | {name}<br>Data unavailable in selected years")
            else:
                row.append(f"{html.escape(econ)} | {name}<br>Worst annual growth: "
                           f"{record['GDP growth']:.2f}% ({int(record['Year'])})")
        hover.append(row)
    fig = go.Figure(go.Heatmap(z=values, x=labels, y=countries,
                               colorscale="RdBu", zmin=-extremes, zmax=extremes, zmid=0,
                               text=hover, hovertemplate="%{text}<extra></extra>",
                               hoverongaps=False, xgap=4, ygap=3,
                               colorbar=dict(title="GDP growth (%)")))
    fig.update_layout(template=None, height=max(340, 115 + 38*len(countries)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(125,135,145,.15)",
                      margin=dict(l=16, r=20, t=30, b=30),
                      xaxis=dict(title="Economic shock"),
                      yaxis=dict(autorange="reversed", title="Economy"))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    with st.expander("Show country-by-crisis observations"):
        st.dataframe(rows.rename(columns={"GDP growth": "Worst GDP growth (%)"}),
                     use_container_width=True, hide_index=True)
    st.caption("Some economies may have avoided a fall in total real GDP even if their annual "
               "growth slowed. A positive trough means growth remained positive during the years checked.")


def historical_world_bank_snapshot(source: pd.DataFrame, countries: list[str],
                                   crisis_key: str) -> dict:
    """Descriptive WDI observations for selected economies in a fixed shock window.

    Uses the unfiltered WDI history (or an uploaded WDI CSV backup) so a custom
    chart-year slider does not accidentally erase historical crisis evidence.
    Missing observations remain missing; no event effect is estimated.
    """
    spec = CRISES[crisis_key]
    frame = source[source["country"].isin(countries) &
                   source["year"].between(*spec["shock"])].copy()
    growth = frame.dropna(subset=["gdp_growth"])
    landmark = growth[growth["year"] == spec["landmark"]]
    negative = int(landmark["gdp_growth"].lt(0).sum())
    denominator = int(landmark["country"].nunique())
    worst = growth.loc[growth["gdp_growth"].idxmin()] if not growth.empty else None
    fx = frame.dropna(subset=["fx_depreciation"])
    fx = fx[fx["fx_depreciation"] > 0]
    largest_fx = fx.loc[fx["fx_depreciation"].idxmax()] if not fx.empty else None
    labor = frame.dropna(subset=["unemployment_change"])
    labor = labor[labor["unemployment_change"] > 0]
    largest_labor = (labor.loc[labor["unemployment_change"].idxmax()]
                     if not labor.empty else None)
    return {
        "landmark": spec["landmark"], "negative": negative, "denominator": denominator,
        "worst": worst, "largest_fx": largest_fx, "largest_labor": largest_labor,
        "growth_pairs": len(growth),
    }


def historical_wdi_sentence(snapshot: dict, crisis_key: str) -> str:
    """A concise reproducible statement, without presenting correlation as causation."""
    if not snapshot["denominator"]:
        return "World Bank GDP-growth observations are unavailable for the reference year."
    sentence = (f"In {snapshot['landmark']}, {snapshot['negative']} of "
                f"{snapshot['denominator']} selected economies with GDP data "
                "recorded negative annual growth.")
    worst = snapshot["worst"]
    if worst is not None:
        sentence += (f" Across the {year_span(CRISES[crisis_key]['shock'])} window, "
                     f"{worst['country']} had the lowest observed annual rate "
                     f"({worst['gdp_growth']:.2f}% in {int(worst['year'])}).")
    return sentence


def render_historical_context(full: pd.DataFrame, countries: list[str],
                              crisis_key: str, from_csv: bool = False) -> None:
    """Verified institutional history + independently calculated WDI evidence."""
    st.subheader("Historical context")
    origin = ("uploaded World Bank CSV backup" if from_csv else
              "World Bank World Development Indicators (API)")
    wdi_link = "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG"
    if crisis_key == ALL_TIME_KEY:
        st.caption("Three distinct events across the 1990–2025 period. Historical accounts "
                   "come from the cited institutions; figures below are calculated from WDI observations.")
        columns = st.columns(3, gap="medium")
        for col, (key, spec) in zip(columns, CRISES.items()):
            snapshot = historical_world_bank_snapshot(full, countries, key)
            with col:
                with st.container(border=True):
                    st.markdown(f"**{spec['title']}**")
                    st.caption(year_span(spec["shock"]))
                    st.write(spec["context"])
                    st.caption(historical_wdi_sentence(snapshot, key))
                    st.markdown(f"[Historical reference]({SOURCES[spec['source']]})")
        st.caption("WDI snapshot source: " + origin + ". Crisis snapshots use their fixed "
                   "historical windows for the selected economies, regardless of the year slider.")
        st.markdown(f"[World Bank GDP growth series]({wdi_link})")
    else:
        spec = CRISES[crisis_key]
        snap = historical_world_bank_snapshot(full, countries, crisis_key)
        with st.container(border=True):
            st.markdown(f"**{spec['title']} · {year_span(spec['shock'])}**")
            st.write(spec["context"])
            st.caption("Historical sequence: " + "  |  ".join(
                f"{year}: {event}" for year, event in spec["milestones"]
            ))
            st.markdown(f"**Economic mechanism to examine:** {spec['mechanism']}")
            st.caption(spec["question"])
            st.markdown("**World Bank data snapshot**")
            col1, col2, col3 = st.columns(3)
            col1.metric(f"GDP contractions in {spec['landmark']}",
                        f"{snap['negative']} / {snap['denominator']}" if snap['denominator'] else "N/A")
            worst = snap['worst']
            col2.metric("Lowest annual GDP growth",
                        f"{worst['gdp_growth']:.2f}%" if worst is not None else "N/A")
            if crisis_key == "asian":
                extra = snap['largest_fx']
                col3.metric("Largest annual FX depreciation",
                            f"{extra['fx_depreciation']:.2f}%" if extra is not None else "N/A")
                label = (f"{extra['country']} ({int(extra['year'])})" if extra is not None else "No usable FX movement")
            else:
                extra = snap['largest_labor']
                col3.metric("Largest annual unemployment rise",
                            f"{extra['unemployment_change']:.2f} pp" if extra is not None else "N/A")
                label = (f"{extra['country']} ({int(extra['year'])})" if extra is not None else "No usable unemployment change")
            if worst is not None:
                st.caption(f"Lowest growth: {worst['country']} ({int(worst['year'])}). "
                           f"Third metric: {label}. Each maximum is one observed economy-year, "
                           "not an average or a causal estimate.")
            for source_name in spec["background_sources"]:
                st.markdown(f"[Historical reference: {source_name}]({SOURCES[source_name]})")
            if crisis_key == "asian":
                st.markdown(
                    f"[World Bank historical report: Indonesia in Crisis (1998)]"
                    f"({SOURCES['World Bank (1998) — Indonesia in Crisis']})"
                )
        st.caption("WDI snapshot source: " + origin + ". Values come from source observations "
                   "at runtime, not a fixed historical table. Figures use the selected economies "
                   "and the complete historical shock window, even if the year slider differs. "
                   "Missing observations are excluded, not counted as zero.")
        st.markdown(f"[World Bank GDP growth series]({wdi_link})")


def _crisis_source_frame(full: pd.DataFrame, countries: list[str], crisis_key: str) -> pd.DataFrame:
    """Use the published fixed study window for each independent crisis dashboard."""
    window = CRISES[crisis_key]["view"]
    return full[full["country"].isin(countries) & full["year"].between(*window)].copy()


def _crisis_line_chart(frame: pd.DataFrame, crisis_key: str, field: str,
                       chart_title: str, y_label: str):
    spec = CRISES[crisis_key]
    observed = frame.dropna(subset=[field])
    if observed.empty:
        st.info(f"No {chart_title.lower()} observations for this event.")
        return
    fig = px.line(observed, x="year", y=field, color="country",
                  color_discrete_map=COLORS, markers=True,
                  title=chart_title,
                  labels={"year": "Year", field: y_label, "country": "Economy"})
    fig.add_vrect(x0=spec["shock"][0] - 0.5, x1=spec["shock"][1] + 0.5,
                  fillcolor="rgba(192,149,92,0.13)", line_width=0)
    fig.update_xaxes(dtick=max(1, (spec["view"][1] - spec["view"][0]) // 6))
    tune_fig(fig, y_label, height=330)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def render_event_dashboard(full: pd.DataFrame, countries: list[str], crisis_key: str):
    """A dedicated dashboard for one crisis: 4 KPIs, 4 trends, severity and recovery.

    Each event always uses its complete comparison window. The global year slider
    remains available for the general Overview; it cannot silently truncate this
    event's before/shock/after research calculations.
    """
    if crisis_key not in CRISES:
        st.info("Choose an individual crisis to open its economic dashboard.")
        return
    spec = CRISES[crisis_key]
    frame = _crisis_source_frame(full, countries, crisis_key)
    shock_all = frame[frame["year"].between(*spec["shock"])]
    stats = crisis_dashboard_stats(frame, full, crisis_key)
    shock, affected, worst = stats["shock"], stats["affected"], stats["worst"]
    unemployment_rises = shock_all.loc[shock_all["unemployment_change"] > 0].dropna(
        subset=["unemployment_change"]
    )
    largest_job_rise = (unemployment_rises.loc[unemployment_rises["unemployment_change"].idxmax()]
                        if not unemployment_rises.empty else None)
    mean_shock_inflation = shock_all["inflation"].mean()

    st.markdown(f"#### {spec['title']} — Economic Dashboard")
    st.caption(
        f"World Bank WDI | {year_span(spec['view'])} study window | "
        f"{year_span(spec['shock'])} shock years | "
        f"{spec['baseline']} constant-price GDP baseline. "
        "The event dashboard uses the full study window; the Overview uses the sidebar year slider."
    )
    st.markdown(f"**Research question:** {spec['question']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Economies with GDP contraction",
              f"{int(affected.lt(0).sum())} / {len(affected)}" if len(affected) else "N/A",
              help="Economies with at least one negative annual GDP growth rate within the shock window, "
                   "among those with reported GDP growth.")
    m2.metric("Lowest annual GDP growth",
              f"{float(worst['gdp_growth']):.2f}%" if worst is not None else "N/A",
              help="Minimum single-year GDP growth among selected economies during shock years.")
    m3.metric("Largest unemployment rise",
              f"{float(largest_job_rise['unemployment_change']):+.2f} pp"
              if largest_job_rise is not None else "N/A",
              help="Maximum positive year-over-year change in the unemployment rate during the shock years. "
                   "Measured in percentage points, not percent.")
    m4.metric("Mean shock-year inflation", format_pct(mean_shock_inflation),
              help="Unweighted average of available economy-year CPI inflation rates during the shock years. "
                   "Not cumulative inflation.")
    if worst is not None:
        st.caption(
            f"GDP trough: {worst['country']} in {int(worst['year'])}. "
            f"Unemployment increases and GDP troughs can occur in different years. "
            f"Real-GDP recovery observed in {stats['recovered_count']} of "
            f"{stats['recovery_eligible']} evaluable economies that fell below their baseline."
        )
    else:
        st.caption("Insufficient annual GDP growth data to identify a shock trough.")

    tab_activity, tab_prices, tab_outcomes = st.tabs([
        "Output & employment", "Inflation & currencies", "Shock severity & recovery"
    ])
    with tab_activity:
        left, right = st.columns(2, gap="large")
        with left:
            _crisis_line_chart(frame, crisis_key, "gdp_growth", "Real GDP growth", "Annual growth (%)")
        with right:
            _crisis_line_chart(frame, crisis_key, "unemployment_change",
                               "Change in unemployment rate", "Annual change (percentage points)")
        st.caption("Annual unemployment changes use the immediately preceding year's WDI observation "
                   "from the full 1990–2025 dataset. The shaded region marks the shock window.")
    with tab_prices:
        left, right = st.columns(2, gap="large")
        with left:
            _crisis_line_chart(frame, crisis_key, "inflation", "Consumer price inflation", "Annual CPI inflation (%)")
        with right:
            _crisis_line_chart(frame, crisis_key, "fx_depreciation",
                               "Official exchange-rate change", "Annual change (%)")
        st.caption("Positive exchange-rate changes mean more local-currency units per US dollar "
                   "(depreciation). Annual averages hide intrayear swings, and exchange-rate "
                   "regimes differ across economies. A missing observation is not zero.")
    with tab_outcomes:
        left, right = st.columns([1, 1.05], gap="large")
        with left:
            st.markdown("**Five deepest annual GDP contractions during the shock**")
            ranking = top_five(shock, "Deepest GDP contraction")
            if ranking.empty:
                st.info("No negative GDP-growth observation is available in the selected shock window.")
            else:
                plot = px.bar(ranking, x="score", y="country", orientation="h",
                              color="country", color_discrete_map=COLORS,
                              hover_data={"year_label": True},
                              title=f"Deepest contractions | {year_span(spec['shock'])}",
                              labels={"score": "Annual GDP growth (%)", "country": "Economy"})
                plot.update_yaxes(categoryorder="array",
                                  categoryarray=ranking["country"].iloc[::-1].tolist())
                plot.update_traces(texttemplate="%{x:.2f}%", textposition="outside", cliponaxis=False)
                tune_fig(plot, height=330)
                plot.update_layout(showlegend=False)
                st.plotly_chart(plot, use_container_width=True, theme="streamlit")
                st.caption("One worst annual growth observation per economy; not cumulative output loss.")
        with right:
            st.markdown(f"**Real GDP recovery | {spec['baseline']} = 100**")
            statuses = stats["recovery_table"]
            if statuses.empty:
                st.info("Recovery observations are unavailable.")
            else:
                cols = ["Economy", "Recovery status", "First observed year", "Years after trough"]
                st.dataframe(statuses[cols], hide_index=True, use_container_width=True)
                st.caption("First observed return to pre-shock GDP after a measured trough; "
                           "not a claim of sustained recovery. Missing dates remain missing.")

    # Separate sourced history from what the WDI observations actually measure.
    if worst is not None:
        contraction_count = int(affected.lt(0).sum())
        st.markdown("**Economic interpretation**")
        interpretation = (
            f"Of {len(affected)} selected economies with GDP data in the shock window, "
            f"{contraction_count} recorded at least one year of declining real GDP. "
            f"The lowest annual growth was {float(worst['gdp_growth']):.2f}% "
            f"in {worst['country']} ({int(worst['year'])}). "
        )
        if largest_job_rise is not None:
            interpretation += (
                f"The largest observed annual rise in unemployment was "
                f"{float(largest_job_rise['unemployment_change']):.2f} percentage points "
                f"in {largest_job_rise['country']} ({int(largest_job_rise['year'])}). "
            )
        if pd.notna(mean_shock_inflation):
            interpretation += (
                f"The unweighted mean inflation rate across reported shock-window "
                f"economy-years was {mean_shock_inflation:.2f}%. "
            )
        interpretation += (
            "These indicators need not reach their worst values in the same year. "
            "Differences in trade exposure, financial conditions, domestic activity, "
            "and policy responses may help explain the patterns, but this descriptive "
            "dashboard cannot isolate causal effects."
        )
        st.write(interpretation)
    st.caption(
        f"Shock-window data coverage: GDP growth {int(shock_all['gdp_growth'].notna().sum())}/{len(shock_all)}, "
        f"unemployment change {int(shock_all['unemployment_change'].notna().sum())}/{len(shock_all)}, "
        f"CPI inflation {int(shock_all['inflation'].notna().sum())}/{len(shock_all)} economy-years. "
        "Missing observations are excluded rather than set to zero."
    )
    st.caption("Source: World Bank, World Development Indicators (GDP growth, ILO-modeled unemployment, "
               "CPI inflation, constant-price GDP and official exchange rates). "
               "Historical explanation and citations appear in Historical Context above.")



def render_full_dashboard(full: pd.DataFrame, selected: pd.DataFrame,
                          countries: list[str], years: tuple[int, int],
                          crisis_key: str) -> None:
    """Show all four indicators, contraction rankings and crisis comparisons together.

    All Time follows the sidebar's year range. Each named crisis uses its
    fixed study window, without changing the four original analysis modes.
    """
    event_mode = crisis_key in CRISES
    spec = CRISES[crisis_key] if event_mode else None
    frame = (_crisis_source_frame(full, countries, crisis_key)
             if event_mode else selected)
    frame = frame.sort_values(["country", "year"]).copy()
    window = spec["view"] if event_mode else years

    st.subheader("Full Economic Dashboard")
    st.caption(
        f"{'Full Graph Dashboard' if not event_mode else CRISIS_NAMES[crisis_key]}  |  {year_span(window)}  |  "
        f"{len(countries)} selected economies. "
        + ("All panels use this crisis's full historical study window."
           if event_mode else "All panels follow the sidebar year range.")
    )

    # Line charts: every observation comes from the World Bank WDI source.
    # No interpolation of missing observations, smoothing, or fitted trend lines.
    def trend_panel(field: str, title: str, unit: str) -> None:
        observed = frame.dropna(subset=[field])
        if observed.empty:
            st.info(f"No available {title.lower()} values for this selection.")
            return
        fig = px.line(
            observed, x="year", y=field, color="country", markers=True,
            color_discrete_map=COLORS, title=title,
            labels={"year": "Year", field: unit, "country": "Economy"},
        )
        if event_mode:
            fig.add_vrect(x0=spec["shock"][0] - 0.5,
                          x1=spec["shock"][1] + 0.5,
                          fillcolor="rgba(192,149,92,0.12)", line_width=0)
        tune_fig(fig, unit, height=370)
        fig.update_xaxes(dtick=max(1, (window[1] - window[0]) // 6))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.22,
                        xanchor="left", x=0, title=None, font=dict(size=10)),
            margin=dict(l=12, r=12, t=50, b=110),
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    row1_left, row1_right = st.columns(2, gap="large")
    with row1_left:
        trend_panel("gdp_growth", "Real GDP growth over time", "Annual GDP growth (%)")
    with row1_right:
        trend_panel("unemployment", "Unemployment rate over time", "Unemployment rate (%)")

    row2_left, row2_right = st.columns(2, gap="large")
    with row2_left:
        by_country = (frame.groupby("country", as_index=False)["inflation"].mean()
                      .dropna(subset=["inflation"]).sort_values("inflation"))
        if by_country.empty:
            st.info("No annual inflation observations in this selection.")
        else:
            fig = px.bar(
                by_country, x="country", y="inflation", color="country",
                color_discrete_map=COLORS,
                title="Average annual inflation by economy",
                labels={"country": "Economy", "inflation": "Mean inflation (%)"},
            )
            tune_fig(fig, "Mean annual inflation (%)", height=370)
            fig.update_layout(showlegend=False, margin=dict(l=12,r=12,t=50,b=95))
            fig.update_xaxes(tickangle=-35)
            fig.update_traces(hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.caption("Arithmetic mean of reported annual CPI inflation rates; not cumulative inflation.")
    with row2_right:
        trend_panel("fx_depreciation", "Annual exchange-rate change", "LCU per US$ change (%)")
        st.caption("Positive values mean depreciation against the US dollar. "
                   "Currency regimes differ; annual averages may hide intrayear swings.")

    st.divider()
    bottom_left, bottom_right = st.columns([1.0, 1.15], gap="large")
    with bottom_left:
        st.markdown("**Top 5: deepest annual GDP contractions**")
        severity = frame[frame["year"].between(*spec["shock"])] if event_mode else frame
        ranked = top_five(severity, "Deepest GDP contraction")
        if ranked.empty:
            st.info("No negative annual GDP growth is reported for this comparison period.")
        else:
            bar = px.bar(
                ranked, x="score", y="country", orientation="h", color="country",
                color_discrete_map=COLORS,
                title=(f"Shock years: {year_span(spec['shock'])}" if event_mode
                       else f"Selected years: {year_span(years)}"),
                hover_data={"year_label": True},
                labels={"score": "GDP growth (%)", "country": "Economy"},
            )
            bar.update_traces(texttemplate="%{x:.2f}%", textposition="outside", cliponaxis=False)
            bar.update_yaxes(categoryorder="array", categoryarray=ranked["country"].iloc[::-1].tolist())
            tune_fig(bar, "", height=350)
            bar.update_layout(showlegend=False, margin=dict(l=12,r=28,t=50,b=30))
            st.plotly_chart(bar, use_container_width=True, theme="streamlit")
            st.caption("One lowest annual growth rate per economy; not cumulative GDP loss.")

    with bottom_right:
        if event_mode:
            st.markdown(f"**Real GDP recovery | {spec['baseline']} = 100**")
            _, recovery = recovery_data(full, frame, crisis_key)
            if recovery.empty:
                st.info("Not enough constant-price GDP observations to evaluate recovery.")
            else:
                st.dataframe(
                    recovery[["Economy", "Recovery status", "First observed year"]],
                    hide_index=True, use_container_width=True,
                )
                st.caption("Recovery means the first observed return to pre-crisis real GDP "
                           "after a measured trough, not necessarily a sustained recovery.")
        else:
            st.markdown("**Three-crisis comparison**")
            comparisons = []
            for key, event in CRISES.items():
                scope = _crisis_source_frame(full, countries, key)
                metrics = crisis_dashboard_stats(scope, full, key)
                worst = metrics["worst"]
                comparisons.append({
                    "Crisis": event["short"],
                    "Lowest annual GDP growth (%)": (
                        round(float(worst["gdp_growth"]), 2) if worst is not None else np.nan),
                    "Trough economy": str(worst["country"]) if worst is not None else "N/A",
                    "Economies contracting": int(metrics["affected"].lt(0).sum()),
                })
            st.dataframe(pd.DataFrame(comparisons), hide_index=True,
                         use_container_width=True,
                         column_config={"Lowest annual GDP growth (%)":
                                        st.column_config.NumberColumn(format="%.2f%%")})
            st.caption("Historical shock windows are fixed (1997–99, 2008–10, 2020–21) "
                       "and do not follow the selected-year slider. "
                       "Each trough is one observed economy-year, not a regional average.")

    with st.expander("Open the full economy-year heatmap", expanded=False):
        render_heatmap(frame, countries, window)


def render_all_time_overview(full: pd.DataFrame, selected: pd.DataFrame,
                             countries: list[str], years: tuple[int, int]):
    """Show the full-period view without treating it as an additional crisis."""
    st.subheader(f"All-time economic overview | {years[0]}–{years[1]}")
    st.caption("A long-run perspective on eight Asian economies and three major economic shocks. "
               "Country and year filters still apply to the trends, rankings, and report.")
    observed = selected.dropna(subset=["gdp_growth"])
    worst = observed.loc[observed["gdp_growth"].idxmin()] if not observed.empty else None
    m1, m2, m3 = st.columns(3)
    m1.metric("Economies with GDP data", f"{observed['country'].nunique()} / {len(countries)}")
    m2.metric("Lowest observed annual growth",
              f"{worst['gdp_growth']:.2f}%" if worst is not None else "N/A")
    m3.metric("Average annual GDP growth", format_pct(observed["gdp_growth"].mean()))
    if worst is not None:
        st.caption(f"The lowest observed growth was in {worst['country']} ({int(worst['year'])}). "
                   "Mean growth rates weight available economy-year observations equally.")
    st.markdown("**The three historical crisis windows**")
    st.caption("The heatmap compares each economy's weakest GDP-growth year in 1997–1999, "
               "2008–2010, and 2020–2021. Empty cells indicate unavailable observations.")
    render_three_crisis_comparison(selected, countries)
    st.caption("Select an individual crisis above to focus the entire report on that event. "
               "The three dashboards below remain available for side-by-side exploration.")
    st.markdown("### Individual crisis dashboards")
    st.caption("Explore any of the three dedicated crisis dashboards without leaving All Time. "
               "Each uses its complete historical comparison window. Your selected economies "
               "apply, while All Time's dataset and Executive Summary exports remain unchanged.")
    focus = st.selectbox("Open crisis dashboard", options=list(CRISES),
                         format_func=lambda key: CRISES[key]["short"],
                         key="all_time_dashboard_focus")
    render_event_dashboard(full, countries, focus)


def render_crisis_explorer(full: pd.DataFrame, selected: pd.DataFrame,
                           countries: list[str], years: tuple[int, int], crisis_key: str):
    """Dedicated event view: comparable dashboard followed by detailed methods."""
    if crisis_key not in CRISES:
        st.info("Select an economic crisis to explore its impact.")
        return
    spec = CRISES[crisis_key]
    event_frame = _crisis_source_frame(full, countries, crisis_key)
    render_event_dashboard(full, countries, crisis_key)
    st.divider()
    st.markdown("### Detailed crisis comparisons")
    st.caption(f"Each tab uses the complete {year_span(spec['view'])} study window, "
               "consistent with this crisis's dedicated CSV and Executive Summary. "
               "The Overview charts continue to follow your custom year slider.")
    tabs = st.tabs(["Before vs. after", "GDP recovery index", "Currency pressure", "Compare all three"])
    with tabs[0]:
        render_crisis_comparison(event_frame, crisis_key)
    with tabs[1]:
        render_recovery(full, event_frame, crisis_key)
    with tabs[2]:
        render_currency_pressure(event_frame, crisis_key)
    with tabs[3]:
        st.caption("The cross-crisis heatmap always includes all three shock windows "
                   "for the selected economies, independently of the year slider.")
        all_selected = full[full["country"].isin(countries)]
        render_three_crisis_comparison(all_selected, countries)


def render_rankings(df: pd.DataFrame, crisis_key: str):
    st.subheader("Top 5 economic rankings")
    left, right = st.columns([1.2, .9])
    with left:
        choice = st.selectbox("Rank by", list(RANK_OPTIONS), key="rank_choice")
    with right:
        if crisis_key == ALL_TIME_KEY:
            st.caption("Ranking period: all selected years")
            scope = "Selected years"
        else:
            scope = st.selectbox("Ranking period", ["Selected years", "Active crisis shock window"],
                                 key="rank_scope")
    subset = (df[df["year"].between(*CRISES[crisis_key]["shock"])]
              if scope == "Active crisis shock window" and crisis_key in CRISES else df)
    ranked = top_five(subset, choice)
    st.caption("Each economy appears once. Annual metrics use each economy's most extreme "
               "available year; the average-growth option uses the simple mean of available "
               "rates. Rankings respond to your selected economies and years.")
    if ranked.empty:
        st.info("No qualifying observations for this metric and period. "
                "Try another period or ranking metric.")
        return
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        bars = px.bar(ranked, x="score", y="country", orientation="h", color="country",
                      color_discrete_map=COLORS, text="score", title=choice,
                      labels={"score": RANK_OPTIONS[choice][3], "country": "Economy"},
                      hover_data={"year_label": True})
        bars.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
        bars.update_yaxes(categoryorder="array", categoryarray=ranked["country"].iloc[::-1].tolist())
        tune_fig(bars, height=max(320, 95 + 49 * len(ranked)))
        bars.update_layout(showlegend=False)
        bars.add_vline(x=0, line_color="#A4B0BE", line_width=1)
        st.plotly_chart(bars, use_container_width=True, theme="streamlit")
    with right:
        view = ranked.rename(columns={"rank": "Rank", "country": "Economy",
                                      "year_label": "Year", "score": "Value", "unit": "Unit"})
        st.dataframe(view, hide_index=True, use_container_width=True,
                     column_config={"Value": st.column_config.NumberColumn(format="%.2f")})


def sidebar_country_selector() -> list[str]:
    """Keep sidebar short: country checkboxes live in a popover, not eight large tags."""
    for name, code in COUNTRIES.items():
        st.session_state.setdefault(f"economy_{code}", True)
    selected_count = sum(bool(st.session_state[f"economy_{code}"]) for code in COUNTRIES.values())
    # A normal Streamlit element avoids the negative-margin overlap of custom HTML labels.
    st.markdown("**Economies**")
    with st.popover(f"{selected_count} of {len(COUNTRIES)} economies selected", use_container_width=True):
        left, right = st.columns(2)
        if left.button("Select all", key="all_countries", use_container_width=True):
            for code in COUNTRIES.values():
                st.session_state[f"economy_{code}"] = True
        if right.button("Core five", key="core_countries", use_container_width=True):
            for name, code in COUNTRIES.items():
                st.session_state[f"economy_{code}"] = name in CORE_CRISIS
        for name, code in COUNTRIES.items():
            st.checkbox(name, key=f"economy_{code}")
        st.caption("Core five = Indonesia, Malaysia, Thailand, Philippines and South Korea.")
    return [name for name, code in COUNTRIES.items() if st.session_state[f"economy_{code}"]]


def render_labor_analysis(data: pd.DataFrame):
    """Okun-style descriptive results at three adjustment levels."""
    st.subheader("Growth and changes in unemployment")
    stats = okun_statistics(data)
    cards = st.columns(4)
    cards[0].metric("Pooled correlation", f"{stats['pooled_r']:+.2f}" if stats["pooled_r"] is not None else "N/A")
    cards[1].metric("Country-adjusted slope", f"{stats['slope']:+.3f}" if stats["slope"] is not None else "N/A")
    cards[2].metric("Country + year slope", f"{stats['two_way_slope']:+.3f}" if stats["two_way_slope"] is not None else "N/A")
    cards[3].metric("Matched economy-years", str(stats["n"]))
    st.caption("Annual Δ unemployment = unemployment rate this year − unemployment rate last year "
               "(percentage points). Each slope is the change in unemployment (pp) associated "
               "with +1 percentage point of GDP growth; these are descriptive regressions, not causal effects.")
    raw_tab, within_tab, two_way_tab = st.tabs(["Observed", "Country adjusted", "Country + year adjusted"])
    pairs = stats["pairs"]
    with raw_tab:
        if pairs.empty:
            st.info("No complete GDP-growth and unemployment-change pairs in this selection.")
        else:
            fig = px.scatter(pairs, x="gdp_growth", y="unemployment_change", color="country",
                             color_discrete_map=COLORS, hover_data=["year"],
                             title="GDP growth and changes in unemployment",
                             labels={"gdp_growth": "Real GDP growth (%)",
                                     "unemployment_change": "Unemployment change (pp)", "country": "Economy"})
            fig.add_hline(y=0, line_dash="dot", line_color="#8A98A6", line_width=1)
            tune_fig(fig, "Unemployment change (pp)")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        st.caption("Raw pooled observations mix long-run country differences and shocks common to a given year.")
    with within_tab:
        if stats["slope"] is None:
            st.info("Not enough within-economy variation in this selected sample.")
        else:
            within_pairs = stats["within_pairs"]
            fig = px.scatter(within_pairs, x="growth_deviation", y="unemployment_deviation", color="country",
                             color_discrete_map=COLORS, hover_data=["year"],
                             title="Deviations from each economy's mean",
                             labels={"growth_deviation": "GDP growth deviation (pp)",
                                     "unemployment_deviation": "Unemployment change deviation (pp)",
                                     "country": "Economy"})
            xx = np.linspace(float(within_pairs["growth_deviation"].min()),
                             float(within_pairs["growth_deviation"].max()), 100)
            fig.add_trace(go.Scatter(x=xx, y=stats["slope"] * xx, mode="lines",
                                     name="Descriptive fitted slope", line=dict(color="#99B6C9", dash="dash")))
            tune_fig(fig, "Unemployment change, demeaned (pp)")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.caption(f"Country-adjusted slope {stats['slope']:+.3f} pp per +1 pp GDP growth; "
                       "within-economy correlation "
                       f"{stats['within_r']:+.2f}. Common year effects remain uncontrolled.")
    with two_way_tab:
        if stats["two_way_slope"] is None:
            st.info("Insufficient variation or degrees of freedom to adjust for both countries and years.")
        else:
            partial = stats["two_way_pairs"]
            fig = px.scatter(partial, x="growth_residual", y="unemployment_residual", color="country",
                             color_discrete_map=COLORS, hover_data=["year"],
                             title="Country- and year-adjusted relationship",
                             labels={"growth_residual": "GDP growth after adjustment (pp)",
                                     "unemployment_residual": "Unemployment change after adjustment (pp)",
                                     "country": "Economy"})
            xx = np.linspace(float(partial["growth_residual"].min()),
                             float(partial["growth_residual"].max()), 100)
            fig.add_trace(go.Scatter(x=xx, y=stats["two_way_slope"] * xx, mode="lines",
                                     name="Descriptive adjusted fit", line=dict(color="#99B6C9", dash="dash")))
            tune_fig(fig, "Adjusted unemployment change (pp)")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            slope = stats["two_way_slope"]
            if slope < -0.02:
                assessment = "The estimated association is inverse, as commonly expected in an Okun-style comparison."
            elif slope > 0.02:
                assessment = "The estimated association is positive in this sample; the basic inverse pattern is not present."
            else:
                assessment = "The linear association is small after these adjustments."
            st.caption(f"Adjusted slope: {slope:+.3f} pp of unemployment change per +1 pp GDP growth. "
                       + assessment + " Adding country and calendar-year indicators removes their average differences, "
                       "but cannot address every omitted factor or establish causality.")
    with st.expander("Cross-economy sensitivity and method"):
        per_country = stats["country_table"]
        if not per_country.empty:
            st.dataframe(per_country, hide_index=True, use_container_width=True,
                         column_config={"Correlation (r)": st.column_config.NumberColumn(format="%.2f"),
                                        "Slope (pp per 1pp GDP)": st.column_config.NumberColumn(format="%.3f")})
            st.caption("Separate economy-level slopes can differ in sign or size. Each requires at least six "
                       "paired years; no significance tests or causal claims are made.")
        else:
            st.info("Not enough paired years for separate economy-level slope estimates.")
        st.markdown("**Method:** Complete pairs only; unemployment change uses consecutive years from the "
                    "unfiltered source dataset. Country adjustment removes each economy's mean. The two-way "
                    "calculation also removes calendar-year means using indicator variables. Without controls "
                    "for labor-market institutions, productivity and other factors, it is not a causal estimate. "
                    f"See [Federal Reserve Bank of St. Louis (2013)]({SOURCES['Federal Reserve Bank of St. Louis (2013) — Output and Unemployment']}).")

def render_inflation_analysis(data: pd.DataFrame):
    st.subheader("Inflation across economies")
    means = (data.groupby("country", as_index=False).agg(
        inflation=("inflation", "mean"), years=("inflation", "count"))
        .dropna(subset=["inflation"]).sort_values("inflation"))
    if means.empty:
        st.info("No inflation observations are available in the selected period.")
        return
    fig = px.bar(means, x="country", y="inflation", color="country",
                 color_discrete_map=COLORS, title="Average annual CPI inflation",
                 labels={"country": "Economy", "inflation": "Average annual inflation (%)"},
                 hover_data=["years"])
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", cliponaxis=False)
    tune_fig(fig, "Annual average CPI inflation (%)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption("These are arithmetic means of available annual inflation rates—not cumulative price increases or "
               "population-weighted regional inflation. Hover to see each economy's available-year count.")


def render_data_checks(data: pd.DataFrame):
    """Auditable source-data coverage, consistency checks and provenance."""
    st.subheader("Data quality and methodology")
    checks, comparison = check_data_quality(data)
    a, b, c, d = st.columns(4)
    a.metric("Complete core rows", f"{checks['complete_three']} / {len(data)}")
    b.metric("Duplicate country-years", str(checks["duplicates"]))
    c.metric("GDP growth cross-checks", str(checks["growth_comparisons"]))
    d.metric("Differences > 0.5 pp", str(checks["large_growth_differences"]))
    st.caption("Completeness is counted for GDP growth, unemployment, and inflation together; "
               "optional GDP-level and currency observations are checked separately.")
    if checks["duplicates"] or checks["missing_identifiers"]:
        st.error("Identifier validation failed. Do not use this dataset before correcting the source file.")
    if checks["large_growth_differences"]:
        st.warning("Some reported GDP-growth rates and rates calculated from constant-price GDP "
                   "differ by over 0.5 percentage points. These are review flags, not automatic errors; "
                   "source vintages, breaks or methods can differ.")
    flags = {k: v for k, v in checks["plausibility_flags"].items() if v}
    if flags:
        st.warning("Values to inspect (not removed): " + "; ".join(f"{k}: {v}" for k, v in flags.items()))
    if not checks["duplicates"] and not checks["missing_identifiers"] and not flags and not checks["large_growth_differences"]:
        st.caption("No identifier problems or flagged values were detected by these specified checks.")
    st.markdown("**Coverage by economy and indicator**")
    coverage = indicator_coverage(data)
    st.dataframe(coverage, hide_index=True, use_container_width=True,
                 column_config={field: st.column_config.NumberColumn(format="%.1f%%")
                                for field in coverage.columns if field not in ("Economy", "Years selected")})
    st.caption("Coverage percentages count observed WDI cells; ILO unemployment is itself a modeled series "
               "and should not be confused with complete direct-survey coverage. "
               "Zero missing cells does not prove the underlying measurements are error-free.")
    with st.expander("Source consistency: reported vs. calculated GDP growth"):
        st.markdown("Validation: 100 × (constant-price GDP this year / preceding year's GDP − 1). "
                    "Only consecutive positive GDP levels are compared. This check uses the complete source "
                    "grid **before** the year filter is applied. It never overwrites the WDI growth series.")
        if comparison.empty:
            st.info("No complete pairs for the GDP consistency check.")
        else:
            table = comparison.assign(abs_gap=comparison["Difference (pp)"].abs())
            st.dataframe(table.nlargest(20, "abs_gap").drop(columns="abs_gap"),
                         hide_index=True, use_container_width=True)
            median = checks["median_growth_difference_pp"]
            st.caption(f"Median absolute difference: {median:.3f} percentage points across "
                       f"{checks['growth_comparisons']} comparable economy-years. A difference of 0.5 pp "
                       "is a diagnostic review threshold, not an official World Bank tolerance.")
    with st.expander("Variable definitions, transformations and interpretation"):
        st.markdown("**Primary annual WDI series:** GDP growth (`NY.GDP.MKTP.KD.ZG`), modeled ILO "
                    "unemployment (`SL.UEM.TOTL.ZS`), CPI inflation (`FP.CPI.TOTL.ZG`). "
                    "**Supporting series:** constant-price GDP (`NY.GDP.MKTP.KD`) and official annual "
                    "exchange rate in local currency per USD (`PA.NUS.FCRF`).")
        st.markdown("**Processing:** Left join on ISO3 and year to preserve the 8 × 36 country-year grid. "
                    "Convert numeric observations, keep missing values as missing, and check for duplicate "
                    "identifiers. Annual unemployment changes are **percentage-point differences**; exchange-rate "
                    "depreciation is the **percent change in LCU/USD** when adjacent observations are positive. "
                    "No interpolation or zero substitution. Country/year adjustments use complete labor-growth pairs.")
        st.markdown("**Interpretation:** Pre/post means are unweighted annual rates and only compared when full "
                    "three-year windows are observed. Recovery is the first *observed* year that real GDP again "
                    "reaches the pre-shock benchmark, not proof of a sustained recovery. Adjusted regression "
                    "slopes are conditional descriptive associations—not estimated causal effects.")
    st.markdown("**Official sources and historical background**")
    for title, link in SOURCES.items():
        st.markdown(f"- [{title}]({link})")
    current = data[data["year"].eq(2025)]
    if not current.empty:
        incomplete = current[current[list(SOURCE_FIELDS)].isna().any(axis=1)]
        if not incomplete.empty:
            st.info(f"In 2025, {len(incomplete)} of {len(current)} selected economies have at least one "
                    "missing source indicator. The calculations use only observed values; missing is not zero.")
    st.caption("The API returns World Bank WDI observations and may revise previous years. "
               "Coverage and denominators vary by indicator and year; recheck the latest year before submitting.")

def main():
    st.set_page_config(page_title="MacroLens Asia | 1990-2025", page_icon="📊", layout="wide",
                       initial_sidebar_state="expanded")
    # The UI follows the user's Streamlit Settings > Appearance choice.
    # Surface, border and text colors come from the active app/sidebar theme;
    # avoid forcing white controls or dark labels in injected CSS.
    st.markdown("""
    <style>
      .block-container {padding-top:1.45rem; max-width:1390px}
      h1 {letter-spacing:-.035em}
      [data-testid="stMetric"] {
          border:1px solid var(--st-border-color, rgba(125,135,145,.30));
          border-radius:13px;
          padding:13px 17px;
          background:var(--st-secondary-background-color, transparent);
      }
      /* Clear, restrained editorial typography across both Streamlit themes. */
      .block-container h1 {
          font-size: clamp(2.2rem, 3.2vw, 3rem);
          font-weight: 780;
          letter-spacing: -.045em;
          line-height: 1.14;
      }
      .ml-crisis-title {
          font-size: clamp(1.4rem, 2.2vw, 1.65rem);
          font-weight: 750;
          letter-spacing: -.025em;
          line-height: 1.25;
          margin: 0;
      }
      .ml-crisis-help {
          font-size: .875rem;
          line-height: 1.5;
          opacity: .75;
          margin: .12rem 0 .15rem 0;
          max-width: 70rem;
      }
      .st-key-full_graph_toggle button {
          font-size: .9rem;
          font-weight: 650;
          letter-spacing: -.005em;
          min-height: 2.35rem;
          border-radius: .55rem;
          padding: .38rem .9rem;
      }
      /* Compact sidebar brand without hardcoding light/dark colors. */
      [data-testid="stSidebar"] .ml-brand {
          color: inherit;
          margin: 0;
          padding: 0;
      }
      [data-testid="stSidebar"] .ml-brand-title {
          font-family: inherit;
          font-size: 2.6rem;
          font-weight: 800;
          letter-spacing: -0.055em;
          line-height: 1.04;
      }
      [data-testid="stSidebar"] .ml-brand-credit {
          font-size: 0.83rem;
          opacity: .73;
          font-weight: 400;
          margin-top: 0.24rem;
      }
      [data-testid="stSidebar"] .ml-brand-rule {
          border-top: 1px solid var(--st-border-color, rgba(125,135,145,.3));
          margin: .75rem 0 .45rem 0;
      }
      /* Sidebar filter labels use native Streamlit spacing.
         Do not override stVerticalBlock gaps or apply negative margins. */
      [data-testid="stSidebar"] .ml-explore-title {
          font-size: 1.3rem;
          font-weight: 700;
          line-height: 1.2;
          margin: 0;
      }
      /* Keep Streamlit's native hide-sidebar arrow visible and clickable. */
      [data-testid="stSidebarCollapseButton"] {
          visibility: visible !important;
          opacity: 1 !important;
      }
    </style>""", unsafe_allow_html=True)
    st.caption("WORLD BANK / WORLD DEVELOPMENT INDICATORS  ·  MACROECONOMIC RESEARCH DASHBOARD")
    st.title("MacroLens Asia")
    st.markdown("**Eight Asian economies · 1990-2025 · Growth, inflation, employment and financial crises**")
    st.caption("Economic research dashboard · by NV-215")

    if "selected_years" not in st.session_state:
        st.session_state["selected_years"] = PERIOD_PRESETS["Full period (1990–2025)"]
    if "period_preset" not in st.session_state:
        st.session_state["period_preset"] = "Full period (1990–2025)"
    valid_crises = (*CRISES, ALL_TIME_KEY)
    # Safely migrate any session state left by the old five-button version.
    if st.session_state.get("active_crisis") not in valid_crises:
        prior = st.session_state.get("primary_crisis")
        st.session_state["active_crisis"] = prior if prior in valid_crises else ALL_TIME_KEY
    st.session_state["last_valid_crisis"] = st.session_state["active_crisis"]
    if st.session_state.get("primary_crisis") not in valid_crises:
        st.session_state["primary_crisis"] = st.session_state["active_crisis"]
    if "full_graph_open" not in st.session_state:
        st.session_state["full_graph_open"] = False

    # Sidebar order: compact brand, exploration, explanatory notes, data controls.
    # Uses Streamlit's native sidebar toggle (the arrow at the top-right).
    with st.sidebar:
        st.markdown("""
        <div class="ml-brand">
          <div class="ml-brand-title">MacroLens</div>
          <div class="ml-brand-credit">by NV-215</div>
          <div class="ml-brand-rule"></div>
          <div class="ml-explore-title">Explore</div>
        </div>
        """, unsafe_allow_html=True)
        countries = sidebar_country_selector()
        st.selectbox("Time period", options=list(PERIOD_PRESETS),
                     key="period_preset", on_change=period_changed)
        years = st.slider("Year range", min_value=START_YEAR, max_value=END_YEAR,
                          key="selected_years", step=1, on_change=years_changed)
        st.divider()
        with st.expander("Indicator notes", expanded=False):
            st.caption("GDP: real annual growth (%). Unemployment: ILO modeled estimate (% of labor force). "
                       "Annual unemployment changes: percentage points. Exchange-rate changes: annual change "
                       "in local currency per US dollar; positive means depreciation.")
            st.markdown("[World Bank WDI](https://data.worldbank.org/) · Cached for 24 hours")
        with st.expander("Data source / CSV", expanded=False):
            uploaded = st.file_uploader("Load cleaned CSV", type=["csv"],
                                        help="Optional offline dataset exported from this dashboard.")
            refresh = st.button("Refresh World Bank data", use_container_width=True)
        st.caption(f"{len(countries)} economies · {years[0]}–{years[1]}")

    if refresh:
        fetch_all_data.clear()
        fetch_indicator.clear()
        st.rerun()
    if uploaded is not None:
        try:
            full, failed = read_backup(uploaded), []
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Unable to load the CSV: {exc}")
            st.stop()
        st.info("Showing your saved World Bank dataset. Its download date may differ from today.")
    else:
        try:
            with st.spinner("Loading World Bank observations (cached by indicator)..."):
                full, failed = fetch_all_data()
        except RuntimeError as exc:
            st.error(str(exc))
            st.info("The World Bank API can be slow. Try refreshing or upload a cleaned CSV.")
            st.stop()
        if failed:
            st.warning("The main dashboard is available, but these optional indicators "
                       "could not be downloaded: " + ", ".join(failed)
                       + ". Their crisis charts may be incomplete.")

    if not countries:
        st.warning("Select at least one economy to continue.")
        st.stop()
    data = full[full["country"].isin(countries) & full["year"].between(*years)].copy()
    if not data[list(CORE_FIELDS)].notna().any().any():
        st.warning("No valid GDP, employment or inflation values are available for this selection.")
        st.stop()

    st.markdown('<div class="ml-crisis-title">Compare economic crises</div>',
                unsafe_allow_html=True)
    st.segmented_control(
        "Crisis analysis", options=[*CRISES, ALL_TIME_KEY],
        format_func=lambda key: CRISIS_NAMES[key],
        key="primary_crisis", on_change=primary_crisis_changed,
        label_visibility="collapsed",
    )
    active_crisis = _valid_crisis_key(st.session_state.get("active_crisis"))
    st.markdown(
        '<p class="ml-crisis-help">Choose an event or All Time. The Overview follows the sidebar year range; '
        'each named crisis uses its fixed historical study window. All Time keeps selected-year exports.</p>',
        unsafe_allow_html=True,
    )

    # One action, located beneath the explanation: the current crisis determines
    # which complete set of graphs to show. No fifth crisis or second selector.
    button_col, _ = st.columns([1.25, 3.75], gap="small")
    with button_col:
        if st.button("Full Graph Dashboard", key="full_graph_toggle",
                     use_container_width=True, type="secondary"):
            st.session_state["full_graph_open"] = not st.session_state["full_graph_open"]
    graph_open = bool(st.session_state["full_graph_open"])

    if graph_open:
        if active_crisis in CRISES:
            dashboard_years = CRISES[active_crisis]["view"]
            dashboard_data = _crisis_source_frame(full, countries, active_crisis)
            st.caption(f"Full graphs · {CRISES[active_crisis]['title']} · "
                       f"{year_span(dashboard_years)}. Select another crisis above to change the graphs; "
                       "click Full Graph Dashboard again to return to detailed analysis.")
        else:
            dashboard_data, dashboard_years = data, years
            st.caption(f"Full graphs · All Time · {year_span(years)}. "
                       "Uses the sidebar year range; click Full Graph Dashboard again "
                       "to return to detailed analysis.")
    else:
        dashboard_data, dashboard_years = data, years
        st.divider()
        render_historical_context(full, countries, active_crisis,
                                  from_csv=(uploaded is not None))
    st.divider()

    st.markdown("**Selected-period macroeconomic indicators**")
    st.caption(f"Selected: {len(countries)} economies · {year_span(dashboard_years)} · "
               f"{len(dashboard_data)} possible economy-year observations")
    okun = okun_statistics(dashboard_data)
    corr, complete_pairs = okun["pooled_r"], okun["n"]
    a, b, c, d = st.columns(4)
    a.metric("Average GDP growth", format_pct(dashboard_data["gdp_growth"].mean()))
    b.metric("Average unemployment", format_pct(dashboard_data["unemployment"].mean()))
    c.metric("Average inflation", format_pct(dashboard_data["inflation"].mean()))
    d.metric("Growth vs Δ unemployment r", f"{corr:+.2f}" if corr is not None else "N/A")
    st.caption(f"Averages use available economy-years with equal weight. Correlation compares GDP growth "
               f"with annual changes in unemployment, using {complete_pairs} complete pairs; not a causal estimate.")
    st.divider()

    if graph_open:
        render_full_dashboard(full, data, countries, years, active_crisis)
        return

    overview, crisis_tab, relationships, data_tab = st.tabs([
        "Overview", "Crisis analysis", "Economic relationships", "Data & report"
    ])
    with overview:
        st.subheader("Macroeconomic trends")
        trend_tabs = st.tabs(["GDP growth", "Unemployment", "Inflation", "Currency depreciation"])
        for tab, field in zip(trend_tabs, METRIC_LABELS):
            with tab:
                plot_trend(data, field, years)
                if field == "fx_depreciation":
                    st.caption("Positive exchange-rate changes indicate currency depreciation; annual averages "
                               "can conceal abrupt intrayear movements.")
        with st.expander("Explore the complete economy-year heatmap", expanded=True):
            render_heatmap(data, countries, years)
        st.subheader("Three economic findings")
        for i, (heading, insight) in enumerate(economic_insights(data, active_crisis), 1):
            st.markdown(f"**{i}. {heading}**")
            st.write(insight)

    with crisis_tab:
        if active_crisis == ALL_TIME_KEY:
            render_all_time_overview(full, data, countries, years)
        else:
            render_crisis_explorer(full, data, countries, years, active_crisis)
        with st.expander("Top 5 economic rankings", expanded=True):
            ranking_frame = (data if active_crisis == ALL_TIME_KEY
                             else _crisis_source_frame(full, countries, active_crisis))
            render_rankings(ranking_frame, active_crisis)

    with relationships:
        render_labor_analysis(data)
        st.divider()
        with st.expander("Inflation comparison", expanded=True):
            render_inflation_analysis(data)

    with data_tab:
        st.subheader("Source data and Executive Summary")
        report_data, report_years, report_slug = export_scope(
            full, data, countries, years, active_crisis
        )
        if active_crisis == ALL_TIME_KEY:
            st.caption(
                f"All Time | {report_years[0]}–{report_years[1]}. "
                "The selected CSV and original full-period report follow your year and economy filters."
            )
            selected_label = "Selected dataset (CSV)"
            selected_filename = "macrolens_selected_data.csv"
            report_label = "Executive Summary (PDF)"
            report_filename = "macrolens_asia_executive_summary.pdf"
        else:
            spec = CRISES[active_crisis]
            st.markdown(f"**{spec['title']}**")
            st.caption(
                f"Research period: {report_years[0]}–{report_years[1]} · "
                f"Shock window: {year_span(spec['shock'])} · "
                f"Pre-crisis GDP benchmark: {spec['baseline']}. "
                "For consistent crisis comparisons, these two downloads include the "
                "complete event analysis window even if you zoom into fewer chart years."
            )
            selected_label = "Crisis dataset (CSV)"
            selected_filename = (
                f"macrolens_{report_slug}_{report_years[0]}_{report_years[1]}.csv"
            )
            report_label = "Crisis Executive Summary (PDF)"
            report_filename = f"macrolens_{report_slug}_executive_summary.pdf"
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "Full dataset (CSV)",
                data=full.to_csv(index=False).encode("utf-8-sig"),
                file_name="macrolens_world_bank_8economies_1990_2025.csv",
                mime="text/csv", use_container_width=True,
            )
        with dl2:
            st.download_button(
                selected_label,
                data=report_data.to_csv(index=False).encode("utf-8-sig"),
                file_name=selected_filename,
                mime="text/csv", use_container_width=True,
            )
        with dl3:
            try:
                pdf = create_analysis_summary(
                    report_data, countries, report_years, active_crisis
                )
                st.download_button(
                    report_label, data=pdf, file_name=report_filename,
                    mime="application/pdf", use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Unable to generate PDF: {exc}")
        st.caption(
            f"Report evidence: World Bank WDI observations for "
            f"{len(countries)} selected economies and {report_years[0]}–{report_years[1]}. "
            "Historical background comes from the institutional sources linked above. "
            "Missing values remain unavailable, not zero."
        )
        st.divider()
        st.markdown("**Data quality for the exported selection**")
        render_data_checks(report_data)


if __name__ == "__main__":
    main()
