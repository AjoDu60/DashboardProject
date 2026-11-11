# MacroLens Asia — Crisis Explorer (1990–2025)

An interactive Streamlit dashboard for eight Asian economies using **real, secondary World Bank World Development Indicators**. The app does **not** ship example economic observations; it downloads official data at runtime.

## Quick start (Windows / VS Code)

1. Put `app.py` and `requirements.txt` in the same folder.
2. Open the folder in VS Code. Open Terminal > New Terminal.
3. Run:

   ```powershell
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. If the World Bank API temporarily times out, choose **Refresh World Bank data** to retry. Individual successful series are cached for 24 hours.
5. Once successfully loaded, choose **Full dataset (CSV)** to save a backup you can upload later if your network is unavailable. **Previous-version backups will not work** because the upgraded dashboard uses additional World Bank series.

## What is included

- Eight economies: Indonesia, Japan, Malaysia, Thailand, the Philippines, South Korea, Singapore, and Hong Kong SAR
- Full period: **1990–2025**, plus presets for the 1997–1998 Asian Financial Crisis, the 2008–2009 global financial crisis, and the 2020 pandemic
- Country, year-range, metric, ranking-period, and ranking-type interactions
- Four KPI cards, macro trends, economy-year heatmap, scatter plot, inflation bar chart, before/after comparison, GDP recovery index and exchange-rate pressure chart
- Top 5 rankings for GDP contractions, inflation, mean GDP growth, unemployment increases and currency depreciation
- Downloadable cleaned and filtered CSV files and a dynamically written Executive Summary PDF with **author–year citations and clickable source links**

## Indicators and interpretation

| Indicator | World Bank WDI code | Notes |
|---|---|---|
| Annual real GDP growth (%) | `NY.GDP.MKTP.KD.ZG` | Growth rate, not GDP level |
| Unemployment (% labor force) | `SL.UEM.TOTL.ZS` | Modeled ILO estimate |
| Consumer-price inflation (%) | `FP.CPI.TOTL.ZG` | Annual CPI change |
| Constant-price GDP (US$) | `NY.GDP.MKTP.KD` | Used within each economy to create 1996=100 recovery indices |
| Official exchange rate (LCU per US$) | `PA.NUS.FCRF` | Annual-average rate; percentage increase means currency depreciation |

Annual unemployment changes are expressed in **percentage points**, while exchange-rate changes are **percentage changes**. Both are calculated from consecutive years in the unfiltered data so the first chosen year isn't wrongly treated as having no preceding data (1990 has no 1989 predecessor in this dashboard).

## Crisis comparisons

- **Before:** mean annual growth over 1994–1996
- **Crisis trough:** lowest single-year growth rate from 1997–1999
- **After:** mean annual growth over 2000–2002
- **Recovery index:** each economy's real GDP level divided by its own 1996 GDP level, multiplied by 100. The available plot covers 1996–2005, intersected with your selected years. Positive GDP growth alone does not mean the economy has regained its pre-crisis level.

The before/crisis/after values summarize distinct statistics—do not interpret a crisis trough as an average. Country filter and year-range filter affect the comparison. Use the **Asian Financial Crisis (1994–2002)** preset to view all three comparison windows.

## Data quality and academic use

The complete unfiltered grid has 8 × 36 = **288 possible economy-year rows**, but some WDI values may be missing, particularly in 2025. The dashboard leaves gaps missing, excludes them from relevant calculations, and reports missing counts; it never replaces missing values with zero. The FX and GDP-level requests are optional: if these fail, the core three-indicator dashboard can still run and shows a warning.

Every KPI/graph is descriptive. A pooled correlation between GDP growth and unemployment *levels* is not an estimate of Okun's Law or a causal effect. Differences in exchange-rate regimes also limit direct cross-economy interpretation.

**References** are linked in the app and PDF, using organization names instead of numbered brackets. Key historical reading: IMF (1999), *The Asia Crisis: Causes, Policy Responses, and Outcomes*; Federal Reserve History (2013), *Asian Financial Crisis*.

## Deployment

Upload these two files to a GitHub repository and deploy `app.py` at https://share.streamlit.io/ . The app needs internet access to the World Bank API on the first run. After checking every filter, submit the public Streamlit URL and a reviewed Executive Summary PDF.

## Validation performed

Python syntax, chart-creation flow, ranking computations, missing-value behavior, CSV round-trip, and ReportLab PDF generation were tested **offline with synthetic test fixtures (not included here)**. The generated test PDFs were confirmed to be one page with working link annotations. **Live World Bank API and real browser deployment were not tested** in the testing environment and should be verified locally before submission.
