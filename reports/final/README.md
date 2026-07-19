# Cross-Market Momentum LaTeX Report

This directory contains the final English technical report built from the frozen
Cross-Market Momentum evidence.

## Files

- `cross_market_momentum_report.tex` -- editable LaTeX source.
- `cross_market_momentum_report.pdf` -- compiled report.
- `output/pdf/cross_market_momentum_report.pdf` -- verified delivery copy.
- `build_report_assets.py` -- validates frozen inputs and regenerates all tables
  and figures.
- `generated/figures/` -- vector PDF figures.
- `generated/tables/` -- LaTeX table fragments.
- `generated/source_manifest.csv` -- SHA-256 manifest of material source files.

## Rebuild

From the repository root:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/cmm-mpl python reports/final/build_report_assets.py

cd reports/final
tectonic cross_market_momentum_report.tex
```

The report builder does not rerun the baseline backtests. It reads the frozen
baseline v5 and corrected Phase 3 outputs and fails if their expected evidence
shape is not present.
