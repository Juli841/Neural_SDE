# Stochastic Scenario Lab

Interest-rate scenario generation and funding-risk analysis for euro-area yield curves. The aim is to ask two questions:

1. **Modelling:** does a richer model (time-varying volatility, later a neural SDE) capture properties of rate dynamics that a classical linear model misses?
2. **Financial relevance:** do those differences change estimates of funding shortfalls for a liability portfolio?

The roadmap is in [`stochastic_scenario_lab_roadmap.md`](stochastic_scenario_lab_roadmap.md).

## Status

| Stage | Content | State |
|---|---|---|
| Data exploration | ECB AAA, Bundesbank and Federal Reserve zero curves (1Y, 2Y, 5Y, 10Y, 20Y, 30Y) | done |
| var1 | PCA, VAR(1), constant-sigma Student-t shocks, Monte Carlo; ALM example; validation | done |
| garch | GARCH(1,1) volatility on the var1 factors, written by hand to mimic the `arch` library; scenario generator and validation (no ALM yet) | done |
| neural SDE | Neural SDE (planned) | planned |

## The baseline model

Daily changes of the six yields are reduced to three principal components. The factor scores follow a VAR(1) with zero drift, and the residuals are independent Student-t shocks with constant volatility. Curves are rebuilt from the factors and simulated for 250 business days (about one year), 25,000 paths.

## Findings so far

**Asset-liability example (var1).** A flat liability of 1 per year for 30 years is covered by static portfolios of 5Y and 30Y zero-coupon bonds, valued at a one-year horizon on each simulated curve. The funding ratio (assets / remaining liabilities) at day 250:

| Portfolio | Mean | Std | VaR95 | ES95 |
|---|---|---|---|---|
| Short (5Y only) | 0.996 | 0.062 | 0.895 | 0.869 |
| Matched (duration-matched 5Y/30Y) | 1.000 | 0.014 | 0.978 | 0.972 |
| Long (30Y only) | 1.008 | 0.152 | 0.772 | 0.724 |

Duration matching removes most of the funding risk; what remains comes from slope and curvature moves.

**Validation (var1).** The simulated 250-day change distribution is compared with realised changes: one fit up to 2018 and tested afterwards, and an expanding window with origins at every year-end 2012 to 2022. Main result: constant volatility is the weak point. Bands are far too wide in calm years (2012 to 2019) and too narrow when the 2022 rate hikes arrive, so pooled coverage (about 0.83 to 0.90 for a nominal 0.90) hides a regime pattern. Simulated daily volatility is 1.5 to 3 times too high at the short end in calm years and about half of the realised level in 2022 and 2023. This motivates the GARCH model. Details and plots are in `notebooks/var1/03_validation.ipynb`.

**GARCH volatility.** One GARCH(1,1) with Student-t shocks per factor removes the volatility clustering in the daily residuals and improves daily-std calibration modestly, but does not improve 250-day coverage (mean |coverage - nominal| for the 5-95% band: 0.193 against 0.185 for the constant-volatility model). The horizon gap is mostly a trending effect: the 250-day variance ratio is 5.7 at 1Y and 2.5 at 2Y in the data against about 1.1 in both models. Longer VAR lags and an ARMA(1,1) drift (exploratory, `notebooks/scratch/`) do not close it, and the same pattern appears on the longer German (Bundesbank) history. Details in `notebooks/garch/03_validation.ipynb`; all models are compared with one workflow in `notebooks/model_comparison.ipynb`.

**Caveats.** There are only about 7 to 11 independent years behind each coverage number, because 250-day windows overlap. The modelling choices (3 factors, zero drift, VAR(1), Student-t) were made after looking at the full sample, so the validation is not strictly clean.

## Repository layout

```
src/scenario_lab/
    data.py            yield-curve loaders: ECB (AAA, ALL), Bundesbank (DE), Federal Reserve (US)
    factors.py         PCA of yield changes
    validation.py      workflow shared by all models: design A (single cut), design B (expanding window),
                       coverage, variance ratio, summary table, heatmaps
    models/
        __init__.py    model interface (fit / simulate) and the MODELS registry
        var1.py        PCA + VAR(1) + constant-sigma Student-t shocks
        sigma_garch.py single-factor GARCH(1,1), Student-t (formulas in sigma_garch.md)
        var1_garch.py  PCA + VAR(1) + one GARCH(1,1) per factor
notebooks/
    exploration/                     the same exploration for each dataset
        01_ecb_aaa.ipynb             ECB AAA (main series)
        02_bundesbank_de.ipynb       Bundesbank German curve
        03_fed_us.ipynb              Federal Reserve US curve
    var1/                            classical model: PCA + VAR(1) + constant-sigma Student-t
        01_pca_factors.ipynb
        02_scenario_generator.ipynb
        03_validation.ipynb          designs A and B
        04_alm.ipynb                 asset-liability example
    garch/                           same layout, GARCH(1,1) volatility (no ALM yet)
        01_pca_factors.ipynb         shared PCA + volatility clustering in the scores
        02_scenario_generator.ipynb
        03_validation.ipynb
    neural_sde/                      empty, planned
    model_comparison.ipynb           all registered models through the same validation workflow
    scratch/                         exploratory: VAR(p) lags, ARMA drift, Bundesbank data (not part of the model folders)
tests/                              interface, GARCH recovery and validation tests (synthetic data, no network)
```

## Comparing models

Every model is a `fit(changes)` / `simulate(params, y0, n_paths, n_days, seed)` pair registered in `scenario_lab.models.MODELS`. `notebooks/model_comparison.ipynb` runs the same validation code (same data, origins, seeds and path counts, fit on data up to each origin only) for every registered model and reports one summary table. A new model, such as the M5 neural SDE, has to implement the two functions and be added to `MODELS`; nothing else changes.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run jupyter lab
uv run pytest
```

Yield curves are downloaded on first use and cached in `data/raw/` (not committed).

## Data

Main series: ECB euro-area government bond yield curve, AAA-rated issuers, spot rates in percent, daily from 2004-09-06 (`load_curve("AAA")`).

Longer histories for robustness checks, same format (1Y, 2Y, 5Y, 10Y, 20Y, 30Y, percent):

- `load_curve("DE")`: Bundesbank Svensson zero-coupon curve, German federal securities, daily from 2000-08.
- `load_curve("US")`: Federal Reserve (Gurkaynak-Sack-Wright) Treasury zero-coupon curve, daily from 1985-11.
- `load_curve("ALL")`: ECB all euro-area issuers, same period as AAA.
