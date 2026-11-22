# Stochastic Scenario Lab

Interest-rate scenario generation and funding-risk analysis for euro-area yield curves. The aim is to ask two questions:

1. **Modelling:** does a richer model (time-varying volatility, later a neural SDE) capture properties of rate dynamics that a classical linear model misses?
2. **Financial relevance:** do those differences change estimates of funding shortfalls for a liability portfolio?

The roadmap is in [`stochastic_scenario_lab_roadmap.md`](stochastic_scenario_lab_roadmap.md).

## Status

| Milestone | Content | State |
|---|---|---|
| M1 | ECB AAA euro-area spot curves (1Y, 2Y, 5Y, 10Y, 20Y, 30Y), exploration | done |
| M2 | Classical generator: 3-factor PCA, VAR(1), Student-t shocks, Monte Carlo | done |
| M3 | Toy asset-liability example, funding-ratio distribution | done |
| M4 | Validation against history (single cut and expanding window) | done |
| M4.5 | GARCH(1,1) volatility, written by hand to mimic the `arch` library | in progress |
| M5 | Neural SDE | planned |

## The baseline model

Daily changes of the six yields are reduced to three principal components. The factor scores follow a VAR(1) with zero drift, and the residuals are independent Student-t shocks with constant volatility. Curves are rebuilt from the factors and simulated for 250 business days (about one year), 25,000 paths.

## Findings so far

**Asset-liability example (M3).** A flat liability of 1 per year for 30 years is covered by static portfolios of 5Y and 30Y zero-coupon bonds, valued at a one-year horizon on each simulated curve. The funding ratio (assets / remaining liabilities) at day 250:

| Portfolio | Mean | Std | VaR95 | ES95 |
|---|---|---|---|---|
| Short (5Y only) | 0.996 | 0.062 | 0.895 | 0.869 |
| Matched (duration-matched 5Y/30Y) | 1.000 | 0.014 | 0.978 | 0.972 |
| Long (30Y only) | 1.008 | 0.152 | 0.772 | 0.724 |

Duration matching removes most of the funding risk; what remains comes from slope and curvature moves.

**Validation (M4).** The simulated 250-day change distribution is compared with realised changes: one fit up to 2018 and tested afterwards, and an expanding window with origins at every year-end 2012 to 2022. Main result: constant volatility is the weak point. Bands are far too wide in calm years (2012 to 2019) and too narrow when the 2022 rate hikes arrive, so pooled coverage (about 0.83 to 0.90 for a nominal 0.90) hides a regime pattern. Simulated daily volatility is 1.5 to 3 times too high at the short end in calm years and about half of the realised level in 2022 and 2023. This motivates M4.5. Details and plots are in `notebooks/05_validation.ipynb`.

**Caveats.** There are only about 7 to 11 independent years behind each coverage number, because 250-day windows overlap. The modelling choices (3 factors, zero drift, VAR(1), Student-t) were made after looking at the full sample, so the validation is not strictly clean.

## Repository layout

```
src/scenario_lab/
    data.py          download and cache ECB yield curves
    factors.py       PCA of yield changes
    model.py         fit / simulate the classical generator, validation helpers
    model_garch.py   GARCH(1,1) with Student-t innovations (in progress)
    model_garch.md   formulas for the GARCH implementation
notebooks/
    01_explore_yield_curves.ipynb
    02_pca_factors.ipynb
    03_scenario_generator.ipynb
    04_alm.ipynb         asset-liability example
    05_validation.ipynb  validation, designs A and B
    06_garch.ipynb       GARCH step (in progress)
tests/
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run jupyter lab
uv run pytest
```

Yield curves are downloaded from the ECB data portal on first use and cached in `data/raw/` (not committed).

## Data

ECB euro-area government bond yield curve, AAA-rated issuers, spot rates in percent, daily from 2004-09-06.
