# Stochastic Scenario Lab

**Working title:** Neural SDEs for Interest-Rate Scenario Generation and Liability Risk  
**Implementation language:** Python

## Research question

> Can neural stochastic differential equations generate more realistic interest-rate scenarios than classical stochastic models, and does the difference meaningfully affect estimates of pension-fund liability risk?

We will investigate two questions:

1. **Modelling:** Does ML capture properties of interest-rate dynamics that a traditional SDE misses?
2. **Financial relevance:** Do those differences meaningfully change our estimates of funding shortfalls and liability exposure?

The second question prevents the project from becoming machine learning for its own sake.

## Approach

Build this as a milestone-based research project with adjustable scope, rather than fixed deadlines. **Every milestone must produce something that works, can be tested, and teaches us something about the financial problem.** We'll make design decisions as we encounter them and prioritize a working Python implementation over an elaborate architecture.

The first four milestones form a complete standalone project. The neural SDE is a research extension; a dashboard is optional.

## Roadmap

### M1 — Data and exploration

**Estimated effort:** 2–4 focused sessions

Obtain euro-area government bond yield curves from the European Central Bank (ECB). Clean the data, examine their evolution, and understand relationships among maturities.

**Deliverable:** A reproducible Python notebook with historical curves, rate changes, and preliminary statistical analysis.

**Decision point:** Which maturities and sampling frequency should we model?

### M2 — Classical stochastic model

**Estimated effort:** 3–5 sessions

Apply principal component analysis (PCA) to reduce the yield curve to a few factors. Fit a multivariate Ornstein–Uhlenbeck (OU) or VAR(1) model to their dynamics and simulate future curves using Monte Carlo.

**Deliverable:** A working economic scenario generator producing hundreds or thousands of possible interest-rate paths.

**Decision point:** How adequate is a linear Gaussian model? Are three PCA factors sufficient?

### M3 — Asset–liability application

**Estimated effort:** 2–4 sessions

Create a synthetic pension fund with a fixed-income portfolio and scheduled pension payments. Revalue assets and remaining liabilities under each simulated yield curve.

**Deliverable:** Funding-ratio distributions, shortfall probabilities, and stress-scenario analysis.

**Decision point:** Is a deterministic liability schedule sufficient, or would inflation indexation add important insight?

### M4 — Validation (minimum viable project complete)

**Estimated effort:** 3–5 sessions

Test the classical generator on historical periods not used for calibration. Compare forecasts and scenario distributions against observed rate movements and simple historical benchmarks.

**Deliverable:** A defensible baseline report and an end-to-end working research application in Python.

**Decision point:** What deficiencies, if any, justify introducing time-varying volatility (M4.5) and then a neural SDE?

### M4.5 — GARCH(1,1) volatility baseline

**Estimated effort:** 2–3 sessions

Add time-varying volatility to the classical generator: GARCH(1,1) per PCA factor with Student-t shocks, simulating from the current conditional volatility at the forecast origin. Re-run the M4 validation (single cut and expanding window) and compare coverage with the constant-volatility baseline.

**Deliverable:** A volatility-clustering baseline, and a validation comparison showing what time-varying volatility fixes (calm and turbulent regimes) and what it leaves open (for example mean reversion at the long end).

**Decision point:** Is the remaining gap large enough to justify a neural SDE?

### M5 — Neural SDE

**Estimated effort:** 5–10+ sessions; research-intensive

Introduce PyTorch and learn state-dependent drift and diffusion functions. Compare the generated scenarios, out-of-sample behaviour, and liability-risk estimates against M2–M4.

A generic neural SDE takes the form

$$
dX_t = f_\theta(X_t,t)\,dt + G_\theta(X_t,t)\,dW_t.
$$

**Deliverable:** A trained neural scenario generator with a reproducible comparison against classical models.

**Decision point:** Does the additional model complexity improve anything financially important?

### M6 — Research application and extensions (optional)

**Estimated effort:** Flexible, after the research works

Package the code, build interactive visualizations, and potentially add a small Streamlit dashboard. A FastAPI service or separate frontend can follow if justified.

**Deliverable:** A usable demonstration and documented research findings.

## Initial design decisions

These are **starting defaults, not irreversible commitments**.

| Component | Initial choice | Why |
|---|---|---|
| Language | Python 3.12+ | Single-language research stack |
| Data | ECB AAA euro yield curves | Public daily term-structure data |
| Maturities | 1, 2, 5, 10, 20, 30 years | Small but representative yield curve |
| Sampling | Weekly, initially | Reduces data volume and short-term noise |
| State variables | 3 PCA factors | Manageable stochastic state space |
| Classical model | Multivariate OU | Interpretable continuous-time baseline |
| Simulation horizon | 1 year initially | Easier to validate historically |
| Liabilities | Fixed nominal payments over 20 years | Simple but rate-sensitive |
| ML | PyTorch neural SDE | Flexible nonlinear model |
| Interface | Jupyter notebooks first | Prioritize research over frontend |

**Data source:** [ECB yield-curve datasets](https://data.ecb.europa.eu/data/datasets/YC)

**Important distinction:** The *simulation horizon* and *liability maturity* are different. We can simulate one year of rate movements while asking how the present value of the following 20 years of pension payments changes.

## Python project structure

Keep the repository small initially:

```text
stochastic-scenario-lab/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_explore_yield_curves.ipynb
│   ├── 02_classical_sde.ipynb
│   └── 03_liability_scenarios.ipynb
├── src/
│   └── scenario_lab/
│       ├── data.py
│       ├── factors.py
│       ├── models/
│       │   └── ou.py
│       ├── simulation.py
│       └── liabilities.py
├── tests/
├── pyproject.toml
└── README.md
```

Add neural models and experiment management later rather than creating a large empty architecture now.

**Early dependencies:** NumPy, pandas, SciPy, scikit-learn, matplotlib, and Jupyter. Add PyTorch in M5.

## Engineering and research rules

1. **Implement the important mathematics ourselves.** NumPy for linear algebra is fine, but we should understand and implement stochastic simulations and parameter estimation rather than hide everything behind an API.
2. **Validate each model before moving forward.** For an OU simulation, for example, compare the simulated conditional mean, variance, and correlation structure with theoretical properties.
3. **Keep the ML comparison fair.** Fit preprocessing and models on training data only, use chronological holdout periods, and do not assume the neural model will outperform classical models.

## First working session: M1.1 — Build the yield-curve dataset

**Scope:** Just four tasks.

- [ ] Set up the Python project and environment.
- [ ] Download and parse six ECB yield-curve series.
- [ ] Create a clean pandas DataFrame indexed by date.
- [ ] Plot historical yield curves and individual rate series.

**Expected DataFrame structure (illustrative numbers, not ECB observations):**

| Date | 1Y | 2Y | 5Y | 10Y | 20Y | 30Y |
|---|---:|---:|---:|---:|---:|---:|
| $t_1$ | 2.10 | 2.25 | 2.50 | 2.75 | 3.05 | 3.15 |
| $t_2$ | 2.12 | 2.28 | 2.53 | 2.78 | 3.08 | 3.18 |

*Rates are expressed in percent.*

**Exit criterion:** A reproducible dataset and plots that we understand and trust.

Once this is done, we'll inspect the actual data and decide whether PCA is the right next step.

## Questions deliberately left open

- Which maturities and historical sample period are most appropriate?
- Should the model use daily or weekly observations?
- Are three PCA factors enough to describe the variation we care about?
- Should the baseline be continuous-time OU, discrete VAR(1), or both?
- How should we validate long-horizon scenarios with limited realized historical outcomes?
- Will inflation-linked liabilities materially improve the case study?
- Which neural SDE components should be learned: drift, diffusion, or both?
- Do the benefits of an ML-based generator outweigh its complexity and estimation risk?

We'll resolve these **as the evidence emerges**, not upfront.

## Success criterion

A meaningful outcome does **not** require the neural SDE to beat every baseline. A credible finding might be that a simple linear Gaussian model misses certain tail behaviours, or that a neural model fits some statistics better but does not materially change funding-risk estimates.

**The goal is not to build the most complicated model. It is to establish whether additional complexity makes financial scenario analysis better.**
