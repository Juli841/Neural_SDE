"""Validation workflow shared by all scenario models (see models/__init__.py for the interface).

Design A: one fixed cut. Fit up to the cut, simulate from that curve, compare with the realised 250-day changes.
Design B: expanding window. Year-end origins; at each origin fit, simulate, and test on the windows that START in
the following 250 trading days (walk-forward: every window is scored once, against the model fitted just before it).

Coverage = fraction of realised 250-day changes inside the simulated [lo, hi] quantile band, per maturity
(nominal 0.90 for 5-95%, 0.50 for 25-75%). Judge descriptively: 250-day windows overlap, so each year is about
one independent observation.
"""
import numpy as np
import pandas as pd
from scipy import stats

HORIZON = 250
FIT_END = "2018-12-31"          # default cut for design A
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BANDS = {"cov90": (0.05, 0.95, 0.90), "cov50": (0.25, 0.75, 0.50)}


# ---------------------------------------------------------------- building blocks

def realised_changes(levels: pd.DataFrame, h: int = HORIZON) -> pd.DataFrame:
    """Overlapping h-day changes levels.shift(-h) - levels, indexed by the window start date."""
    return (levels.shift(-h) - levels).dropna()


def horizon_quantiles(curves: np.ndarray, h: int = HORIZON, qs=QUANTILES, maturities=("1Y", "2Y", "5Y", "10Y", "20Y", "30Y")) -> pd.DataFrame:
    """Quantiles of curves[:, h] - curves[:, 0] per maturity (rows = quantiles)."""
    change = curves[:, h] - curves[:, 0]
    return pd.DataFrame(np.quantile(change, qs, axis=0), index=qs, columns=list(maturities))


def coverage(sim_changes: np.ndarray, realised: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    """Per maturity, fraction of realised changes inside the simulated [lo, hi] band."""
    lows, highs = np.quantile(sim_changes, [lo, hi], axis=0)
    inside = (realised.values >= lows) & (realised.values <= highs)
    return pd.Series(inside.mean(axis=0), index=realised.columns, name="coverage")


def daily_moments(daily_changes: np.ndarray) -> dict:
    """Std, excess kurtosis and correlation matrix of daily changes (call on simulated and on actual)."""
    return {"std": daily_changes.std(axis=0), "excess_kurt": stats.kurtosis(daily_changes, axis=0),
            "corr_matrix": np.corrcoef(daily_changes, rowvar=False)}


def variance_ratio(curves: np.ndarray) -> np.ndarray:
    """var(h-day change) / (h * var(daily change)) per maturity from simulated curves. 1 = uncorrelated daily changes."""
    h = curves.shape[1] - 1
    daily = np.diff(curves, axis=1).reshape(-1, curves.shape[2])
    return (curves[:, -1] - curves[:, 0]).var(axis=0) / (h * daily.var(axis=0))


def realised_variance_ratio(levels: pd.DataFrame, h: int = HORIZON) -> pd.Series:
    """Same ratio on the data, using all overlapping h-day windows."""
    return realised_changes(levels, h).var() / (h * levels.diff().dropna().var())


def year_end_origins(first: int, last: int) -> list[str]:
    return [f"{y}-12-31" for y in range(first, last + 1)]


# ---------------------------------------------------------------- one forecast origin

def simulate_from_origin(model, levels: pd.DataFrame, origin: str, n_paths: int = 25_000, n_days: int = HORIZON,
                         seed: int = 0) -> np.ndarray:
    """Fit on daily changes up to `origin` (inclusive), simulate from the curve at `origin`. No look-ahead."""
    changes = levels.diff().dropna()
    params = model.fit(changes.loc[:origin])
    y0 = levels.loc[:origin].iloc[-1].values
    return model.simulate(params, y0, n_paths=n_paths, n_days=n_days, seed=seed)


# ---------------------------------------------------------------- the two designs

def design_a(model, levels: pd.DataFrame, cut: str = FIT_END, **sim_kw) -> dict:
    """Single cut. Returns {'cov90', 'cov50': DataFrame(maturity x [train, test]), 'quantiles': DataFrame}.
    test = windows starting at or after the cut; train = windows ending inside the fitting sample."""
    curves = simulate_from_origin(model, levels, cut, **sim_kw)
    sim = curves[:, -1] - curves[:, 0]
    realised = realised_changes(levels)
    test, train = realised.loc[cut:], realised.loc[:cut].iloc[:-HORIZON]
    out = {k: pd.DataFrame({"train": coverage(sim, train, lo, hi), "test": coverage(sim, test, lo, hi)})
           for k, (lo, hi, _) in BANDS.items()}
    out["quantiles"] = horizon_quantiles(curves, maturities=list(levels.columns))
    out["n_test_windows"] = len(test)
    return out


def design_b(model, levels: pd.DataFrame, origins: list[str], **sim_kw) -> dict:
    """Expanding window. Returns {'cov90', 'cov50', 'std_ratio': DataFrame(origin x maturity)}.
    std_ratio = simulated daily std / actual daily std in the test year (1 = calibrated)."""
    changes = levels.diff().dropna()
    realised = realised_changes(levels)
    cols = list(levels.columns)
    rows = {"cov90": {}, "cov50": {}, "std_ratio": {}}
    for o in origins:
        curves = simulate_from_origin(model, levels, o, **sim_kw)
        sim = curves[:, -1] - curves[:, 0]
        test = realised[realised.index > o].iloc[:HORIZON]
        for k, (lo, hi, _) in BANDS.items():
            rows[k][o] = coverage(sim, test, lo, hi)
        sim_std = np.diff(curves, axis=1).reshape(-1, len(cols)).std(axis=0)
        act_std = changes[changes.index > o].iloc[:HORIZON].std().values
        rows["std_ratio"][o] = pd.Series(sim_std / act_std, index=cols)
    return {k: pd.DataFrame(v).T for k, v in rows.items()}


# ---------------------------------------------------------------- comparing models

def compare(models: dict, levels: pd.DataFrame, origins: list[str], cut: str = FIT_END, **sim_kw) -> dict:
    """Run design A and B with identical settings for every model. models: {key: Model}.
    Returns {key: {'name', 'A': design_a(...), 'B': design_b(...)}}."""
    return {k: {"name": m.name, "A": design_a(m, levels, cut, **sim_kw), "B": design_b(m, levels, origins, **sim_kw)}
            for k, m in models.items()}


def summary(results: dict) -> pd.DataFrame:
    """One row per model. Lower is better in every column.
    A_dev90: design A test coverage, mean |coverage - 0.90| over maturities.
    B_dev90 / B_dev50: design B, mean |coverage - nominal| over origins and maturities (5-95% / 25-75%).
    B_logstd: design B, mean |log(simulated / actual daily std)|."""
    rows = {}
    for k, r in results.items():
        rows[r["name"]] = {
            "A_dev90": (r["A"]["cov90"]["test"] - 0.9).abs().mean(),
            "B_dev90": (r["B"]["cov90"] - 0.9).abs().values.mean(),
            "B_dev50": (r["B"]["cov50"] - 0.5).abs().values.mean(),
            "B_logstd": np.abs(np.log(r["B"]["std_ratio"].values)).mean(),
        }
    return pd.DataFrame(rows).T


def variance_ratio_table(models: dict, levels: pd.DataFrame, **sim_kw) -> pd.DataFrame:
    """Realised variance ratio against each model's (full-sample fit, simulated from today's curve)."""
    cols = list(levels.columns)
    out = {"realised": realised_variance_ratio(levels)}
    changes = levels.diff().dropna()
    for m in models.values():
        curves = m.simulate(m.fit(changes), levels.iloc[-1].values, **sim_kw)
        out[m.name] = pd.Series(variance_ratio(curves), index=cols)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- plots

def plot_heatmaps(results: dict, key: str = "cov90", nominal: float = 0.9):
    """Per-origin heatmap of design B `key` (coverage or std_ratio) for every model. Blue above nominal, red below."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    ratio = key == "std_ratio"
    norm = TwoSlopeNorm(vcenter=1, vmin=0.4, vmax=2.5) if ratio else TwoSlopeNorm(vcenter=nominal, vmin=0, vmax=1)
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 0.35 * len(next(iter(results.values()))["B"][key]) + 2),
                             sharey=True, squeeze=False)
    for ax, (k, r) in zip(axes[0], results.items()):
        tab = r["B"][key]
        im = ax.imshow(tab.values, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns)
        ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels([o[:4] for o in tab.index])
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                ax.text(j, i, f"{tab.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(r["name"], fontsize=9)
    axes[0, 0].set_ylabel("origin year (tested on the following year)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    return fig
