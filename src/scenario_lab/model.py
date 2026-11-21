"""M2 baseline scenario generator: 3-factor PCA + VAR(1) on daily factor changes + independent Student-t shocks.

Decisions are in LOG.md. Zero drift, no mean reversion, factors-only reconstruction.
"""
from typing import NamedTuple

from scenario_lab.factors import fit_pca
from scipy import stats
import numpy as np
import pandas as pd

FIT_END = "2018-12-31"   # validation fit uses data up to here; production fit uses everything
N_PATHS, N_DAYS = 25_000, 250
MIN_DF = 3.0             # floor so the variance stays finite


class Params(NamedTuple):
    V: np.ndarray        # (n_mats, 3) PCA loadings (from factors.fit_pca)
    Phi: np.ndarray      # (3, 3) VAR(1): s_t = Phi @ s_{t-1} + eps_t
    df: np.ndarray       # (3,) Student-t degrees of freedom per factor, floored at MIN_DF
    sigma: np.ndarray    # (3,) residual std of each factor (the t draws are scaled to this)


def fit(changes: pd.DataFrame, k: int = 3) -> Params:
    """Fit on daily yield changes (T, n_mats).
    Steps: fit_pca -> scores = changes @ V (zero drift: do NOT subtract a mean)
    -> VAR(1) without trend on the scores -> residuals -> per-factor t fit, df = max(df, MIN_DF),
    sigma = residual std.
    """
    _, V, _ = fit_pca(changes.values, k)
    scores = changes.values @ V
    X = scores[:-1] # yesterday's values
    Y = scores[1:] #today's values

    phi_t = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    epsilon_hat = Y - X @ phi_t

    sigma = np.array([np.std(epsilon_hat[:, j]) for j in range(k)])
    z_hat = epsilon_hat / sigma

    df = np.array([stats.t.fit(z_hat[:, j], floc = 0) for j in range(k)])[:, 0]
    degrees = np.maximum(df, MIN_DF)
    return Params(V= V, Phi = phi_t.T, df = degrees, sigma = sigma)

def simulate(p: Params, y0: np.ndarray, n_paths: int = N_PATHS, n_days: int = N_DAYS, seed: int = 0) -> np.ndarray:
    """Return curves, shape (n_paths, n_days + 1, n_mats); [:, 0] is y0.

    Per day: eps ~ t(df) * sigma * sqrt((df-2)/df), s_t = Phi @ s_{t-1} + eps,
    curve_t = curve_{t-1} + V @ s_t. Start with s_0 = 0 (or the last observed scores; decide).
    """
    rng = np.random.default_rng(seed)
    k = p.V.shape[1]
    s_t = np.zeros((n_paths, k))
    eps = p.sigma * np.sqrt((p.df - 2) / p.df) * stats.t.rvs(p.df, size=(n_paths, n_days, k), random_state=rng)
    S = np.empty((n_paths, n_days, k))
    for t in range(n_days):
        s_t = s_t @ p.Phi.T + eps[:, t]
        S[:, t] = s_t
    dy = S @ p.V.T
    y0 = np.asarray(y0)
    return np.concatenate([np.broadcast_to(y0, (n_paths, 1, y0.size)), y0 + np.cumsum(dy, axis=1)], axis=1)

# ---- M4 validation helpers (not needed for M2, except theory_check) ----
# fit on <= FIT_END, simulate from the curve at FIT_END, compare with later data

def horizon_quantiles(curves: np.ndarray, h: int = N_DAYS, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """Check 1 (simulated side): quantiles of curves[:, h] - curves[:, 0] per maturity."""
    raise NotImplementedError


def realised_changes(levels: pd.DataFrame, h: int = N_DAYS) -> pd.DataFrame:
    """Check 1 (realised side): overlapping h-day changes levels.shift(-h) - levels, dropna."""
    raise NotImplementedError


def coverage(sim_changes: np.ndarray, realised: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    """Check 2: per maturity, fraction of realised changes inside the simulated [lo, hi] band."""
    raise NotImplementedError


def daily_moments(daily_changes: np.ndarray) -> dict:
    """Check 3: std, excess kurtosis and correlation matrix of daily changes. Call on simulated and on actual."""
    raise NotImplementedError


def theory_check(p: Params, curves: np.ndarray, h: int = 5) -> pd.DataFrame:
    """Check 4: simulated mean/variance of the factor state after h days vs the VAR(1) closed form
    (variance via sum Phi^k Sigma Phi^k'; for t shocks Sigma = diag(sigma**2)). Catches coding bugs."""
    dy = np.diff(curves, axis = 1)
    S = dy @ p.V
    mean = np.mean(S[:, h-1], axis = 0)
    std = np.std(S[:, h-1], axis = 0)
    C = np.zeros(shape = (len(p.sigma), len(p.sigma)))
    for t in range(h):
        C = p.Phi @ C @ p.Phi.T + np.diag(p.sigma ** 2)
    theory_std = np.sqrt(np.diag(C))
    return pd.DataFrame({"theory mean": np.zeros(shape = mean.shape), "sim_mean" :mean,"theory std": theory_std, "sim_std": std})
