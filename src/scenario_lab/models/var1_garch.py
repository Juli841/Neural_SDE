"""var1_garch: scenario generator: same 3-factor PCA + VAR(1) as var1, but the shocks come from one
Garch11 per factor (sigma_garch) instead of constant-sigma Student-t draws.

Only the shock distribution changes. PCA loadings V, VAR matrix Phi and the curve reconstruction are
reused from var1. Factors stay independent, as in var1 (multivariate GARCH was dropped from the plan).
"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from scenario_lab.models import var1 as base
from scenario_lab.models.sigma_garch import Garch11

N_PATHS, N_DAYS = base.N_PATHS, base.N_DAYS


class GarchVarParams(NamedTuple):
    V: np.ndarray            # (n_mats, k) PCA loadings, same as base.Params.V
    Phi: np.ndarray          # (k, k) VAR(1) matrix, same as base.Params.Phi
    garch: list              # k fitted Garch11 objects, one per factor (params + sigma2_last inside)


def fit(changes: pd.DataFrame, k: int = 3) -> GarchVarParams:
    """Fit on daily yield changes (T, n_mats).
    Steps:
    1. base.fit(changes, k) -> V and Phi (ignore its df and sigma).
    2. scores = changes.values @ V; VAR residuals = scores[1:] - scores[:-1] @ Phi.T, shape (T-1, k).
    3. For each factor j: Garch11().fit(residuals[:, j]).
    """
    b = base.fit(changes, k)
    scores = changes.values @ b.V
    resid = scores[1:] - scores[:-1] @ b.Phi.T
    return GarchVarParams(V=b.V, Phi=b.Phi, garch=[Garch11().fit(resid[:, j]) for j in range(k)])


def shocks(p: GarchVarParams, n_paths: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """GARCH shocks eps, shape (n_paths, n_days, k): factor j is p.garch[j].simulate(n_paths, n_days, rng),
    stacked on the last axis (the same layout base.simulate uses for its eps)."""
    return np.stack([g.simulate(n_paths, n_days, rng) for g in p.garch], axis=-1)


def simulate(p: GarchVarParams, y0: np.ndarray, n_paths: int = N_PATHS, n_days: int = N_DAYS, seed: int = 0) -> np.ndarray:
    """Return curves, shape (n_paths, n_days + 1, n_mats); [:, 0] is y0. Same contract as base.simulate.

    Steps: rng = np.random.default_rng(seed); eps = shocks(p, n_paths, n_days, rng);
    return base.curves_from_shocks(p.Phi, p.V, y0, eps)  (the shared VAR loop + reconstruction).
    """
    rng = np.random.default_rng(seed)
    return base.curves_from_shocks(p.Phi, p.V, y0, shocks(p, n_paths, n_days, rng))
