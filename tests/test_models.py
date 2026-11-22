"""Smoke and sanity tests on synthetic data (no network). Run: uv run pytest"""
import numpy as np
import pandas as pd
import pytest

from scenario_lab import validation as v
from scenario_lab.models import MODELS
from scenario_lab.models.sigma_garch import Garch11

MATS = ["1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]


@pytest.fixture(scope="module")
def levels():
    """Six correlated yields from three GARCH-like factors, 3000 business days."""
    rng = np.random.default_rng(0)
    T = 3000
    load = np.array([[1, .6, .3], [1, .5, .1], [1, .2, -.2], [1, 0, -.4], [1, -.2, -.3], [1, -.3, -.1]])
    sig = np.full(3, 0.03); f = np.zeros((T, 3))
    for t in range(T):
        f[t] = sig * rng.standard_t(6, 3)
        sig = np.sqrt(1e-6 + 0.08 * f[t] ** 2 + 0.9 * sig ** 2)
    y = 3 + np.cumsum(f @ load.T, axis=0)
    return pd.DataFrame(y, index=pd.bdate_range("2005-01-03", periods=T), columns=MATS)


@pytest.mark.parametrize("key", list(MODELS))
def test_model_interface(key, levels):
    """fit/simulate contract: shape (n_paths, n_days + 1, n_mats), starts at y0, finite."""
    m = MODELS[key]
    changes = levels.diff().dropna()
    p = m.fit(changes)
    y0 = levels.iloc[-1].values
    c = m.simulate(p, y0, n_paths=200, n_days=30, seed=1)
    assert c.shape == (200, 31, 6) and np.isfinite(c).all()
    assert np.allclose(c[:, 0], y0)
    assert np.array_equal(c, m.simulate(p, y0, n_paths=200, n_days=30, seed=1))        # reproducible


def test_garch_recovers_parameters():
    rng = np.random.default_rng(0)
    om, al, be, nu, T = 2e-6, 0.08, 0.90, 6.0, 4000
    eps = np.zeros(T); s2 = om / (1 - al - be)
    for t in range(T):
        eps[t] = np.sqrt(s2) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        s2 = om + al * eps[t] ** 2 + be * s2
    g = Garch11().fit(eps)
    assert abs(g.params.alpha - al) < 0.04 and abs(g.params.beta - be) < 0.04
    assert 0.9 < g.params.alpha + g.params.beta < 1.0 + 1e-6
    assert 3 < g.params.nu < 12
    sim = g.simulate(5000, 5, np.random.default_rng(1))
    assert sim.shape == (5000, 5) and abs(sim[:, 0].var() / g.sigma2_last - 1) < 0.1      # day-0 variance = sigma2_last


def test_coverage_and_validation_shapes(levels):
    # coverage of a simulated sample against itself is the nominal band width
    sim = np.random.default_rng(0).normal(size=(20_000, 6))
    real = pd.DataFrame(np.random.default_rng(1).normal(size=(5_000, 6)), columns=MATS)
    assert np.allclose(v.coverage(sim, real, 0.05, 0.95), 0.9, atol=0.02)

    m = {"var1": MODELS["var1"]}
    res = v.compare(m, levels, origins=["2012-12-31", "2013-12-31"], cut="2012-12-31", n_paths=300)
    r = res["var1"]
    assert r["B"]["cov90"].shape == (2, 6) and r["B"]["std_ratio"].shape == (2, 6)
    assert r["A"]["cov90"].shape == (6, 2)
    s = v.summary(res)
    assert list(s.columns) == ["A_dev90", "B_dev90", "B_dev50", "B_logstd"] and np.isfinite(s.values).all()
