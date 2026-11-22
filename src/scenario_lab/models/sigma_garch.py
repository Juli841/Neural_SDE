"""GARCH(1,1) with Student-t innovations, written by hand to mimic the `arch` library.

One `Garch11` fits ONE series (one PCA factor's VAR residuals). The three factors are fitted independently
(known weakness: their volatility regimes are correlated, see notebooks/garch/02_scenario_generator.ipynb).
Decisions are in LOG.md.

arch conventions to copy:
  - zero mean; parameters omega, alpha, beta, nu fitted jointly by quasi-maximum-likelihood
  - sigma2_t = omega + alpha * eps_{t-1}**2 + beta * sigma2_{t-1}
  - sigma2_0 = backcast: exponentially weighted mean of the first 75 squared residuals, weights 0.94**i
  - bounds omega >= 0, 0 <= alpha, beta <= 1, 2.05 <= nu <= 500; constraint alpha + beta < 1
  - scipy.optimize.minimize (SLSQP) on the negative log-likelihood
  - starting values: small grid over (alpha, beta), omega by variance targeting var * (1 - alpha - beta), nu ~ 8
  - rescale the data (x100) before fitting, convert omega back afterwards
"""
from typing import NamedTuple

import numpy as np
from scipy import optimize, stats, special

BACKCAST_OBS, BACKCAST_DECAY = 75, 0.94
SCALE = 100.0            # fit on eps * SCALE so the data are on a sensible scale


class Garch11Params(NamedTuple):
    omega: float
    alpha: float
    beta: float
    nu: float            # Student-t degrees of freedom


class Garch11:
    """Zero-mean GARCH(1,1) with Student-t innovations (unit-variance t)."""

    def __init__(self):
        self.params: Garch11Params | None = None
        self.sigma2_last: float | None = None   # conditional variance for the NEXT day, simulation start

    @staticmethod
    def backcast(eps: np.ndarray) -> float:
        """sigma2_0: exponentially weighted mean of the first BACKCAST_OBS squared residuals."""
        tau = min(75,len(eps))
        normalization = np.sum(np.array([0.94 ** j for j in range(tau)]))
        ws = np.array([(0.94 ** i) / normalization for i in range(tau)])
        sigma2_0 = np.sum(ws * (eps[:tau] ** 2))
        return sigma2_0

    @staticmethod
    def variance(eps: np.ndarray, omega: float, alpha: float, beta: float, sigma2_0: float) -> np.ndarray:
        """Conditional variances sigma2_t for t = 0..T (length T + 1): the last element is the forecast for the day after eps."""
        sigma2_t = np.zeros(len(eps)+1)
        sigma2_t[0] = sigma2_0
        for i in range(len(eps)):
            sigma2_t[i+1] = omega + alpha * (eps[i] ** 2) + beta * sigma2_t[i]
        return sigma2_t

    def neg_loglik(self, theta: np.ndarray, eps: np.ndarray, sigma2_0: float) -> float:
        """theta = (omega, alpha, beta, nu). Negative Student-t log-likelihood of eps given the variance recursion.
        The t is scaled to unit variance: z = eps / sigma_t, density of z * sqrt(nu / (nu - 2)) with the Jacobian."""
        l_t = np.zeros(len(eps))
        niu = theta[3]
        sigma2_t = Garch11.variance(eps, theta[0], theta[1], theta[2], sigma2_0)
        for i in range(len(eps)):
            l_t[i] = special.gammaln((niu+1) / 2) - special.gammaln(niu / 2) - 1/2 * np.log(np.pi * (niu -2)) - 1/2 * np.log(sigma2_t[i]) - (niu+1)/2 * np.log(1 + eps[i] ** 2 / (sigma2_t[i] * (niu -2)))
        return -np.sum(l_t)

    def start_values(self, eps: np.ndarray, sigma2_0: float) -> np.ndarray:
        """Grid over (alpha, beta), omega by variance targeting, nu = 8; return the best-likelihood point."""
        alpha = [0.01, 0.05, 0.1, 0.2]
        beta = [0.5,0.7,0.8,0.9,0.95]
        vr = np.var(eps)
        cand = []
        scores = []
        for i in range(len(alpha)):
            for j in range(len(beta)):
                if (alpha[i] + beta[j]) < 1:
                    theta = np.array([vr * (1-alpha[i]-beta[j]), alpha[i], beta[j], 8])
                    cand.append(theta)
                    scores.append(self.neg_loglik(theta, eps, sigma2_0))
        index = np.argmin(scores)
        return cand[index]

    def fit(self, eps: np.ndarray) -> "Garch11":
        """Rescale by SCALE, minimise neg_loglik with bounds and the alpha + beta < 1 constraint, store self.params
        (omega converted back to the original scale) and self.sigma2_last."""
        x = SCALE * eps
        sigma2_0 = Garch11.backcast(x)
        x0 = self.start_values(x, sigma2_0)
        bounds = [
            [1e-8, 10 * np.var(x)],              # omega > 0 keeps sigma2 > 0 (log(0) during the search)
            [0,1],
            [0,1],
            [2.05,500]
        ]
        res = optimize.minimize(self.neg_loglik,x0 ,args = (x, sigma2_0), method="SLSQP",bounds=bounds, constraints=({'type': 'ineq', 'fun': lambda th: 1 - th[1] - th[2]}))
        if not res.success:
            raise ValueError("Optimizer failed to converge")
        theta_star =  res.x
        sigma2_t = Garch11.variance(x, theta_star[0], theta_star[1], theta_star[2], sigma2_0)
        theta_star[0] = theta_star[0]/(SCALE ** 2)
        self.sigma2_last = sigma2_t[-1] / (SCALE ** 2)
        self.params = Garch11Params(theta_star[0], theta_star[1], theta_star[2], theta_star[3])
        return self

    def simulate(self, n_paths: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
        """Shocks eps of shape (n_paths, n_days), starting from self.sigma2_last. Each day:
        sigma2_t = omega + alpha * eps_{t-1}**2 + beta * sigma2_{t-1}, eps_t = sigma_t * z_t, z_t unit-variance t(nu)."""

        result = np.zeros(shape=(n_paths, n_days))
        current_sigma = self.sigma2_last
        for i in range(n_days):
            zs = stats.t.rvs(self.params.nu, size=(n_paths, ), random_state=rng) * np.sqrt((self.params.nu-2)/self.params.nu)
            eps = np.sqrt(current_sigma) * zs
            result[:, i] = eps
            current_sigma = self.params.omega + self.params.alpha * eps ** 2  + self.params.beta * current_sigma
        return result
