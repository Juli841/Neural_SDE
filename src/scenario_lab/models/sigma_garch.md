# GARCH(1,1) with Student-t: what `sigma_garch.py` has to compute

Companion to `sigma_garch.py` (class `Garch11`, one factor at a time). Indices are 0-based, as in numpy.
Input: `eps`, the VAR(1) residuals of one factor, length $T$, so `eps[0]` ... `eps[T-1]`, mean about 0.

## 1. The model

$$\varepsilon_t = \sigma_t z_t, \qquad z_t \sim \text{Student-t}(\nu)\ \text{scaled to unit variance}$$

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$$

- $\omega > 0$: base level. $\alpha$: reaction to yesterday's shock. $\beta$: memory of yesterday's variance.
- Persistence $\alpha + \beta < 1$ (stationary). Long-run variance $\omega / (1 - \alpha - \beta)$.
- Unit-variance t: draw $t_\nu$ and multiply by $\sqrt{(\nu - 2)/\nu}$, so $\text{Var}(z) = 1$ (needs $\nu > 2$).

## 2. `backcast(eps)` - the starting variance

$\tau = \min(75, T)$, weights $w_i = 0.94^{\,i}$ for $i = 0..\tau-1$, normalised to sum 1:

$$\sigma_0^2 = \sum_{i=0}^{\tau-1} w_i\, \varepsilon_i^2 \quad\text{with}\quad w_i \leftarrow \frac{0.94^i}{\sum_j 0.94^j}$$

This is the variance of the first observation `eps[0]`. It is a weighted average of the first 75 squared residuals, heavier on the earliest ones.

## 3. `variance(eps, omega, alpha, beta, sigma2_0)` - the recursion

Return an array `s2` of length $T + 1$:

$$s2[0] = \sigma_0^2 \ (\text{backcast}), \qquad s2[t] = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\, s2[t-1], \quad t = 1..T$$

- `s2[t]` is the conditional variance of `eps[t]` for $t < T$ (use it in the likelihood).
- **`s2[T]` is the forecast for the day after the data ends.** It becomes `sigma2_last`, the starting variance of the simulation.
- It is a loop over $t$ (the recursion cannot be vectorised).

**Check:** with $\alpha = \beta = 0$ every entry from index 1 on equals $\omega$.

## 4. `neg_loglik(theta, eps, sigma2_0)` - the objective

$\theta = (\omega, \alpha, \beta, \nu)$. Compute `s2 = variance(...)` and use `s2[:T]` (drop the forecast). With $z_t = \varepsilon_t / \sigma_t$, the log-density of one observation under the unit-variance t is

$$\ell_t = \ln\Gamma\!\left(\tfrac{\nu+1}{2}\right) - \ln\Gamma\!\left(\tfrac{\nu}{2}\right) - \tfrac12 \ln\!\big(\pi(\nu-2)\big) - \tfrac12 \ln \sigma_t^2 - \tfrac{\nu+1}{2}\ln\!\left(1 + \frac{\varepsilon_t^2}{\sigma_t^2\,(\nu-2)}\right)$$

$$\text{neg\_loglik} = -\sum_{t=0}^{T-1} \ell_t$$

- The $-\tfrac12\ln\sigma_t^2$ term is the Jacobian of $\varepsilon = \sigma z$. It is easy to forget.
- `scipy.special.gammaln` gives $\ln\Gamma$.
- Equivalent: `stats.t.logpdf(eps / sigma, nu, scale=np.sqrt((nu-2)/nu)) - np.log(sigma)`.

## 5. `start_values(eps, sigma2_0)`

A small grid, keep the best (lowest `neg_loglik`):

- $\alpha \in \{0.01, 0.05, 0.1, 0.2\}$, $\beta \in \{0.5, 0.7, 0.8, 0.9, 0.95\}$ (skip pairs with $\alpha + \beta \ge 1$). Exact values are not important.
- $\omega$ by variance targeting: $\omega = \text{Var}(\varepsilon)\,(1 - \alpha - \beta)$, so the model's long-run variance equals the sample variance.
- $\nu = 8$.

## 6. `fit(eps)`

1. **Rescale:** $x = \text{eps} \cdot 100$ (`SCALE`). Fit everything on $x$.
2. `sigma2_0 = backcast(x)`, start from `start_values`.
3. `scipy.optimize.minimize(neg_loglik, x0, method="SLSQP", bounds=..., constraints=...)`:
   - bounds: $\omega \in [0, \infty)$ (use a large upper bound such as $10\,\text{Var}(x)$), $\alpha, \beta \in [0, 1]$, $\nu \in [2.05, 500]$
   - constraint: $\alpha + \beta < 1$, e.g. `{"type": "ineq", "fun": lambda th: 1 - th[1] - th[2]}`
4. **Convert back:** $\omega_{\text{orig}} = \omega / 100^2$. $\alpha, \beta, \nu$ are scale-free.
5. Store `params` and `sigma2_last` = last element of `variance(...)` on the original scale (divide the scaled one by $100^2$).

## 7. `simulate(n_paths, n_days, rng)`

For all paths at once (vectors of length `n_paths`), looping over days:

$$\sigma^2 \leftarrow \text{sigma2\_last}\ (\text{same for every path})$$

$$\text{for each day: } z = t_\nu \sqrt{(\nu-2)/\nu},\quad \varepsilon = \sqrt{\sigma^2}\, z,\quad \sigma^2 \leftarrow \omega + \alpha\,\varepsilon^2 + \beta\,\sigma^2$$

The first simulated day uses `sigma2_last`, the variance of the day after the data ends. The variance now differs between paths from day 2 on, which is the point: paths that get a big shock become more volatile. Return `eps` with shape `(n_paths, n_days)`.

## 8. How it plugs into `model.py` later

- Today `fit` takes `sigma` (constant) and `df` per factor from the VAR residuals. With GARCH, run `Garch11().fit(epsilon_hat[:, j])` for each factor $j$ on those residuals.
- `simulate` replaces `eps = p.sigma * ... * t.rvs(...)` by the three GARCH shock arrays, one per factor; the VAR(1) update and the curve reconstruction stay the same.

## Checks before trusting it

1. $\alpha = \beta = 0$: `variance` returns $\omega$ everywhere (after index 0).
2. Simulate a long series from known $(\omega, \alpha, \beta, \nu)$, fit it back: parameters should be recovered (roughly; $\nu$ is the hardest). Persistence $\alpha + \beta$ is the most important.
3. Simulated average variance is close to $\omega / (1 - \alpha - \beta)$.
4. Fitted $\alpha + \beta < 1$ on the real factors, and the log-likelihood is better than the constant-volatility model's on the same residuals.
