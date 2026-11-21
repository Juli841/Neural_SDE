"""PCA factor extraction for yield-curve changes."""
import numpy as np


def fit_pca(X, k=3):
    """PCA on a (T, n) array/DataFrame via eigendecomposition of the covariance.

    Returns mean (n,), V (n, k) loadings with columns sorted by variance
    (sign fixed so the largest-magnitude entry is positive), explained (k,) variance ratios.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(X - mean, rowvar=False))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vecs *= np.sign(vecs[np.abs(vecs).argmax(axis=0), range(vecs.shape[1])])
    return mean, vecs[:, :k], vals[:k] / vals.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)) @ np.diag([5, 3, 1, .5, .2, .1])
    m, V, ev = fit_pca(X, 3)
    assert V.shape == (6, 3) and np.allclose(V.T @ V, np.eye(3)) and ev[0] > ev[1] > ev[2]
    print("ok", ev)
