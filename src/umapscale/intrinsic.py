"""Intrinsic-dimension estimators (label-free, computed on the subset to be embedded)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _knn_dists(X, k, n_query=4000, seed=0):
    idx = np.random.default_rng(seed).choice(len(X), min(n_query, len(X)), replace=False)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[idx])
    return np.maximum(d[:, 1:], 1e-12)


def mle(X, k=15, n_query=4000, seed=0):
    """Levina-Bickel MLE with MacKay-Ghahramani averaging (the pre-registered estimator)."""
    d = _knn_dists(X, k, n_query, seed)
    return float(1.0 / np.mean(np.log(d[:, -1:] / d[:, :-1])))


def twonn(X, n_query=4000, seed=0, discard=0.1):
    """TwoNN (Facco et al. 2017), slope through the origin after discarding the top 10%."""
    d = _knn_dists(X, 2, n_query, seed)
    mu = np.sort(d[:, 1] / d[:, 0])
    mu = mu[np.isfinite(mu) & (mu > 1)]
    m = int((1 - discard) * len(mu))
    F = np.arange(1, m + 1) / len(mu)
    x, yv = np.log(mu[:m]), -np.log(1 - F)
    return float((x @ yv) / (x @ x))
