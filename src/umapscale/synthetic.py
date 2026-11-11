"""Synthetic manifolds with known intrinsic dimension.

A d-dimensional latent cube is mapped smoothly into the ambient space with a
fixed random Fourier-feature embedding, then randomly rotated. The intrinsic
dimension is exactly d by construction; the map is smooth and (with
overwhelming probability) injective for the frequencies used. Points are drawn
from a mixture of `n_clusters` Gaussian blobs in latent space so that the data
also has cluster labels for kNN accuracy.
"""
import numpy as np


def manifold(n, d, ambient=50, n_clusters=10, freq=1.0, seed=0):
    rng = np.random.default_rng(seed)
    geo = np.random.default_rng(10_000 + d)          # embedding fixed per d, independent of n and seed
    centers = geo.uniform(-1, 1, (n_clusters, d))
    y = rng.integers(0, n_clusters, n)
    Z = centers[y] + rng.normal(scale=0.35, size=(n, d))
    m = ambient // 2
    W = geo.normal(scale=freq, size=(d, m))
    b = geo.uniform(0, 2 * np.pi, m)
    F = np.concatenate([np.sin(Z @ W + b), np.cos(Z @ W + b)], axis=1) / np.sqrt(m)
    Q, _ = np.linalg.qr(geo.normal(size=(F.shape[1], ambient)))
    X = (F @ Q).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    return X, y


def from_spec(spec, n=None):
    """'synthetic:d[:ambient[:seed]]' -> (X, y). The sweep subsamples to the requested
    size, so the pool is generated at the largest size needed (default 1M)."""
    parts = spec.split(":")[1:]
    d = int(parts[0]); ambient = int(parts[1]) if len(parts) > 1 else 50
    seed = int(parts[2]) if len(parts) > 2 else 0
    return manifold(n or 1_000_000, d, ambient=ambient, seed=seed)
