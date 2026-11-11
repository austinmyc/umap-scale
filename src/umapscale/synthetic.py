"""Synthetic manifolds with exactly known intrinsic dimension d and controlled geometry.

Latent points z in R^d are mapped into R^ambient by a smooth random Fourier-feature
map: phi(z) = [sin(Wz + b), cos(Wz + b)] / sqrt(m), with m = 64 frequencies
(W ~ N(0, freq^2)), followed by a random orthonormal projection to `ambient` dims
and per-coordinate standardization. The Jacobian of phi has rank d almost surely
(m >= d), and a generic linear projection of a d-manifold into more than 2d
dimensions is injective (Whitney), so the intrinsic dimension is exactly d for
d < ambient / 2 (d <= 24 with ambient = 50).

Geometry is controlled across d:
- family "uniform": z ~ Uniform[0, 1]^d. Curvature per latent direction is set by
  `freq` and is the same for every d. No labels.
- family "clusters": K Gaussian blobs whose expected center separation and
  within-blob spread are both fixed in absolute distance (per-dimension scales
  shrink as 1/sqrt(d)), so the separation-to-width ratio does not change with d.

A realization fixes both the random geometry (W, b, projection, centers) and the
sample; different realizations give independent manifolds of the same type.
"""
import numpy as np

N_FREQ = 64


def manifold(n, d, ambient=50, realization=0, family="uniform", freq=1.0,
             n_clusters=10, separation=2.0, width=0.5):
    geo = np.random.default_rng([d, realization, 7919])        # geometry of this realization
    rng = np.random.default_rng([d, realization, 104729])      # sample of this realization
    y = None
    if family == "uniform":
        Z = rng.uniform(0.0, 1.0, (n, d))
    elif family == "clusters":
        centers = geo.normal(scale=separation / np.sqrt(2 * d), size=(n_clusters, d))
        y = rng.integers(0, n_clusters, n)
        Z = centers[y] + rng.normal(scale=width / np.sqrt(2 * d), size=(n, d))
    else:
        raise ValueError(f"unknown family {family}")
    W = geo.normal(scale=freq, size=(d, N_FREQ))
    b = geo.uniform(0, 2 * np.pi, N_FREQ)
    F = np.concatenate([np.sin(Z @ W + b), np.cos(Z @ W + b)], axis=1) / np.sqrt(N_FREQ)
    Q, _ = np.linalg.qr(geo.normal(size=(F.shape[1], ambient)))   # orthonormal columns
    X = (F @ Q).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    return X, y


def parse(spec):
    """'synthetic:d[:ambient[:realization[:family]]]'"""
    p = spec.split(":")[1:]
    return dict(d=int(p[0]), ambient=int(p[1]) if len(p) > 1 else 50,
                realization=int(p[2]) if len(p) > 2 else 0, family=p[3] if len(p) > 3 else "uniform")


def from_spec(spec, n=1_000_000):
    a = parse(spec)
    return manifold(n, a["d"], ambient=a["ambient"], realization=a["realization"], family=a["family"])
