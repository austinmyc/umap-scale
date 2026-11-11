"""Epoch-budget laws: prediction from frozen coefficients, and post-hoc fitting.

A law is ln E = sum_i beta_i * basis_i, with bases built from L = ln n,
I = ln d (intrinsic dimension) and d. The frozen, pre-registered laws live in
laws/frozen_laws.json and must not be refit on test data.
"""
import json, os
import numpy as np

FROZEN = os.path.join(os.path.dirname(__file__), "..", "..", "laws", "frozen_laws.json")
BASES = {
    "1": lambda L, I: np.ones_like(L),
    "L": lambda L, I: L,
    "I": lambda L, I: I,
    "L*I": lambda L, I: L * I,
    "L/d": lambda L, I: L / np.exp(I),
    "L^2": lambda L, I: L ** 2,
    "I^2": lambda L, I: I ** 2,
}
FORMS = {
    "power_law": ["1", "L"],
    "theory_1_over_d": ["1", "L", "L/d"],
    "reduced_quadratic": ["1", "L", "I", "L*I"],
    "full_quadratic": ["1", "L", "I", "L^2", "L*I", "I^2"],
}


def load_frozen(path=FROZEN):
    return json.load(open(path))


def predict(law, n, d):
    """Predicted epochs for n points with intrinsic dimension d."""
    n = np.asarray(n, float); d = np.asarray(d, float)
    if "rule" in law:
        return np.where(n <= 10000, 500.0, 200.0)
    L, I = np.log(n), np.log(d)
    return np.exp(sum(b * BASES[nm](L, I) for b, nm in zip(law["beta"], law["basis"])))


def fit(basis, n, d, epochs_needed, ridge=0.0):
    """Least squares (optionally ridge on standardized non-intercept columns) of ln E on the basis.
    Returns a law dict usable by `predict`."""
    L, I = np.log(np.asarray(n, float)), np.log(np.asarray(d, float))
    y = np.log(np.asarray(epochs_needed, float))
    A = np.column_stack([BASES[b](L, I) for b in basis])
    mu, sd = A[:, 1:].mean(0), A[:, 1:].std(0) + 1e-12
    Z = np.column_stack([np.ones(len(y)), (A[:, 1:] - mu) / sd])
    w = np.linalg.solve(Z.T @ Z + np.diag([0.0] + [ridge] * (Z.shape[1] - 1)), Z.T @ y)
    beta = np.concatenate([[w[0] - np.sum(w[1:] * mu / sd)], w[1:] / sd])
    return {"basis": list(basis), "beta": beta.tolist()}
