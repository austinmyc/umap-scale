"""Label-free data features computable before (or at the very start of) a UMAP run.

Usage: python -m umapscale.features --out results/features/features.jsonl

For every development case (real and synthetic), rebuilds the exact subset that was
embedded and computes:

  covariance:  log_pr (participation ratio), top10_var, eig_decay
  density:     cv_knn_dist (CV of the 15th-NN distance), log_rel_nn, id_cv (spread of
               per-point MLE dimension)
  kNN graph:   hubness (skew of in-degree), mutual_frac (fraction of mutual 15-NN edges)
  spectrum of the UMAP fuzzy graph (normalized Laplacian eigenvalues lambda_2..lambda_12):
               log_gap2 = ln lambda_2 (slowest diffusion mode), log_gap11 = ln lambda_11,
               cluster_gap = ln(lambda_11 / lambda_10), n_components
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from . import data, synthetic
from .metrics import fuzzy_graph

REAL = {"fashion_mnist": [5000, 10000, 20000, 40000, 70000], "mnist": [5000, 10000, 20000, 40000, 70000],
        "letter": [2500, 5000, 10000, 20000], "pendigits": [1375, 2750, 5500, 10992]}
SYN_N = [2500, 5000, 10000, 20000, 40000]


def cases():
    for ds, ns in REAL.items():
        yield ds, ns, lambda ds=ds: data.load(ds, dequantize=data.is_integer_valued(ds))[0]
    for fam in ["uniform", "clusters"]:
        for d in [2, 4, 8, 12, 16, 24]:
            for r in range(3):
                name = f"synthetic:{d}:50:{r}" + (":clusters" if fam == "clusters" else "")
                yield name, SYN_N, lambda name=name: synthetic.from_spec(name, n=max(SYN_N))[0]


def spectral(G):
    """lambda_conn (smallest normalized-Laplacian eigenvalue > 1e-6 among the 12 smallest) and K (components)
    of UMAP's fuzzy graph G (scipy sparse, symmetric)."""
    G = G.tocsr()
    K = int(connected_components(G, directed=False)[0])
    deg = np.asarray(G.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(np.maximum(deg, 1e-12)))
    mu = np.sort(eigsh(Dm @ G @ Dm, k=12, which="LA", tol=1e-5, maxiter=20000)[0])[::-1]
    lam = np.clip(1 - mu, 0, None)
    nz = lam[lam > 1e-6]
    return {"lambda_conn": float(nz[0]) if len(nz) else float("nan"), "n_components": K, "laplacian_eigs": lam.tolist()}


def compute(X, k=15, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    Xc = X - X.mean(0)
    lam = np.clip(np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1], 1e-12, None)
    m = min(len(lam), 30)
    out["log_pr"] = float(np.log(lam.sum() ** 2 / (lam ** 2).sum()))
    out["top10_var"] = float(lam[:10].sum() / lam.sum())
    out["eig_decay"] = float(-np.polyfit(np.log(np.arange(1, m + 1)), np.log(lam[:m]), 1)[0])

    dist, idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X).kneighbors(X)
    dist = np.maximum(dist, 1e-12)
    dk = dist[:, -1]
    out["cv_knn_dist"] = float(dk.std() / dk.mean())
    a, b = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
    out["log_rel_nn"] = float(np.log(np.median(dk) / np.median(np.linalg.norm(X[a] - X[b], axis=1))))
    inv = np.mean(np.log(dist[:, -1:] / dist[:, 1:-1]), axis=1)          # per-point 1/d
    inv = inv[inv > 0]
    out["id_cv"] = float(np.std(1 / inv) / np.mean(1 / inv))
    cnt = np.bincount(idx[:, 1:].ravel(), minlength=len(X)).astype(float)
    out["hubness"] = float(((cnt - cnt.mean()) ** 3).mean() / cnt.std() ** 3)
    A = sp.csr_matrix((np.ones(idx[:, 1:].size), (np.repeat(np.arange(len(X)), k), idx[:, 1:].ravel())),
                      shape=(len(X), len(X)))
    out["mutual_frac"] = float(A.multiply(A.T).nnz / A.nnz)

    G = fuzzy_graph(X, (idx.astype(np.int32), dist.astype(np.float32)), n_neighbors=k).tocsr()
    out["n_components"] = int(connected_components(G, directed=False)[0])
    deg = np.asarray(G.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(deg))
    S = Dm @ G @ Dm
    mu = np.sort(eigsh(S, k=12, which="LA", tol=1e-5, maxiter=20000)[0])[::-1]
    lam_g = np.clip(1 - mu, 1e-9, None)                                     # Laplacian eigenvalues, ascending
    out["log_gap2"] = float(np.log(lam_g[1]))
    out["log_gap11"] = float(np.log(lam_g[10]))
    out["cluster_gap"] = float(np.log(lam_g[10] / lam_g[9]))
    out["laplacian_eigs"] = lam_g.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/features/features.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["dataset"], json.loads(l)["n"]) for l in open(a.out)}
    for ds, ns, loader in cases():
        if all((ds, n) in done for n in ns):
            continue
        X_all = loader()
        perm = np.random.default_rng(0).permutation(len(X_all))
        for n in ns:
            if (ds, n) in done:
                continue
            t = time.time()
            f = {"dataset": ds, "n": n, **compute(np.ascontiguousarray(X_all[perm[:n]]))}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(f) + "\n")
            print(ds, n, f"{time.time() - t:.1f}s", {k: round(v, 3) for k, v in f.items()
                                                     if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main()
