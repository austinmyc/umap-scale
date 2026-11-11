"""Dataset loading and downloading.

Every loader returns (X, y) with X float32 of shape (n, D) and y an int label
array, or None for unlabeled data. Downloaded files are treated as untrusted:
the MNIST pickle is read with a restricted unpickler that only rebuilds numpy
arrays, and tabular files are parsed with pandas.

Built-in datasets (fetched with `python -m umapscale.data download NAME`):
  fashion_mnist, mnist, letter, pendigits, plus any other PMLB dataset name.
Large datasets are supplied as prepared arrays (see `prepare` below):
  npy:<path_to_X.npy>[,<path_to_y.npy>]
Synthetic manifolds: synthetic:<intrinsic_dim>[:<ambient_dim>[:<seed>]]
"""
import argparse, gzip, io, os, pickle, subprocess, urllib.request
import numpy as np

DATA_DIR = os.environ.get("UMAPSCALE_DATA", os.path.join(os.getcwd(), "data"))
PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"
INTEGER_VALUED = {"letter", "pendigits"}


# ---------------------------------------------------------------- downloads
def download(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    if name == "fashion_mnist":
        dst = os.path.join(DATA_DIR, "fashion-mnist")
        if not os.path.exists(dst):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/zalandoresearch/fashion-mnist", dst],
                           check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    elif name == "mnist":
        dst = os.path.join(DATA_DIR, "mnist.pkl.gz")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(
                "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz", dst)
    else:
        dst = os.path.join(DATA_DIR, f"{name}.tsv.gz")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(PMLB_URL.format(name), dst)
    return dst


# ---------------------------------------------------------------- loaders
class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_reconstruct" and module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            from numpy._core.multiarray import _reconstruct
            return _reconstruct
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _pca50(X):
    from sklearn.decomposition import PCA
    return PCA(50, random_state=0).fit_transform(X).astype(np.float32)


def _load_fashion_mnist():
    base = os.path.join(DATA_DIR, "fashion-mnist", "data", "fashion")
    out = []
    for kind in ("train", "t10k"):
        with gzip.open(f"{base}/{kind}-images-idx3-ubyte.gz") as f:
            Xk = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        with gzip.open(f"{base}/{kind}-labels-idx1-ubyte.gz") as f:
            yk = np.frombuffer(f.read(), np.uint8, offset=8)
        out.append((Xk, yk))
    X = np.vstack([o[0] for o in out]).astype(np.float32) / 255.0
    return _pca50(X), np.concatenate([o[1] for o in out]).astype(int)


def _load_mnist():
    with gzip.open(os.path.join(DATA_DIR, "mnist.pkl.gz"), "rb") as f:
        tr, va, te = _NumpyOnlyUnpickler(io.BytesIO(f.read()), encoding="latin1").load()
    X = np.vstack([tr[0], va[0], te[0]]).astype(np.float32)
    return _pca50(X), np.concatenate([tr[1], va[1], te[1]]).astype(int)


def _load_pmlb(name, dequantize=False, seed=0):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.tsv.gz"), sep="\t")
    y = df.pop("target").to_numpy()
    X = df.to_numpy(np.float64)
    if dequantize:
        X = X + np.random.default_rng(seed).uniform(-0.5, 0.5, X.shape)
    return StandardScaler().fit_transform(X).astype(np.float32), y


def _load_npy(spec):
    parts = spec.split(",")
    X = np.load(parts[0], mmap_mode=None).astype(np.float32)
    y = np.load(parts[1]) if len(parts) > 1 else None
    return X, y


def load(name, dequantize=False):
    """Load a dataset by name. `dequantize` only affects integer-valued tabular data
    and is used for intrinsic-dimension estimation, never for the embedding itself."""
    if name == "fashion_mnist":
        return _load_fashion_mnist()
    if name == "mnist":
        return _load_mnist()
    if name.startswith("npy:"):
        return _load_npy(name[4:])
    if name.startswith("synthetic:"):
        from .synthetic import from_spec
        return from_spec(name)
    return _load_pmlb(name, dequantize=dequantize and name in INTEGER_VALUED)


def is_integer_valued(name):
    return name in INTEGER_VALUED


# ---------------------------------------------------------------- preparing large datasets
def prepare_h5ad(path, out_prefix, n_pcs=50, label_key=None):
    """Single-cell AnnData -> standard PCA-50 arrays (normalize, log1p, 2000 HVGs, scale, PCA)."""
    import scanpy as sc
    ad = sc.read_h5ad(path)
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
    sc.pp.scale(ad, max_value=10); sc.tl.pca(ad, n_comps=n_pcs)
    np.save(f"{out_prefix}_X.npy", ad.obsm["X_pca"].astype(np.float32))
    if label_key:
        np.save(f"{out_prefix}_y.npy", ad.obs[label_key].astype("category").cat.codes.to_numpy())


def prepare_embeddings(path, out_prefix, n_pcs=50):
    """Generic (n, D) float array (e.g. CLIP features, SIFT) -> PCA-50 arrays."""
    from sklearn.decomposition import PCA
    X = np.load(path).astype(np.float32)
    if X.shape[1] > n_pcs:
        X = PCA(n_pcs, random_state=0).fit_transform(X).astype(np.float32)
    np.save(f"{out_prefix}_X.npy", X)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("names", nargs="+")
    h = sub.add_parser("prepare-h5ad"); h.add_argument("path"); h.add_argument("out_prefix"); h.add_argument("--label-key")
    e = sub.add_parser("prepare-embeddings"); e.add_argument("path"); e.add_argument("out_prefix")
    a = ap.parse_args()
    if a.cmd == "download":
        for nm in a.names:
            print(nm, "->", download(nm))
    elif a.cmd == "prepare-h5ad":
        prepare_h5ad(a.path, a.out_prefix, label_key=a.label_key)
    else:
        prepare_embeddings(a.path, a.out_prefix)
