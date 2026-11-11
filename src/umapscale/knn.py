"""Neighbor-graph construction. umap-learn accepts a precomputed (indices, distances)
pair, so the expensive kNN search can run on GPU (FAISS) while umap-learn's
CPU optimizer does the layout."""
import numpy as np


def knn(X, k=15, backend="umap", seed=0):
    """Return (indices int32 (n,k), distances float32 (n,k)), self included in column 0."""
    if backend == "umap":
        from umap.umap_ import nearest_neighbors
        i, d, _ = nearest_neighbors(X, k, "euclidean", {}, False, np.random.RandomState(seed))
        return i.astype(np.int32), d.astype(np.float32)
    if backend in ("faiss", "faiss-gpu"):
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        index = faiss.IndexFlatL2(X.shape[1])
        if backend == "faiss-gpu":
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(X)
        d2, i = index.search(X, k)
        return i.astype(np.int32), np.sqrt(np.maximum(d2, 0)).astype(np.float32)
    if backend == "sklearn":
        from sklearn.neighbors import NearestNeighbors
        d, i = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
        return i.astype(np.int32), d.astype(np.float32)
    raise ValueError(f"unknown knn backend {backend}")
