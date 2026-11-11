"""Embedding quality measures."""
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class Evaluator:
    """Precomputes high-dimensional neighbors for a fixed evaluation subset so that
    many embeddings of the same data can be scored cheaply and comparably."""

    def __init__(self, X, y=None, n_eval=5000, k=15, seed=0):
        rng = np.random.default_rng(seed)
        self.X, self.y, self.k = X, y, k
        self.idx = rng.choice(len(X), min(n_eval, len(X)), replace=False)
        self.hd = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[self.idx], return_distance=False)[:, 1:]
        a, b = rng.choice(len(X), 20000), rng.choice(len(X), 20000)
        self.pairs = (a[a != b], b[a != b])
        self.hd_pair = np.linalg.norm(X[self.pairs[0]] - X[self.pairs[1]], axis=1)

    def recall(self, E):
        ld = NearestNeighbors(n_neighbors=self.k + 1).fit(E).kneighbors(E[self.idx], return_distance=False)[:, 1:]
        return float(np.mean([len(set(a) & set(b)) / self.k for a, b in zip(self.hd, ld)]))

    def knn_accuracy(self, E, max_points=100_000, seed=0):
        if self.y is None:
            return None
        idx = np.arange(len(E))
        if len(E) > max_points:
            idx = np.random.default_rng(seed).choice(len(E), max_points, replace=False)
        return float(cross_val_score(KNeighborsClassifier(5), E[idx], self.y[idx], cv=5, n_jobs=-1).mean())

    def global_spearman(self, E):
        """Spearman correlation between input-space and embedding distances on random pairs."""
        from scipy.stats import spearmanr
        ld = np.linalg.norm(E[self.pairs[0]] - E[self.pairs[1]], axis=1)
        return float(spearmanr(self.hd_pair, ld).correlation)

    def __call__(self, E, accuracy=True):
        return {"recall15": self.recall(E),
                "knn_acc": self.knn_accuracy(E) if accuracy else None,
                "global_spearman": self.global_spearman(E)}
