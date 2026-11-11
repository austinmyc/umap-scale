"""Embedding quality measures."""
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def fuzzy_graph(X, knn_graph, n_neighbors=15, seed=0):
    """UMAP's symmetrized fuzzy neighbor graph (the graph the layout optimizes)."""
    from umap.umap_ import fuzzy_simplicial_set
    i, d = knn_graph
    G, _, _ = fuzzy_simplicial_set(X, n_neighbors, np.random.RandomState(seed), "euclidean",
                                   knn_indices=i, knn_dists=d)
    return G.tocoo()


class Evaluator:
    """Precomputes high-dimensional neighbors for a fixed evaluation subset so that
    many embeddings of the same data can be scored cheaply and comparably.

    If `graph` (UMAP's fuzzy graph) is given, also reports the layout objective UMAP's
    SGD descends in expectation: weighted attraction over graph edges plus
    `neg_rate` times the mean repulsion over random pairs, with UMAP's default
    low-dimensional kernel q = 1 / (1 + a * dist^(2b)) (min_dist = 0.1, spread = 1)."""

    def __init__(self, X, y=None, n_eval=5000, k=15, seed=0, graph=None, neg_rate=5.0, n_rep_pairs=200_000):
        rng = np.random.default_rng(seed)
        self.X, self.y, self.k = X, y, k
        self.graph, self.neg_rate = graph, neg_rate
        if graph is not None:
            from umap.umap_ import find_ab_params
            self.a, self.b = find_ab_params(1.0, 0.1)
            ra, rb = rng.integers(0, len(X), n_rep_pairs), rng.integers(0, len(X), n_rep_pairs)
            self.rep_pairs = (ra[ra != rb], rb[ra != rb])
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

    def umap_loss(self, E):
        q = lambda d2: 1.0 / (1.0 + self.a * np.power(np.maximum(d2, 0.0), self.b))
        G = self.graph
        d2 = np.sum((E[G.row] - E[G.col]) ** 2, axis=1)
        attract = float(np.sum(G.data * -np.log(np.clip(q(d2), 1e-12, 1.0))) / np.sum(G.data))
        d2r = np.sum((E[self.rep_pairs[0]] - E[self.rep_pairs[1]]) ** 2, axis=1)
        repulse = float(np.mean(-np.log(np.clip(1.0 - q(d2r), 1e-4, 1.0))))
        return attract + self.neg_rate * repulse, attract, repulse

    def __call__(self, E, accuracy=True):
        out = {"recall15": self.recall(E),
               "knn_acc": self.knn_accuracy(E) if accuracy else None,
               "global_spearman": self.global_spearman(E)}
        if self.graph is not None:
            out["umap_loss"], out["loss_attract"], out["loss_repulse"] = self.umap_loss(E)
        return out
