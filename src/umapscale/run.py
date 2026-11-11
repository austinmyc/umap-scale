"""One UMAP layout run with optional optimizer modifications.

Modifications act only on the full-resolution layout. UMAP 0.6's recursive
initialization runs its own Adam layouts on coarsened graphs through the same
kernel; those are left untouched (an earlier version of this experiment
accidentally modified them too).

- neg_schedule: callable f(progress in [0,1]) -> negatives per positive edge
  (UMAP's default is a constant 5). Adam optimizer only.
- block_size: number of source points per Adam update block within an epoch
  (UMAP 0.6 uses all points, i.e. one update per point per epoch, computed from
  positions frozen at the start of the epoch). Adam optimizer only.
"""
import time, warnings
import numpy as np
import umap
import umap.layouts as _L

warnings.filterwarnings("ignore", message=".*compatibility_layout.*")
warnings.filterwarnings("ignore", message=".*knn_search_index.*")
warnings.filterwarnings("ignore", message=".*n_jobs value.*")
_ORIG_ADAM = _L.optimize_layout_euclidean_single_epoch_adam
_ARG_EPN, _ARG_EPOCH = 11, 14       # positions of epochs_per_negative_sample and epoch index
BASE_NEG_RATE = 5.0

NEG_SCHEDULES = {
    "const5": lambda f: 5.0,
    "ramp10to1": lambda f: 10.0 - 9.0 * f,
    "ramp5to1": lambda f: 5.0 - 4.0 * f,
}


class _Patch:
    def __init__(self, n_full, n_epochs, neg_schedule=None, block_size=None):
        self.n_full, self.n_epochs = n_full, n_epochs
        self.neg, self.block = neg_schedule, block_size
        self.layout_seconds = 0.0

    def kernel(self, *args, **kw):
        if args[0].shape[0] != self.n_full:          # coarse layouts inside recursive init
            return _ORIG_ADAM(*args, **kw)
        args = list(args)
        if self.neg is not None:
            rate = max(self.neg(args[_ARG_EPOCH] / self.n_epochs), 1e-6)
            args[_ARG_EPN] = (args[_ARG_EPN] * (BASE_NEG_RATE / rate)).astype(args[_ARG_EPN].dtype)
        if self.block is not None:
            kw["block_size"] = int(self.block)
        t = time.perf_counter()
        out = _ORIG_ADAM(*args, **kw)
        self.layout_seconds += time.perf_counter() - t
        return out

    def __enter__(self):
        _L.optimize_layout_euclidean_single_epoch_adam = self.kernel
        return self

    def __exit__(self, *exc):
        _L.optimize_layout_euclidean_single_epoch_adam = _ORIG_ADAM


def embed(X, knn_graph, n_epochs, optimizer="adam", seed=42, n_components=2,
          neg_schedule=None, block_size=None, n_neighbors=15, min_dist=0.1, parallel=False):
    """Returns (embedding, info dict with total and layout seconds).

    parallel=False passes random_state=seed: reproducible, but umap-learn then runs
    single-threaded. parallel=True passes no random_state so the layout uses all
    cores; results then vary slightly run to run (the seed only labels the run)."""
    if optimizer == "compatibility":
        if neg_schedule is not None or block_size is not None:
            raise ValueError("schedules are implemented for the Adam optimizer only")
        kw = dict(compatibility_layout=True)
    else:
        kw = dict(optimizer=optimizer, compatibility_layout=False)
    neg = NEG_SCHEDULES[neg_schedule] if isinstance(neg_schedule, str) else neg_schedule
    patch = _Patch(len(X), n_epochs, neg, block_size)
    t = time.perf_counter()
    with patch:
        E = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                      n_epochs=n_epochs, random_state=None if parallel else seed,
                      precomputed_knn=knn_graph, **kw).fit_transform(X)
    total = time.perf_counter() - t
    return E, {"fit_s": total, "layout_s": patch.layout_seconds if optimizer == "adam" else None}
