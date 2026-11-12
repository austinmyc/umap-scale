"""In-training monitoring and stopping rules for the UMAP 0.6 Adam layout.

The Adam kernel is wrapped (full-resolution layout only). Every `every` epochs the
current embedding is snapshotted and these label-free statistics are computed on a
fixed sample of points:

  recall_mon   15-NN recall of the sample against the input-space kNN graph UMAP already has
  nn_stab      Jaccard overlap of each sample point's embedding 15-NN with the previous check
  cov_res      residual covariance: ||C_t - C_prev||_F / ||C_t||_F of the 2-D embedding covariance
  id_emb       Levina-Bickel MLE dimension of the embedding (15-NN); its relative change is the
               "residual d"
  disp         median displacement since the previous check / RMS radius of the embedding

`recall15` (5,000 separate evaluation points) is recorded only for scoring, never used by a rule.

A live rule may fire at epoch t*. Then either training stops at once ("stop") or the learning
rate is decayed linearly to 0 over max(min_tail, tail_frac * t*) more epochs and training stops
("anneal").
"""
import numpy as np
import umap
import umap.layouts as _L
from scipy.spatial import cKDTree
from .run import _ORIG_ADAM

_ARG_ALPHA, _ARG_EPOCH = 10, 14


class _Stop(Exception):
    pass


def _knn2d(Y, q, k=15):   # any output dimension
    return cKDTree(Y).query(Y[q], k + 1)[1][:, 1:]


def _recall(nb_in, nb_out):
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(nb_in, nb_out)]))


def _id_mle(Y, q, k=15):
    d = cKDTree(Y).query(Y[q], k + 1)[0][:, 1:]
    d = np.maximum(d, 1e-12)
    return float(1.0 / np.mean(np.log(d[:, -1:] / d[:, :-1])))


class Monitor:
    def __init__(self, n_full, knn_idx, every=5, n_mon=2000, n_eval=5000, seed=0,
                 rule=None, mode="stop", tail_frac=0.25, min_tail=10, eval_idx=None, eval_nb=None):
        rng = np.random.default_rng(seed)
        self.n_full, self.every = n_full, every
        self.q_mon = rng.choice(n_full, min(n_mon, n_full), replace=False)
        self.q_eval = rng.choice(n_full, min(n_eval, n_full), replace=False) if eval_idx is None else np.asarray(eval_idx)
        # Monitor: the input neighbors UMAP already has (its kNN table includes the point itself in column 0,
        # so it holds k_mon = width - 1 true neighbors); compared with the same number of embedding neighbors.
        self.k_mon = knn_idx.shape[1] - 1
        self.in_mon = knn_idx[self.q_mon, 1:]
        # Scoring: exact 15-NN of the evaluation points (pass Evaluator.hd so scores match the sweeps' recall15).
        self.in_eval = knn_idx[self.q_eval, 1:] if eval_nb is None else np.asarray(eval_nb)
        self.k_eval = self.in_eval.shape[1]
        self.rule, self.mode, self.tail_frac, self.min_tail = rule, mode, tail_frac, min_tail
        self.hist, self.prev, self.fired, self.tail = [], None, None, None
        self.Y = None

    def _check(self, Y, epoch):
        nb = _knn2d(Y, self.q_mon, self.k_mon)
        C = np.cov(Y.T)
        h = {"epoch": epoch, "recall_mon": _recall(self.in_mon, nb), "id_emb": _id_mle(Y, self.q_mon),
             "recall15": _recall(self.in_eval, _knn2d(Y, self.q_eval, self.k_eval))}
        if self.prev is not None:
            pY, pnb, pC = self.prev
            h["nn_stab"] = float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(nb, pnb)]))
            h["cov_res"] = float(np.linalg.norm(C - pC) / np.linalg.norm(C))
            r = np.sqrt(np.mean(np.sum((Y - Y.mean(0)) ** 2, 1)))
            h["disp"] = float(np.median(np.linalg.norm(Y[self.q_mon] - pY, axis=1)) / r)
        self.prev = (Y[self.q_mon].copy(), nb, C)
        self.hist.append(h)

    def kernel(self, *args, **kw):
        if args[0].shape[0] != self.n_full:
            return _ORIG_ADAM(*args, **kw)
        epoch = int(args[_ARG_EPOCH])
        if self.tail is not None:                     # annealing tail after the rule fired
            t0, T, a0 = self.tail
            if epoch - t0 >= T:
                raise _Stop
            args = list(args)
            args[_ARG_ALPHA] = a0 * (1 - (epoch - t0) / T)
        out = _ORIG_ADAM(*args, **kw)
        self.Y = args[0]
        if (epoch + 1) % self.every == 0:
            self._check(np.asarray(args[0]), epoch + 1)
            if self.rule is not None and self.fired is None and self.rule(self.hist):
                self.fired = epoch + 1
                if self.mode == "stop":
                    raise _Stop
                self.tail = (epoch + 1, max(self.min_tail, int(round(self.tail_frac * (epoch + 1)))),
                             float(args[_ARG_ALPHA]))
        return out

    def run(self, X, knn_graph, n_epochs, seed=42, n_neighbors=15, min_dist=0.1, n_components=2, parallel=False):
        _L.optimize_layout_euclidean_single_epoch_adam = self.kernel
        try:
            Y = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_epochs=n_epochs, random_state=None if parallel else seed, n_components=n_components,
                          precomputed_knn=knn_graph, optimizer="adam", compatibility_layout=False).fit_transform(X)
            stopped = None
        except _Stop:
            Y = np.array(self.Y)
            stopped = self.hist[-1]["epoch"] if self.tail is None else int(self.tail[0] + self.tail[1])
        finally:
            _L.optimize_layout_euclidean_single_epoch_adam = _ORIG_ADAM
        final = _recall(self.in_eval, _knn2d(Y, self.q_eval, self.k_eval))
        return Y, {"stopped_at": stopped, "fired_at": self.fired, "final_recall15": final, "hist": self.hist}


OUTDIM_SPECS = [("fashion_mnist", [10000, 40000]), ("mnist", [10000, 40000]), ("letter", [10000]), ("pendigits", [10992])] + \
    [(f"synthetic:{d}:50:0", [10000]) for d in [2, 4, 8, 16]] + [(f"synthetic:{d}:50:0:clusters", [10000]) for d in [2, 8]]


# ---- rules: functions of the history list, True = fire ----
def _two_below(key, tol, rel=False):
    def rule(h):
        vals = [x.get(key) for x in h]
        if rel:
            vals = [None if i == 0 else abs(vals[i] - vals[i - 1]) / abs(vals[i]) for i in range(len(vals))]
        v = [x for x in vals[-2:] if x is not None]
        return len(v) == 2 and max(v) < tol
    return rule


def _two_above(key, tol):
    def rule(h):
        v = [x.get(key) for x in h[-2:]]
        return len(v) == 2 and None not in v and min(v) > tol
    return rule


def _recall_plateau(tol):
    """recall_mon improved by less than tol (relative) over each of the last two checks."""
    def rule(h):
        r = [x["recall_mon"] for x in h]
        return len(r) >= 3 and all(r[-i] < r[-i - 1] * (1 + tol) for i in (1, 2))
    return rule


RULES = {
    "recall_plateau": lambda tol: _recall_plateau(tol),
    "nn_stability": lambda tol: _two_above("nn_stab", tol),
    "cov_residual": lambda tol: _two_below("cov_res", tol),
    "id_residual": lambda tol: _two_below("id_emb", tol, rel=True),
    "displacement": lambda tol: _two_below("disp", tol),
}


def trajectories(out="results/raw/stopping_traj.jsonl", every=5, seed=42, n_components=2, specs=None):
    """Record full-length monitored trajectories (no rule; n_epochs = umap's default) for every
    development case; recall15 uses the same 5,000 evaluation points as the sweeps."""
    import json, os, time
    from sklearn.neighbors import NearestNeighbors
    from . import data, synthetic
    from .features import REAL, SYN_N
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["dataset"], json.loads(l)["n"]) for l in open(out)}
    if specs is None:
        specs = [(ds, ns) for ds, ns in REAL.items()]
        specs += [(f"synthetic:{d}:50:0" + fam, [10000, 40000]) for fam in ["", ":clusters"] for d in [2, 4, 8, 12, 16, 24]]
    for ds, ns in specs:
        if all((ds, n) in done for n in ns):
            continue
        X_all = synthetic.from_spec(ds, n=max(SYN_N))[0] if ds.startswith("synthetic") else data.load(ds)[0]
        perm = np.random.default_rng(0).permutation(len(X_all))
        for n in ns:
            if (ds, n) in done:
                continue
            X = np.ascontiguousarray(X_all[perm[:n]])
            t = time.time()
            from .metrics import Evaluator
            ev = Evaluator(X, n_eval=5000)
            d, i = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(X).kneighbors(X)
            N = 500 if n <= 10000 else 200
            _, info = Monitor(n, i, every=every, eval_idx=ev.idx, eval_nb=ev.hd).run(X, (i.astype(np.int32), d.astype(np.float32)), N,
                                                                      seed=seed, n_components=n_components)
            with open(out, "a") as fh:
                fh.write(json.dumps({"dataset": ds, "n": n, "n_epochs": N, "seed": seed, "n_components": n_components,
                                     "hist": info["hist"]}) + "\n")
            print(ds, n, f"{time.time() - t:.0f}s", flush=True)


def live(rules, out="results/raw/stopping_live.jsonl", seed=42, tail_frac=0.25, min_tail=10, n_components=2, specs=None):
    """Run each rule live in anneal mode: when it fires at t*, decay the learning rate linearly
    to 0 over max(min_tail, tail_frac * t*) epochs, then stop. rules: list of (name, tol)."""
    import json, os, time
    from sklearn.neighbors import NearestNeighbors
    from . import data, synthetic
    from .metrics import Evaluator
    from .features import REAL, SYN_N
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["dataset"], json.loads(l)["n"], json.loads(l)["rule"]) for l in open(out)}
    if specs is None:
        specs = [(ds, ns) for ds, ns in REAL.items()]
        specs += [(f"synthetic:{d}:50:0" + fam, [10000, 40000]) for fam in ["", ":clusters"] for d in [2, 4, 8, 12, 16, 24]]
    for ds, ns in specs:
        X_all = None
        for n in ns:
            todo = [(r, t) for r, t in rules if (ds, n, f"{r} {t}") not in done]
            if not todo:
                continue
            if X_all is None:
                X_all = synthetic.from_spec(ds, n=max(SYN_N))[0] if ds.startswith("synthetic") else data.load(ds)[0]
                perm = np.random.default_rng(0).permutation(len(X_all))
            X = np.ascontiguousarray(X_all[perm[:n]])
            ev = Evaluator(X, n_eval=5000)
            d, i = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(X).kneighbors(X)
            g = (i.astype(np.int32), d.astype(np.float32))
            N = 500 if n <= 10000 else 200
            for r, tol in todo:
                t = time.time()
                m = Monitor(n, i, eval_idx=ev.idx, eval_nb=ev.hd, rule=RULES[r](tol), mode="anneal", tail_frac=tail_frac, min_tail=min_tail)
                _, info = m.run(X, g, N, seed=seed, n_components=n_components)
                with open(out, "a") as fh:
                    fh.write(json.dumps({"dataset": ds, "n": n, "rule": f"{r} {tol}", "n_epochs": N, "n_components": n_components,
                                         "fired_at": info["fired_at"], "stopped_at": info["stopped_at"] or N,
                                         "final_recall15": info["final_recall15"], "seconds": time.time() - t}) + "\n")
                print(ds, n, r, tol, info["fired_at"], info["stopped_at"], round(info["final_recall15"], 4), flush=True)


def grow_runs(out="results/raw/stopping_grow.jsonl", seed=42, n_components=2, specs=None, rewarm=0.3, tol=0.02):
    import json, os, time
    from sklearn.neighbors import NearestNeighbors
    from . import data, synthetic
    from .metrics import Evaluator
    from .features import REAL, SYN_N
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["dataset"], json.loads(l)["n"]) for l in open(out)}
    if specs is None:
        specs = [(ds, ns) for ds, ns in REAL.items()]
        specs += [(f"synthetic:{d}:50:0" + fam, [10000, 40000]) for fam in ["", ":clusters"] for d in [2, 4, 8, 12, 16, 24]]
    for ds, ns in specs:
        X_all = None
        for n in ns:
            if (ds, n) in done:
                continue
            if X_all is None:
                X_all = synthetic.from_spec(ds, n=max(SYN_N))[0] if ds.startswith("synthetic") else data.load(ds)[0]
                perm = np.random.default_rng(0).permutation(len(X_all))
            X = np.ascontiguousarray(X_all[perm[:n]])
            ev = Evaluator(X, n_eval=5000)
            d, i = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(X).kneighbors(X)
            t = time.time()
            m = Grow(n, i, rewarm=rewarm, tol=tol, eval_idx=ev.idx, eval_nb=ev.hd)
            _, info = m.run(X, (i.astype(np.int32), d.astype(np.float32)), seed=seed, n_components=n_components)
            with open(out, "a") as fh:
                fh.write(json.dumps({"dataset": ds, "n": n, "rule": f"grow rewarm={rewarm} tol={tol}",
                                     "n_epochs": 500 if n <= 10000 else 200, "n_components": n_components,
                                     "stopped_at": info["epochs_run"], "returned_epoch": info["returned_epoch"],
                                     "final_recall15": info["final_recall15"], "stage_scores": info["stage_scores"],
                                     "seconds": time.time() - t}) + "\n")
            print(ds, n, info["epochs_run"], info["returned_epoch"], round(info["final_recall15"], 4), flush=True)


class Grow(Monitor):
    """Start short, extend only while it helps (no restarts).

    Stage 1 is exactly a fresh `stages[0]`-epoch UMAP 0.6 Adam run (UMAP's own learning-rate
    schedule for that length). Each further stage of length L continues from the current
    embedding and Adam state with UMAP's schedule for length L scaled by `rewarm`. After each
    stage the monitor recall (2,000 points against the input kNN graph) is compared with the
    previous stage end; if it improved by less than `tol` (relative), training stops. The
    embedding returned is the best stage end by monitor recall."""

    def __init__(self, n_full, knn_idx, stages=(25, 25, 50, 100, 200), rewarm=0.3, tol=0.02, **kw):
        super().__init__(n_full, knn_idx, **kw)
        from umap.layouts import _create_alpha_schedule
        self.stages, self.tol = list(stages), tol
        sched = [_create_alpha_schedule("adam", stages[0], 1.0, True)]
        for L in stages[1:]:
            sched.append(rewarm * _create_alpha_schedule("adam", L, 1.0, True))
        self.alpha = np.concatenate(sched)
        self.ends = list(np.cumsum(stages))
        self.stage_scores, self.best = [], (-1.0, None, 0)

    def kernel(self, *args, **kw):
        if args[0].shape[0] != self.n_full:
            return _ORIG_ADAM(*args, **kw)
        epoch = int(args[_ARG_EPOCH])
        if epoch >= len(self.alpha):
            raise _Stop
        args = list(args)
        args[_ARG_ALPHA] = float(self.alpha[epoch])
        out = _ORIG_ADAM(*args, **kw)
        self.Y = args[0]
        if (epoch + 1) in self.ends:
            Y = np.asarray(args[0])
            self._check(Y, epoch + 1)
            r = self.hist[-1]["recall_mon"]
            if r > self.best[0]:
                self.best = (r, Y.copy(), epoch + 1)
            prev = self.stage_scores[-1] if self.stage_scores else None
            self.stage_scores.append(r)
            if prev is not None and r < prev * (1 + self.tol):
                self.fired = epoch + 1
                raise _Stop
        return out

    def run(self, X, knn_graph, seed=42, n_components=2, **kw):
        Y, info = super().run(X, knn_graph, len(self.alpha) + 1, seed=seed, n_components=n_components, **kw)
        Yb = self.best[1] if self.best[1] is not None else Y
        info.update({"epochs_run": self.hist[-1]["epoch"], "returned_epoch": self.best[2],
                     "final_recall15": _recall(self.in_eval, _knn2d(Yb, self.q_eval, self.k_eval)), "stage_scores": self.stage_scores})
        return Yb, info


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "outdim":       # trajectories at other output dimensions
        m = int(sys.argv[2])
        trajectories(out=f"results/raw/stopping_traj_m{m}.jsonl", n_components=m, specs=OUTDIM_SPECS)
    elif len(sys.argv) > 1 and sys.argv[1] == "grow":        # grow m
        m = int(sys.argv[2])
        if m == 2:
            grow_runs()
        else:
            grow_runs(out=f"results/raw/stopping_grow_m{m}.jsonl", n_components=m, specs=OUTDIM_SPECS)
    elif len(sys.argv) > 1 and sys.argv[1] == "live":       # live m rule:tol [rule:tol ...]
        m = int(sys.argv[2])
        rules = [(a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[3:]]
        if m == 2:
            live(rules)
        else:
            live(rules, out=f"results/raw/stopping_live_m{m}.jsonl", n_components=m, specs=OUTDIM_SPECS)
    else:
        trajectories()
