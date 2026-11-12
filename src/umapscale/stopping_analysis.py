"""Offline screening of in-training stopping rules on recorded trajectories.

Usage: python -m umapscale.stopping_analysis [--traj results/raw/stopping_traj.jsonl]

Each trajectory is one run at umap's default length (500 epochs if n <= 10k, else 200),
monitored every 5 epochs. A rule is replayed on the history; at the first check where it
fires, the run would stop. Score q = recall15 at the stop / R*, where R* is the best
mean recall15 over the separate fixed-budget runs of the sweeps (the E95_best
reference). Cost = stop epoch / default epochs. A rule that never fires costs the
full run.
"""
import argparse, glob, json
from collections import defaultdict
import numpy as np
from .evaluate import _curve
from .stopping import RULES

GRID = {"recall_plateau": [0.002, 0.005, 0.01, 0.02],
        "nn_stability": [0.2, 0.3, 0.4, 0.5, 0.6],
        "cov_residual": [0.002, 0.005, 0.01, 0.02],
        "id_residual": [0.005, 0.01, 0.02, 0.05],
        "displacement": [0.005, 0.01, 0.02, 0.03]}


def best_recall():
    rows = defaultdict(list)
    for f in glob.glob("results/dev/scaling_*.json"):
        for r in json.load(open(f)):
            if r["optimizer"] == "adam":
                rows[(r.get("dataset", "fashion_mnist"), r["n"], 2)].append(r)
    for p in ["results/raw/synthetic_local.jsonl", "results/raw/synthetic_clusters_local.jsonl"] + \
            sorted(p for p in glob.glob("results/raw/output_dim_m*.jsonl") if "features" not in p):
        for l in open(p):
            r = json.loads(l)
            if r["optimizer"] == "adam" and r.get("variant", "default") == "default":
                rows[(r["dataset"], r["n"], r.get("n_components", 2))].append(r)
    return {k: float(_curve(v)[1].max()) for k, v in rows.items()}


def replay(hist, rule):
    for i in range(len(hist)):
        if rule(hist[: i + 1]):
            return i
    return len(hist) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", default=["results/raw/stopping_traj.jsonl"])
    ap.add_argument("--out", default="results/stopping/screen.json")
    a = ap.parse_args()
    Rstar = best_recall()
    T = [json.loads(l) for p in a.traj for l in open(p)]
    key = lambda t: (t["dataset"], t["n"], t.get("n_components", 2))
    for t in T:      # reference: best of the fixed-budget grid, or of the trajectory itself if higher
        Rstar[key(t)] = max(Rstar.get(key(t), 0.0), max(h["recall15"] for h in t["hist"]))
    print(f"{len(T)} trajectories; output dimensions", sorted({key(t)[2] for t in T}))
    src = lambda t: "real" if not t["dataset"].startswith("synthetic") else ("clusters" if "clusters" in t["dataset"] else "uniform")
    out = {}

    def report(name, stops):
        q = np.array([t["hist"][i]["recall15"] / Rstar[key(t)] for t, i in zip(T, stops)])
        cost = np.array([t["hist"][i]["epoch"] / t["n_epochs"] for t, i in zip(T, stops)])
        ep = np.array([t["hist"][i]["epoch"] for t, i in zip(T, stops)])
        per = {s: float(np.mean(q[[src(t) == s for t in T]] >= 0.9)) for s in ["real", "uniform", "clusters"]
               if any(src(t) == s for t in T)}
        for m in sorted({key(t)[2] for t in T}):
            per[f"m{m}"] = float(np.mean(q[[key(t)[2] == m for t in T]] >= 0.9))
        out[name] = {"median_q": float(np.median(q)), "frac_q90": float(np.mean(q >= 0.9)), "min_q": float(q.min()),
                     "cost_vs_default": float(cost.mean()), "median_epochs": float(np.median(ep)), "frac_q90_by_source": per}
        print(f"{name:28s} {np.median(q):7.3f} {np.mean(q >= 0.9):6.2f} {q.min():6.3f} {cost.mean():6.0%} {np.median(ep):6.0f}   "
              + " ".join(f"{s[:4]} {v:.2f}" for s, v in per.items()))

    print(f"{'rule':28s} {'med q':>7s} {'q>=.9':>6s} {'min q':>6s} {'cost':>6s} {'epochs':>6s}   q>=.9 by source")
    report("no stop (default length)", [len(t["hist"]) - 1 for t in T])
    for e in [40, 60, 100]:
        report(f"fixed stop at {e}", [min(e // 5 - 1, len(t["hist"]) - 1) for t in T])
    for rname, tols in GRID.items():
        for tol in tols:
            report(f"{rname} {tol}", [replay(t["hist"], RULES[rname](tol)) for t in T])
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()


def live_report(paths, traj_paths=()):
    """Score live anneal-mode runs: q = final recall / R*, cost = epochs run / default epochs."""
    Rstar = best_recall()
    for p in traj_paths:
        for l in open(p):
            t = json.loads(l); k = (t["dataset"], t["n"], t.get("n_components", 2))
            Rstar[k] = max(Rstar.get(k, 0.0), max(h["recall15"] for h in t["hist"]))
    rows = [json.loads(l) for p in paths for l in open(p)]
    src = lambda r: "real" if not r["dataset"].startswith("synthetic") else ("clusters" if "clusters" in r["dataset"] else "uniform")
    g = defaultdict(list)
    for r in rows:
        k = (r["dataset"], r["n"], r.get("n_components", 2))
        if k in Rstar:
            g[r["rule"]].append((r, r["final_recall15"] / Rstar[k], r["stopped_at"] / r["n_epochs"]))
    print(f"{'rule (anneal)':26s} {'cases':>5s} {'med q':>6s} {'q>=.9':>6s} {'min q':>6s} {'cost':>6s} {'med ep':>6s}   q>=.9 by source")
    out = {}
    for rule, v in sorted(g.items()):
        q = np.array([x[1] for x in v]); c = np.array([x[2] for x in v]); ep = [x[0]["stopped_at"] for x in v]
        per = {s: float(np.mean([x[1] >= 0.9 for x in v if src(x[0]) == s])) for s in ["real", "uniform", "clusters"]
               if any(src(x[0]) == s for x in v)}
        out[rule] = {"cases": len(v), "median_q": float(np.median(q)), "frac_q90": float(np.mean(q >= 0.9)),
                     "min_q": float(q.min()), "cost_vs_default": float(c.mean()), "median_epochs": float(np.median(ep)), "by_source": per}
        print(f"{rule:26s} {len(v):5d} {np.median(q):6.3f} {np.mean(q >= 0.9):6.2f} {q.min():6.3f} {c.mean():6.0%} {np.median(ep):6.0f}   "
              + " ".join(f"{s[:4]} {x:.2f}" for s, x in per.items()))
    return out
