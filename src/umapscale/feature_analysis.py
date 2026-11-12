"""Do data features beyond n predict the epoch need (E95_best)?

Usage: python -m umapscale.feature_analysis --out results/features/report.json

1. Correlation (Spearman) of each feature with the residual of ln E95_best after the
   v2 power law in n, pooled and per source.
2. Cross-validated comparison: power law in n alone vs power law + one feature, and
   + all features (ridge), weighted so each source counts equally. Splits: leave one
   dataset out, leave one source out, leave one true dimension out.
3. A probe feature that costs a short run: ln(recall at 25 epochs / recall at 10
   epochs), i.e. how much is still improving early (35 epochs of probing).
"""
import argparse, json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from .fit_v2 import table, weights
from .evaluate import recall_at
from .laws import load_frozen, predict

STATIC = ["log_pr", "top10_var", "eig_decay", "cv_knn_dist", "log_rel_nn", "id_cv", "hubness",
          "mutual_frac", "log_gap2", "log_gap11", "cluster_gap", "log_id"]
PROBE = ["probe_25_10"]


def build(feat_path="results/features/features.jsonl"):
    feats = {}
    for l in open(feat_path):
        f = json.loads(l); feats[(f["dataset"], f["n"])] = f
    cases = []
    for c in table():
        if not np.isfinite(c["E95"]) or (c["dataset"], c["n"]) not in feats:
            continue
        f = dict(feats[(c["dataset"], c["n"])])
        f["log_id"] = float(np.log(c["d_mle"]))
        r = c["recall"]
        f["probe_25_10"] = float(np.log(r[25] / r[10])) if 25 in r and 10 in r else np.nan
        c["f"] = f
        c["d_bin"] = c["d_true"] if c["d_true"] is not None else round(c["d_mle"])
        cases.append(c)
    return cases


def design(cases, cols):
    L = np.log([c["n"] for c in cases])
    return np.column_stack([np.ones(len(cases)), L] + [[c["f"][k] for c in cases] for k in cols])


def wfit(cases, cols, ridge=0.0):
    A = design(cases, cols); y = np.log([c["E95"] for c in cases])
    w = weights(cases); w = w / w.mean()
    mu, sd = A[:, 1:].mean(0), A[:, 1:].std(0) + 1e-12
    Z = np.column_stack([np.ones(len(y)), (A[:, 1:] - mu) / sd])
    P = np.diag([0.0] + [ridge] * (Z.shape[1] - 1))
    beta = np.linalg.solve(Z.T @ (Z * w[:, None]) + P, Z.T @ (y * w))
    return lambda cs: np.column_stack([np.ones(len(cs)), (design(cs, cols)[:, 1:] - mu) / sd]) @ beta


def cv(cases, cols, split, ridge=0.0):
    errs, qs = [], []
    for g in sorted({c[split] for c in cases}, key=str):
        tr = [c for c in cases if c[split] != g]; te = [c for c in cases if c[split] == g]
        f = wfit(tr, cols, ridge)
        for c, lp in zip(te, f(te)):
            errs.append(lp - np.log(c["E95"]))
            qs.append(recall_at(np.array(c["E"]), np.array(c["S"]), float(np.exp(lp))))
    errs, qs = np.array(errs), np.array(qs)
    return {"rms": float(np.sqrt(np.mean(errs ** 2))), "median_q": float(np.median(qs)), "frac_q90": float(np.mean(qs >= 0.9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/features/report.json")
    a = ap.parse_args()
    cases = build()
    print("cases:", {s: sum(c["source"] == s for c in cases) for s in ["real", "uniform", "clusters"]})
    law = load_frozen("laws/frozen_laws_v2.json")["adam"]["v2_power_law"]
    res = np.array([np.log(c["E95"]) - np.log(predict(law, c["n"], 1)) for c in cases])
    rep = {"spearman_with_residual": {}, "cv": {}}
    print("\nSpearman correlation of feature with residual ln E95 - ln E_power_law  (pooled | real | uniform | clusters)")
    for k in STATIC + PROBE:
        row = {}
        for s in [None, "real", "uniform", "clusters"]:
            idx = [i for i, c in enumerate(cases) if (s is None or c["source"] == s) and np.isfinite(c["f"][k])]
            row[s or "pooled"] = float(spearmanr([cases[i]["f"][k] for i in idx], res[idx])[0]) if len(idx) > 3 else None
        rep["spearman_with_residual"][k] = row
        print(f"  {k:14s} " + " ".join(f"{v:+6.2f}" if v is not None else "   n/a" for v in row.values()))
    models = {"n only": ([], 0.0)}
    for k in STATIC + PROBE:
        models[f"n + {k}"] = ([k], 0.0)
    models["n + all static (ridge 3)"] = (STATIC, 3.0)
    models["n + all static + probe (ridge 3)"] = (STATIC + PROBE, 3.0)
    print(f"\n{'model':36s} " + " ".join(f"{s:>26s}" for s in ["leave-dataset-out", "leave-source-out", "leave-dimension-out"]))
    print(f"{'':36s} " + " ".join(f"{'RMS  med q  q>=.9':>26s}" for _ in range(3)))
    for name, (cols, ridge) in models.items():
        use = [c for c in cases if all(np.isfinite(c["f"][k]) for k in cols)]
        r = {sp: cv(use, cols, sp, ridge) for sp in ["dataset", "source", "d_bin"]}
        rep["cv"][name] = r
        print(f"{name:36s} " + " ".join(f"{v['rms']:6.3f} {v['median_q']:6.3f} {v['frac_q90']:6.2f}".rjust(26) for v in r.values()))
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
