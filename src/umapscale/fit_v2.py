"""Fit candidate v2 epoch laws under the best-quality target (deviation D3).

Usage: python -m umapscale.fit_v2 --out laws/candidates_v2.json

Data: real development datasets (results/dev), uniform synthetic manifolds and
clustered synthetic manifolds (results/raw/*.jsonl). Each source gets equal total
weight. Validation: leave-one-dataset-out for real data, leave-one-dimension-out
and leave-one-family-out across sources. Error: ln(E_pred / E95_best).

Besides closed-form laws, an adaptive doubling procedure is scored on the same
grids: run budgets 25, 50, 100, ... and stop at the first budget whose recall is
within `tol` of the next doubling's recall; its cost is the sum of budgets run.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from .evaluate import progress_curve, epochs_needed, recall_at
from .laws import fit, predict

DOUBLINGS = [25, 50, 100, 200, 500]   # 500 stands in for 400 (grid limit)


def table(optimizer="adam"):
    out = []
    ids = json.load(open("results/dev/intrinsic_dim.json"))
    rows = []
    for f in glob.glob("results/dev/scaling_*.json"):
        for r in json.load(open(f)):
            r.setdefault("dataset", "fashion_mnist"); rows.append(r)
    g = defaultdict(list)
    for r in rows:
        if r["optimizer"] == optimizer:
            g[(r["dataset"], r["n"])].append(r)
    for (ds, n), rs in g.items():
        out.append(_case("real", ds, n, ids[f"{ds}|{n}"]["mle"], None, rs))
    for path, fam in [("results/raw/synthetic_local.jsonl", "uniform"), ("results/raw/synthetic_clusters_local.jsonl", "clusters")]:
        if not os.path.exists(path):
            continue
        feats = {}
        for l in open(path + ".features.jsonl"):
            f = json.loads(l); feats[(f["dataset"], f["n"])] = f
        g = defaultdict(list)
        for l in open(path):
            r = json.loads(l)
            if r["optimizer"] == optimizer and r.get("variant", "default") == "default":
                g[(r["dataset"], r["n"])].append(r)
        for (ds, n), rs in g.items():
            f = feats[(ds, n)]
            out.append(_case(fam, ds, n, f["id_mle"], f["id_true"], rs))
    return out


def _case(source, ds, n, d_mle, d_true, rs):
    E, S, best_e = progress_curve(rs, "best")
    rec = {int(r["epochs"]): r["recall15"] for r in rs if r["epochs"] > 0}
    return {"source": source, "dataset": ds, "n": n, "d_mle": d_mle, "d_true": d_true,
            "E95": epochs_needed(E, S, 1.0), "E": E.tolist(), "S": S.tolist(), "recall": rec}


def doubling(case, tol=0.02):
    cost = 0
    for a, b in zip(DOUBLINGS, DOUBLINGS[1:]):
        cost += a
        if a in case["recall"] and b in case["recall"] and case["recall"][b] < case["recall"][a] * (1 + tol):
            return a, cost + b          # the next run was needed to decide
    return DOUBLINGS[-1], cost + DOUBLINGS[-1]


def weights(cases):
    cnt = defaultdict(int)
    for c in cases:
        cnt[c["source"]] += 1
    return np.array([1.0 / cnt[c["source"]] for c in cases])


def wfit(basis, cases, ridge=0.0):
    # weighted least squares by row replication through sqrt-weights
    w = np.sqrt(weights(cases)); w = w / w.mean()
    n = np.array([c["n"] for c in cases], float); d = np.array([c["d_mle"] for c in cases], float)
    y = np.log([c["E95"] for c in cases])
    from .laws import BASES
    L, I = np.log(n), np.log(d)
    A = np.column_stack([BASES[b](L, I) for b in basis])
    beta = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    return {"basis": list(basis), "beta": beta.tolist()}


FORMS_V2 = {"constant": ["1"], "power_law": ["1", "L"], "power_law_plus_dim": ["1", "L", "I"]}


def evaluate(cases, law_fn, split):
    errs, qs, cost = [], [], 0.0
    groups = sorted({c[split] for c in cases})
    for g in groups:
        tr = [c for c in cases if c[split] != g]; te = [c for c in cases if c[split] == g]
        law = law_fn(tr)
        for c in te:
            e = float(predict(law, c["n"], c["d_mle"]))
            errs.append(np.log(e / c["E95"])); cost += e
            qs.append(recall_at(np.array(c["E"]), np.array(c["S"]), e))
    errs, qs = np.array(errs), np.array(qs)
    return {"rms": float(np.sqrt(np.mean(errs ** 2))), "median_q": float(np.median(qs)),
            "frac_q90": float(np.mean(qs >= 0.90)), "mean_epochs": cost / len(qs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="laws/candidates_v2.json")
    a = ap.parse_args()
    cases = [c for c in table() if np.isfinite(c["E95"])]
    print("cases per source:", {s: sum(c["source"] == s for c in cases) for s in ("real", "uniform", "clusters")})
    for c in cases:
        c["d_bin"] = c["d_true"] if c["d_true"] is not None else round(c["d_mle"])
    res = {}
    for name, basis in FORMS_V2.items():
        for q in [None, 0.9]:
            def fn(tr, basis=basis, q=q):
                law = wfit(basis, tr)
                if q is not None:   # shift intercept so the law clears the target in ~90% of training cases
                    r = np.log([c["E95"] for c in tr]) - np.log(predict(law, [c["n"] for c in tr], [c["d_mle"] for c in tr]))
                    law["beta"][0] += float(np.quantile(r, q))
                return law
            tag = name + ("_p90" if q else "")
            res[tag] = {split: evaluate(cases, fn, split) for split in ["dataset", "source", "d_bin"]}
            res[tag]["fit_all"] = fn(cases)
    dbl = [doubling(c) for c in cases]
    res["adaptive_doubling"] = {"median_q": float(np.median([recall_at(np.array(c["E"]), np.array(c["S"]), e)
                                                            for c, (e, _) in zip(cases, dbl)])),
                                "frac_q90": float(np.mean([recall_at(np.array(c["E"]), np.array(c["S"]), e) >= 0.9
                                                           for c, (e, _) in zip(cases, dbl)])),
                                "mean_cost_epochs": float(np.mean([k for _, k in dbl]))}
    print(f"\n{'law':26s} " + " ".join(f"{s:>30s}" for s in ["leave-dataset-out", "leave-source-out", "leave-dimension-out"]))
    print(f"{'':26s} " + " ".join(f"{'RMS  med q  q>=.9  epochs':>30s}" for _ in range(3)))
    for k, v in res.items():
        if k == "adaptive_doubling":
            continue
        print(f"{k:26s} " + " ".join(f"{v[s]['rms']:5.2f} {v[s]['median_q']:6.3f} {v[s]['frac_q90']:6.2f} {v[s]['mean_epochs']:7.0f}"
                                     .rjust(30) for s in ["dataset", "source", "d_bin"]))
    d = res["adaptive_doubling"]
    print(f"\nadaptive doubling (25, 50, 100, ...; stop when doubling adds <2% recall): median q {d['median_q']:.3f}, "
          f"q>=0.90 in {d['frac_q90']:.0%}, mean total epochs run {d['mean_cost_epochs']:.0f}")
    print("default rule mean epochs:", np.mean([500 if c["n"] <= 10000 else 200 for c in cases]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
