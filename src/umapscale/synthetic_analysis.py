"""Analysis of the controlled synthetic sweep.

Usage: python -m umapscale.synthetic_analysis results/raw/synthetic_local.jsonl --out results/synthetic_local

1. E95 per (d, realization, n) for the loss target (and recall, for reference).
2. Size exponent per true d: slope of ln E95 on ln n with a separate intercept per
   realization; uncertainty from a bootstrap over realizations.
3. Model comparison for ln E95 with dimension = true d, MLE or TwoNN, by
   leave-one-realization-out (new manifold, seen d) and leave-one-d-out
   (unseen dimension) cross-validation; RMS error in ln E.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
from .evaluate import progress_curve, epochs_needed
from .laws import fit, predict, FORMS

FORMS_SYN = {**FORMS,
             "theory_plus_level: L, L/d, ln d": ["1", "L", "L/d", "I"],
             "constant": ["1"]}


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    feats = {}
    for l in open(path + ".features.jsonl"):
        f = json.loads(l); feats[(f["dataset"], f["n"])] = f
    return rows, feats


def e95_table(rows, feats, target):
    g = defaultdict(list)
    for r in rows:
        if r.get("variant", "default") == "default":
            g[(r["dataset"], r["n"], r["optimizer"])].append(r)
    out = []
    for (ds, n, opt), rs in g.items():
        E, S, _ = progress_curve(rs, target)
        f = feats[(ds, n)]
        out.append({"dataset": ds, "n": n, "optimizer": opt, "d_true": f["id_true"], "realization": f["realization"],
                    "d_mle": f["id_mle"], "d_twonn": f["id_twonn"], "E95": epochs_needed(E, S, 1.0),
                    "S_at_grid": dict(zip(map(int, E), map(float, S)))})
    return out


def exponent_by_d(tab, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for d in sorted({t["d_true"] for t in tab}):
        T = [t for t in tab if t["d_true"] == d and np.isfinite(t["E95"])]
        reals = sorted({t["realization"] for t in T})

        def slope(rs):
            L = np.array([np.log(t["n"]) for t in T if t["realization"] in rs for _ in [0]])
            y = np.array([np.log(t["E95"]) for t in T if t["realization"] in rs])
            R = [t["realization"] for t in T if t["realization"] in rs]
            A = np.column_stack([L] + [np.array([r == k for r in R], float) for k in sorted(set(R))])
            return np.linalg.lstsq(A, y, rcond=None)[0][0]
        k = slope(reals)
        boots = [slope(list(rng.choice(reals, len(reals), replace=True))) for _ in range(n_boot)]
        res[d] = {"exponent": float(k), "ci90": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))],
                  "per_realization": {r: float(np.polyfit([np.log(t["n"]) for t in T if t["realization"] == r],
                                                           [np.log(t["E95"]) for t in T if t["realization"] == r], 1)[0])
                                      for r in reals}}
    return res


def cv(tab, dim_key, group_key, ridge=0.0):
    out = {}
    for name, basis in FORMS_SYN.items():
        errs = []
        for g in sorted({t[group_key] for t in tab}):
            tr = [t for t in tab if t[group_key] != g and np.isfinite(t["E95"])]
            te = [t for t in tab if t[group_key] == g and np.isfinite(t["E95"])]
            law = fit(basis, [t["n"] for t in tr], [t[dim_key] for t in tr], [t["E95"] for t in tr], ridge=ridge)
            pred = predict(law, [t["n"] for t in te], [t[dim_key] for t in te])
            errs += list(np.log(pred) - np.log([t["E95"] for t in te]))
        errs = np.array(errs)
        out[name] = {"rms_log_error": float(np.sqrt(np.mean(errs ** 2))), "median_abs": float(np.median(np.abs(errs)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--targets", default="best,loss,recall")
    ap.add_argument("--out", default="results/synthetic_analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, feats = load(a.results)
    report = {}
    for target in a.targets.split(","):
        tab = [t for t in e95_table(rows, feats, target) if t["optimizer"] == a.optimizer]
        print(f"\n######## target = {target} ########")
        print("E95 (epochs) by true d (rows) and n (columns), mean over realizations [min-max]:")
        ns = sorted({t["n"] for t in tab})
        for d in sorted({t["d_true"] for t in tab}):
            cells = []
            for n in ns:
                v = [t["E95"] for t in tab if t["d_true"] == d and t["n"] == n]
                cells.append(f"{np.mean(v):5.0f} [{min(v):3.0f}-{max(v):3.0f}]")
            print(f"  d={d:2d}: " + " | ".join(cells))
        ex = exponent_by_d(tab)
        print("Size exponent per true d (90% bootstrap CI over realizations):")
        for d, e in ex.items():
            print(f"  d={d:2d}: {e['exponent']:+.3f}  [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}]  per realization "
                  + ", ".join(f"{v:+.2f}" for v in e["per_realization"].values()))
        cvres = {}
        for dim in ["d_true", "d_mle", "d_twonn"]:
            for grp in ["realization", "d_true"]:
                cvres[f"{dim}|leave-one-{grp}-out"] = cv(tab, dim, grp)
        print("Cross-validated RMS error in ln E95 (lower is better):")
        names = list(FORMS_SYN)
        print(f"  {'dimension | split':38s} " + " ".join(f"{n[:18]:>18s}" for n in names))
        for k, v in cvres.items():
            print(f"  {k:38s} " + " ".join(f"{v[n]['rms_log_error']:18.3f}" for n in names))
        report[target] = {"e95": tab, "exponent_by_d": ex, "cv": cvres}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
