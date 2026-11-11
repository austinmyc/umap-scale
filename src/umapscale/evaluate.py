"""Score the frozen laws on sweep results, following PREDICTIONS.md.

Usage:  python -m umapscale.evaluate results/raw/<run>.jsonl [more.jsonl ...] [--laws laws/frozen_laws.json]

For each test case (dataset, n, optimizer): recall is averaged over seeds per
budget; the reference is the 500-epoch recall, replaced by the 1000-epoch
recall when that is more than 2% higher; E95 is log-interpolated; each law's
predicted budget E_law is scored by q = recall(E_law) / reference.
Pass: median q >= 0.94 and q >= 0.90 in at least 90% of cases.
"""
import argparse, json
from collections import defaultdict
import numpy as np
from .laws import load_frozen, predict, FROZEN


def _curve(rows):
    by_e = defaultdict(list)
    for r in rows:
        by_e[r["epochs"]].append(r["recall15"])
    E = np.array(sorted(by_e), float)
    return E, np.array([np.mean(by_e[e]) for e in E])


def reference(E, R):
    ref = R[E == 500][0] if 500 in E else R[-1]
    if 1000 in E and R[E == 1000][0] > 1.02 * ref:
        return float(R[E == 1000][0]), 1000
    return float(ref), 500


def recall_at(E, R, e):
    return float(np.interp(np.log(np.clip(e, E[0], E[-1])), np.log(E), R))


def epochs_needed(E, R, ref, frac=0.95):
    t = frac * ref
    for i in range(len(E)):
        if R[i] >= t:
            if i == 0:
                return float(E[0])
            u = (t - R[i - 1]) / (R[i] - R[i - 1])
            return float(np.exp(np.log(E[i - 1]) + u * (np.log(E[i]) - np.log(E[i - 1]))))
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--laws", default=FROZEN)
    ap.add_argument("--out")
    a = ap.parse_args()
    frozen = load_frozen(a.laws)
    rows, feats = [], {}
    for p in a.results:
        rows += [json.loads(l) for l in open(p) if l.strip()]
        try:
            for l in open(p + ".features.jsonl"):
                f = json.loads(l); feats[(f["dataset"], f["n"])] = f
        except FileNotFoundError:
            pass
    groups = defaultdict(list)
    for r in rows:
        if r.get("variant", "default") == "default":
            groups[(r["dataset"], r["n"], r["optimizer"])].append(r)

    report = {}
    for opt in sorted({k[2] for k in groups}):
        cases = []
        for (ds, n, o), rs in sorted(groups.items()):
            if o != opt:
                continue
            E, R = _curve(rs)
            ref, ref_e = reference(E, R)
            d = feats[(ds, n)]["id_mle"]
            case = {"dataset": ds, "n": n, "d": d, "reference_epochs": ref_e,
                    "E95_measured": epochs_needed(E, R, ref), "laws": {}}
            for name, law in frozen.get(opt, {}).items():
                e = float(predict(law, n, d))
                case["laws"][name] = {"epochs": e, "q": recall_at(E, R, e) / ref}
            cases.append(case)
        print(f"\n=== {opt}: {len(cases)} test cases ===")
        print(f"{'law':28s} {'median q':>9s} {'q>=0.90':>8s} {'pass':>5s} {'epochs vs default':>18s}")
        summary = {}
        for name in frozen.get(opt, {}):
            q = np.array([c["laws"][name]["q"] for c in cases])
            cost = sum(c["laws"][name]["epochs"] for c in cases)
            dcost = sum(c["laws"]["default_rule"]["epochs"] for c in cases)
            ok = bool(np.median(q) >= 0.94 and np.mean(q >= 0.90) >= 0.90)
            summary[name] = {"median_q": float(np.median(q)), "frac_q_ge_0.90": float(np.mean(q >= 0.90)),
                             "pass": ok, "epochs_vs_default": cost / dcost}
            print(f"{name:28s} {np.median(q):9.3f} {np.mean(q >= 0.90):8.2f} {('yes' if ok else 'no'):>5s} {cost / dcost:17.0%}")
        for c in cases:
            print(f"  {c['dataset']:24s} n={c['n']:>8d} d={c['d']:5.1f} measured E95={c['E95_measured']:6.0f} | " +
                  " ".join(f"{k}={v['epochs']:.0f}" for k, v in c["laws"].items()))
        report[opt] = {"summary": summary, "cases": cases}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
