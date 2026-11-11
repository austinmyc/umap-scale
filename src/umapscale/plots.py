"""Figures: collapse plot and epochs-needed vs n with law curves.

Usage:  python -m umapscale.plots results/raw/<run>.jsonl [...] --out figs/
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .evaluate import _curve, reference, epochs_needed
from .laws import load_frozen, predict, FROZEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--laws", default=FROZEN)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--law", default="power_law", help="law used to normalize epochs in the collapse plot")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frozen = load_frozen(a.laws)[a.optimizer]
    rows, feats = [], {}
    for p in a.results:
        rows += [json.loads(l) for l in open(p) if l.strip()]
        if os.path.exists(p + ".features.jsonl"):
            for l in open(p + ".features.jsonl"):
                f = json.loads(l); feats[(f["dataset"], f["n"])] = f
    groups = defaultdict(list)
    for r in rows:
        if r["optimizer"] == a.optimizer and r.get("variant", "default") == "default":
            groups[(r["dataset"], r["n"])].append(r)
    datasets = sorted({k[0] for k in groups})
    cmap = plt.get_cmap("tab10")

    # collapse plot: recall / reference vs epochs / E_law
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for (ds, n), rs in sorted(groups.items()):
        E, R = _curve(rs); ref, _ = reference(E, R)
        e_law = float(predict(frozen[a.law], n, feats[(ds, n)]["id_mle"]))
        c = cmap(datasets.index(ds) % 10)
        ax[0].plot(E, R / ref, "-o", ms=3, color=c, alpha=0.6, label=ds if n == max(k[1] for k in groups if k[0] == ds) else None)
        ax[1].plot(E / e_law, R / ref, "-o", ms=3, color=c, alpha=0.6)
    for x, t in [(ax[0], "Epochs"), (ax[1], f"Epochs / predicted budget ({a.law})")]:
        x.set_xscale("log"); x.set_xlabel(t); x.set_ylabel("Recall / reference recall"); x.axhline(0.95, ls="--", c="gray", lw=0.8)
    ax[1].axvline(1, ls=":", c="k", lw=0.8)
    ax[0].legend(fontsize=8); ax[0].set_title("Raw"); ax[1].set_title("Normalized by the law (curves collapse if it holds)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"collapse_{a.optimizer}_{a.law}.png"), dpi=130); plt.close(fig)

    # measured E95 vs n with law predictions
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in datasets:
        pts = []
        for (d_, n), rs in groups.items():
            if d_ == ds:
                E, R = _curve(rs); ref, _ = reference(E, R)
                pts.append((n, epochs_needed(E, R, ref), feats[(ds, n)]["id_mle"]))
        pts.sort()
        ns = np.array([p[0] for p in pts]); c = cmap(datasets.index(ds) % 10)
        ax.plot(ns, [p[1] for p in pts], "o", color=c, label=f"{ds} (measured)")
        for lname, ls in [("power_law", "-"), ("theory_1_over_d", "--"), ("reduced_quadratic", ":")]:
            ax.plot(ns, [predict(frozen[lname], n, d) for n, _, d in pts], ls, color=c, lw=1, alpha=0.7)
    ax.plot([], [], "k-", label="power_law"); ax.plot([], [], "k--", label="theory_1_over_d"); ax.plot([], [], "k:", label="reduced_quadratic")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("Epochs to 95% of reference recall")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"epochs_vs_n_{a.optimizer}.png"), dpi=130)


if __name__ == "__main__":
    main()
