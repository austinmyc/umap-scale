"""Figure: how epoch need relates to n, intrinsic dimension and the choice of target.

Usage: python -m umapscale.plot_relations --out figs/relations.png
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .fit_v2 import table
from .evaluate import progress_curve, epochs_needed
from .laws import load_frozen, predict

BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # ordinal ramp, d = 2..24
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
DS = [2, 4, 8, 12, 16, 24]


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def end_labels(ax, items, log=True, gap=0.075):
    """items: (x, y, text). Labels at line ends, nudged apart vertically so they never overlap."""
    items = sorted(items, key=lambda t: t[1])
    f = (lambda v: np.log10(v)) if log else (lambda v: v)
    g = (lambda v: 10 ** v) if log else (lambda v: v)
    ys = [f(t[1]) for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (x, _, txt), y in zip(items, ys):
        ax.annotate(txt, (x, g(y)), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2,
                    va="center", annotation_clip=False)


def plain_ticks(ax, axis="both"):
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, axis + "axis")]):
        sf = ScalarFormatter(); sf.set_scientific(False)
        a.set_major_formatter(sf); a.set_minor_formatter(NullFormatter())


def raw_rows(path):
    g = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r["optimizer"] == "adam" and r.get("variant", "default") == "default":
            g[(r["dataset"], r["n"])].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/relations.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cases = [c for c in table() if np.isfinite(c["E95"])]
    v2 = load_frozen("laws/frozen_laws_v2.json")["adam"]
    nn = np.geomspace(1300, 70000, 50)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.patch.set_facecolor("#fcfcfb")
    # row 1: E95_best vs n
    for ax, fam in zip(axes[0, :2], ["uniform", "clusters"]):
        labs = []
        for j, d in enumerate(DS):
            pts = defaultdict(list)
            for c in cases:
                if c["source"] == fam and c["d_true"] == d:
                    pts[c["n"]].append(c["E95"])
            ns = sorted(pts)
            m = [np.mean(pts[n]) for n in ns]
            lo = [m[i] - min(pts[n]) for i, n in enumerate(ns)]
            hi = [max(pts[n]) - m[i] for i, n in enumerate(ns)]
            ax.errorbar(ns, m, yerr=[lo, hi], color=BLUE[j], lw=2, marker="o", ms=5, capsize=0, elinewidth=1)
            labs.append((ns[-1], m[-1], f"d={d}"))
        end_labels(ax, labs)
        ax.plot(nn, predict(v2["v2_power_law"], nn, 10), ls="--", color=INK2, lw=1)
        ax.plot(nn, predict(v2["v2_constant"], nn, 10), ls=":", color=INK2, lw=1)
        style(ax, f"Quality target, {fam} manifolds (true d)")
    ax = axes[0, 2]
    rlabs = []
    for j, ds in enumerate(["fashion_mnist", "mnist", "letter", "pendigits"]):
        pts = sorted((c["n"], c["E95"], c["d_mle"]) for c in cases if c["dataset"] == ds)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=CAT[j], lw=2, ms=5)
        rlabs.append((pts[-1][0], pts[-1][1], f"{ds}\n(d̂ {pts[0][2]:.1f}–{pts[-1][2]:.1f})"))
    end_labels(ax, rlabs, gap=0.11)
    ax.plot(nn, predict(v2["v2_power_law"], nn, 10), ls="--", color=INK2, lw=1, label="v2 power law  12.05·n^0.133")
    ax.plot(nn, predict(v2["v2_constant"], nn, 10), ls=":", color=INK2, lw=1, label="v2 constant  41")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax, "Quality target, real data (MLE d̂)")
    for ax in axes[0]:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(8, 400); ax.set_xlim(1200, 160000)
        ax.set_xlabel("n (points)", fontsize=9, color=INK2)
        plain_ticks(ax, "y")
    axes[0, 0].set_ylabel("Epochs to 95% of best recall", fontsize=9, color=INK2)

    # row 2a: E95_best vs d at the largest synthetic n; real at every n vs MLE d
    ax = axes[1, 0]
    for k, (fam, mk) in enumerate([("uniform", "o"), ("clusters", "s")]):
        for n, col in [(5000, BLUE[1]), (40000, BLUE[4])]:
            v = defaultdict(list)
            for c in cases:
                if c["source"] == fam and c["n"] == n:
                    v[c["d_true"]].append(c["E95"])
            ax.plot(DS, [np.mean(v[d]) for d in DS], "-" + mk, color=col, lw=1.6, ms=5,
                    label=f"{fam}, n={n // 1000}k", alpha=1 if fam == "uniform" else 0.75,
                    ls="-" if fam == "uniform" else "--")
    r = [c for c in cases if c["source"] == "real"]
    ax.scatter([c["d_mle"] for c in r], [c["E95"] for c in r], s=22, color=CAT[1], zorder=3, label="real (MLE d̂), all n")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(DS); ax.set_xticklabels(DS)
    plain_ticks(ax, "y"); ax.set_yticks([20, 30, 50, 100, 200])
    ax.set_xlabel("intrinsic dimension d", fontsize=9, color=INK2)
    ax.set_ylabel("Epochs to 95% of best recall", fontsize=9, color=INK2)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "Quality target vs dimension: flat above d≈4")

    # row 2b: loss target vs n (clusters)
    ax = axes[1, 1]
    g = raw_rows("results/raw/synthetic_clusters_local.jsonl")
    labs = []
    for j, d in enumerate(DS):
        pts = defaultdict(list)
        for (ds, n), rs in g.items():
            if ds.startswith(f"synthetic:{d}:"):
                E, S, _ = progress_curve(rs, "loss")
                pts[n].append(epochs_needed(E, S, 1.0))
        ns = sorted(pts)
        m = [np.nanmean(pts[n]) for n in ns]
        ax.plot(ns, m, "-o", color=BLUE[j], lw=2, ms=5)
        labs.append((ns[-1], m[-1], f"d={d}"))
    end_labels(ax, labs)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(8, 900); ax.set_xlim(2000, 80000)
    plain_ticks(ax)
    ax.set_xlabel("n (points)", fontsize=9, color=INK2)
    ax.set_ylabel("Epochs to 95% of loss reduction", fontsize=9, color=INK2)
    style(ax, "Loss target, clustered manifolds: d matters here")

    # row 2c: recall / best recall vs epochs, clusters n = 40k, realization 0
    ax = axes[1, 2]
    gu = raw_rows("results/raw/synthetic_local.jsonl")
    labs = []
    for j, d in enumerate(DS):
        rs = gu[(f"synthetic:{d}:50:0", 10000)]
        E, S, _ = progress_curve(rs, "best")
        ax.plot(E, S, "-o", color=BLUE[j], lw=2, ms=4)
        labs.append((E[-1], S[-1], f"d={d}"))
    end_labels(ax, labs, log=False, gap=0.045)
    ax.axhline(0.95, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlim(4, 900); ax.set_ylim(0, 1.03)
    ax.set_xlabel("epoch budget (separate runs)", fontsize=9, color=INK2)
    ax.set_ylabel("recall / best recall", fontsize=9, color=INK2)
    style(ax, "Recall vs budget (uniform, n=10k): peaks by ~35–50, then falls at d ≥ 12")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
