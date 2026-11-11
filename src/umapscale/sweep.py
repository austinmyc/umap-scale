"""Resumable sweep over datasets x sizes x optimizers x epoch budgets x seeds.

Usage:  python -m umapscale.sweep configs/<name>.yaml

Each finished run is appended as one JSON line to the config's `output` file;
re-running the same command skips runs already recorded, so long jobs can be
interrupted and restarted. Per-subset features (n, intrinsic dimension) are
written to `<output>.features.jsonl`.
"""
import argparse, json, os, time
import numpy as np, yaml
from . import data, intrinsic, synthetic
from .knn import knn
from .metrics import Evaluator
from .run import embed


def _key(r):
    return (r["dataset"], r["n"], r["optimizer"], r["epochs"], r["seed"], r.get("variant", "default"))


def _done(path):
    if not os.path.exists(path):
        return set()
    return {_key(json.loads(l)) for l in open(path) if l.strip()}


def _pool(name, max_n):
    if name.startswith("synthetic:"):
        parts = name.split(":")[1:]
        d = int(parts[0]); amb = int(parts[1]) if len(parts) > 1 else 50
        return synthetic.manifold(max_n, d, ambient=amb, seed=int(parts[2]) if len(parts) > 2 else 0)
    return data.load(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out = cfg["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = _done(out)
    feat_path = out + ".features.jsonl"
    feats_done = {(json.loads(l)["dataset"], json.loads(l)["n"]) for l in open(feat_path)} if os.path.exists(feat_path) else set()
    variants = cfg.get("variants", {"default": {}})

    for ds in cfg["datasets"]:
        name, sizes = ds["name"], ds["sizes"]
        X_all, y_all = _pool(name, max(sizes))
        perm = np.random.default_rng(cfg.get("subset_seed", 0)).permutation(len(X_all))
        for n in sizes:
            if n > len(X_all):
                print(f"skip {name} n={n} (only {len(X_all)} points)"); continue
            idx = perm[:n]
            X = X_all[idx]; y = None if y_all is None else y_all[idx]
            if (name, n) not in feats_done:
                Xd = X
                if data.is_integer_valued(name):
                    Xd = data.load(name, dequantize=True)[0][idx]
                row = {"dataset": name, "n": n, "ambient_dim": int(X.shape[1]),
                       "id_mle": intrinsic.mle(Xd), "id_twonn": intrinsic.twonn(Xd)}
                with open(feat_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print("features", row, flush=True)
            todo = [(opt, E, s, v) for opt in cfg["optimizers"] for E in cfg["budgets"] + (ds.get("extra_budgets", []))
                    for s in cfg["seeds"] for v in variants
                    if _key({"dataset": name, "n": n, "optimizer": opt, "epochs": E, "seed": s, "variant": v}) not in done]
            if not todo:
                continue
            t = time.perf_counter()
            graph = knn(X, cfg.get("n_neighbors", 15), backend=cfg.get("knn_backend", "umap"))
            knn_s = time.perf_counter() - t
            ev = Evaluator(X, y, n_eval=cfg.get("n_eval", 5000))
            for opt, E, s, v in todo:
                vk = variants[v] or {}
                if opt == "compatibility" and vk:
                    continue
                emb, info = embed(X, graph, E, optimizer=opt, seed=s, parallel=cfg.get("parallel", False),
                                  neg_schedule=vk.get("neg_schedule"), block_size=vk.get("block_size"))
                m = ev(emb, accuracy=cfg.get("knn_accuracy", True))
                row = {"dataset": name, "n": n, "optimizer": opt, "epochs": E, "seed": s, "variant": v,
                       "knn_s": knn_s, **info, **m}
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(name, n, opt, v, E, s, f"{info['fit_s']:.1f}s", f"recall={m['recall15']:.4f}", flush=True)


if __name__ == "__main__":
    main()
