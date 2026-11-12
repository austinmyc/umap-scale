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
from .metrics import Evaluator, fuzzy_graph
from .run import embed


def _key(r):
    return (r["dataset"], r["n"], r["optimizer"], r["epochs"], r["seed"], r.get("variant", "default"))


def _done(path):
    if not os.path.exists(path):
        return set()
    return {_key(json.loads(l)) for l in open(path) if l.strip()}


def _pool(name, max_n):
    if name.startswith("synthetic:"):
        return synthetic.from_spec(name, n=max_n)
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
        X_all, y_all = _pool(name, ds.get("pool_n", max(sizes)))   # pool_n: match subsets of other sweeps
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
                if name.startswith("synthetic:"):
                    row.update({"id_true": synthetic.parse(name)["d"], **synthetic.parse(name)})
                if cfg.get("spectral_features", False):   # inputs of the v3 spectral law
                    from .features import spectral
                    row.update(spectral(fuzzy_graph(X, knn(X, cfg.get("n_neighbors", 15), backend=cfg.get("knn_backend", "umap")),
                                                    cfg.get("n_neighbors", 15))))
                with open(feat_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print("features", row, flush=True)
            todo = [(opt, E, s, v) for opt in cfg["optimizers"] for E in cfg["budgets"] + (ds.get("extra_budgets", []))
                    for s in cfg["seeds"] for v in variants
                    if _key({"dataset": name, "n": n, "optimizer": opt, "epochs": E, "seed": s, "variant": v}) not in done]
            gcfg = cfg.get("grow")
            grow_todo = [s for s in cfg["seeds"] if gcfg and _key({"dataset": name, "n": n, "optimizer": "adam",
                                                                    "epochs": "grow", "seed": s, "variant": "grow"}) not in done]
            if not todo and not grow_todo:
                continue
            t = time.perf_counter()
            graph = knn(X, cfg.get("n_neighbors", 15), backend=cfg.get("knn_backend", "umap"))
            knn_s = time.perf_counter() - t
            G = fuzzy_graph(X, graph, cfg.get("n_neighbors", 15)) if cfg.get("umap_loss", False) else None
            ev = Evaluator(X, y, n_eval=cfg.get("n_eval", 5000), graph=G)
            for opt, E, s, v in todo:
                vk = variants[v] or {}
                if opt == "compatibility" and vk:
                    continue
                emb, info = embed(X, graph, E, optimizer=opt, seed=s, parallel=cfg.get("parallel", False),
                                  n_components=cfg.get("n_components", 2),
                                  neg_schedule=vk.get("neg_schedule"), block_size=vk.get("block_size"))
                m = ev(emb, accuracy=cfg.get("knn_accuracy", True))
                row = {"dataset": name, "n": n, "optimizer": opt, "epochs": E, "seed": s, "variant": v,
                       "n_components": cfg.get("n_components", 2), "knn_s": knn_s, **info, **m}
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(name, n, opt, v, E, s, f"{info['fit_s']:.1f}s", f"recall={m['recall15']:.4f}", flush=True)
            for s in grow_todo:   # frozen v3 grow rule (PREDICTIONS_v3.md)
                from .stopping import Grow
                t0 = time.perf_counter()
                g = Grow(n, np.asarray(graph[0]), stages=tuple(gcfg["stages"]), rewarm=gcfg["rho"], tol=gcfg["tau"],
                         eval_idx=ev.idx, eval_nb=ev.hd)
                emb, ginfo = g.run(X, graph, seed=s, n_components=cfg.get("n_components", 2),
                                   parallel=cfg.get("parallel", False))
                m = ev(emb, accuracy=cfg.get("knn_accuracy", True))
                row = {"dataset": name, "n": n, "optimizer": "adam", "epochs": "grow", "seed": s, "variant": "grow",
                       "n_components": cfg.get("n_components", 2), "knn_s": knn_s, "fit_s": time.perf_counter() - t0,
                       "epochs_run": ginfo["epochs_run"], "returned_epoch": ginfo["returned_epoch"],
                       "stage_scores": ginfo["stage_scores"], **m}
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(name, n, "grow", s, f"ran {ginfo['epochs_run']} kept {ginfo['returned_epoch']}",
                      f"recall={m['recall15']:.4f}", flush=True)


if __name__ == "__main__":
    main()
