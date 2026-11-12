"""Downstream check on LLM text embeddings: does the UMAP epoch budget change BERTopic-style topics?

Usage (GPU server, after `python -m umapscale.llm_embed ... --out data/wiki_minilm`):
  python -m umapscale.topics --prefix data/wiki_minilm --n 100000 300000 --knn-backend faiss-gpu \
      --out results/raw/topics_wiki_minilm.jsonl

Pipeline (BERTopic's default design, implemented here so the UMAP step can be controlled):
  embeddings (L2-normalized) -> UMAP to 5-D (n_neighbors=15, min_dist=0.0; Euclidean on unit vectors ranks
  neighbors as cosine does) -> HDBSCAN (min_cluster_size=10, leaf size default, 'eom') -> topics = clusters,
  -1 = outliers -> c-TF-IDF top-10 words per topic (BERTopic's formula: tf_{t,c} * log(1 + A / f_t)).

UMAP variants per subset (PREDICTIONS_v3.md, LLM-embedding addendum):
  default_A, default_B  umap's default epochs (200 for n > 10k), seeds 42 and 7 -> seed-to-seed baseline
  grow                  frozen v3 grow rule, seed 42
  law_v2                v2_power_law budget (12.05 * n^0.133), seed 42

Recorded per variant: epochs, layout seconds, number of topics, outlier fraction, topic diversity
(unique words among all topics' top-10 / (10 * topics)), NPMI coherence of top-10 words (document
co-occurrence on up to 100k passages), ARI and NMI against default_A (on points both assign to a topic,
and over all points with -1 as its own class), and neighbor recall of the 5-D embedding.
"""
import argparse, json, os, time
import numpy as np


def cluster(Y, min_cluster_size=10):
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                               cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(Y)
    except ImportError:
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_method="eom", n_jobs=-1).fit_predict(Y)


def ctfidf_top_words(texts, labels, top=10, max_features=50000):
    from sklearn.feature_extraction.text import CountVectorizer
    topics = sorted(t for t in set(labels) if t != -1)
    if not topics:
        return {}, None, None
    cv = CountVectorizer(stop_words="english", max_features=max_features, min_df=2)
    D = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    import scipy.sparse as sp
    lab = np.asarray(labels)
    rows = [np.asarray(D[lab == t].sum(0)).ravel() for t in topics]
    tf = np.vstack(rows).astype(float)                      # (topics, words)
    A = tf.sum() / len(topics)
    f = tf.sum(0)
    w = tf * np.log(1 + A / np.maximum(f, 1))
    words = {t: vocab[np.argsort(-w[i])[:top]].tolist() for i, t in enumerate(topics)}
    return words, cv, D


def npmi(words, cv, D, max_docs=100_000, seed=0):
    """Mean NPMI over topics of their top words, from binary document co-occurrence."""
    if not words:
        return float("nan")
    idx = np.random.default_rng(seed).choice(D.shape[0], min(max_docs, D.shape[0]), replace=False)
    B = (D[idx] > 0).astype(np.float32).tocsc()
    N = B.shape[0]
    vocab = {w: i for i, w in enumerate(cv.get_feature_names_out())}
    scores = []
    for ws in words.values():
        cols = [vocab[w] for w in ws if w in vocab]
        if len(cols) < 2:
            continue
        S = B[:, cols]
        p = np.asarray(S.sum(0)).ravel() / N
        P = (S.T @ S).toarray() / N
        vals = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if P[i, j] <= 0:
                    vals.append(-1.0); continue
                pmi = np.log(P[i, j] / (p[i] * p[j]))
                vals.append(pmi / -np.log(P[i, j]))
        scores.append(np.mean(vals))
    return float(np.mean(scores)) if scores else float("nan")


def agreement(a, b):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a, b = np.asarray(a), np.asarray(b)
    both = (a != -1) & (b != -1)
    return {"ari_all": float(adjusted_rand_score(a, b)), "nmi_all": float(normalized_mutual_info_score(a, b)),
            "ari_assigned": float(adjusted_rand_score(a[both], b[both])) if both.sum() > 1 else float("nan"),
            "frac_both_assigned": float(both.mean())}


def run_variant(name, X, graph, seed, n, ev, parallel):
    from .stopping import Grow
    from .run import embed
    t = time.perf_counter()
    if name == "grow":
        g = Grow(n, np.asarray(graph[0]), stages=(25, 25, 50, 100, 200), rewarm=0.3, tol=0.03,
                 eval_idx=ev.idx, eval_nb=ev.hd)
        Y, info = g.run(X, graph, seed=seed, n_components=5, min_dist=0.0, parallel=parallel)
        epochs = info["epochs_run"]
    else:
        epochs = (500 if n <= 10000 else 200) if name.startswith("default") else int(round(12.05 * n ** 0.133))
        Y, _ = embed(X, graph, epochs, seed=seed, n_components=5, min_dist=0.0, parallel=parallel)
    return Y, epochs, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="output prefix of umapscale.llm_embed")
    ap.add_argument("--n", type=int, nargs="+", default=[100000, 300000])
    ap.add_argument("--knn-backend", default="umap")
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--parallel", action="store_true", help="multi-threaded layout (runs not reproducible)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from .knn import knn
    from .metrics import Evaluator
    Xall = np.load(f"{a.prefix}_X.npy", mmap_mode="r")
    texts_all = open(f"{a.prefix}_texts.txt").read().split("\n")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["n"], json.loads(l)["variant"]) for l in open(a.out)}
    for n in a.n:
        X = np.ascontiguousarray(Xall[:n], dtype=np.float32)        # llm_embed already shuffled: prefix = random subset
        texts = texts_all[:n]
        graph = knn(X, 15, backend=a.knn_backend)
        ev = Evaluator(X, n_eval=5000)
        labels = {}
        for name, seed in [("default_A", 42), ("default_B", 7), ("grow", 42), ("law_v2", 42)]:
            if (n, name) in done and name != "default_A":
                continue
            Y, epochs, secs = run_variant(name, X, graph, seed, n, ev, a.parallel)
            lab = cluster(Y, a.min_cluster_size)
            labels[name] = lab
            words, cv, D = ctfidf_top_words(texts, lab)
            K = len(words)
            row = {"prefix": a.prefix, "n": n, "variant": name, "seed": seed, "epochs": epochs, "layout_s": secs,
                   "recall15": ev.recall(Y), "n_topics": K, "outlier_frac": float(np.mean(lab == -1)),
                   "topic_diversity": float(len({w for ws in words.values() for w in ws}) / (10 * K)) if K else float("nan"),
                   "npmi": npmi(words, cv, D) if K else float("nan")}
            if name != "default_A":
                row.update({f"vs_default_A_{k}": v for k, v in agreement(labels["default_A"], lab).items()})
            if (n, name) not in done:
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "prefix"}, flush=True)


if __name__ == "__main__":
    main()
