"""Build an LLM text-embedding dataset for the large-scale test (run on the GPU server).

Usage:
  python -m umapscale.llm_embed --hf-dataset wikimedia/wikipedia --hf-config 20231101.en --column text \
      --n 3000000 --model sentence-transformers/all-MiniLM-L6-v2 --out data/wiki_minilm
  python -m umapscale.llm_embed --texts my_passages.txt --model Qwen/Qwen3-Embedding-0.6B --out data/my_qwen

Writes:
  <out>_X.npy      float32 (n, D) embeddings, L2-normalized, so Euclidean kNN ranks neighbors exactly as
                   cosine similarity does (BERTopic uses metric="cosine")
  <out>_texts.txt  one passage per line (newlines inside a passage replaced by spaces), same order
  <out>_meta.json  model, source, chunking, n, D, seed

Passages: documents are split into chunks of `--chunk-words` whitespace-separated words (default 128);
chunks with fewer than `--min-words` words are dropped. Documents are read in source order, and
the first `--n` chunks are kept, then shuffled with `--seed` so any prefix is a random subset.
`--model` is any sentence-transformers model, including LLM-based embedders (e.g. Qwen3-Embedding,
gte-Qwen, e5-mistral). `--model hashing` is a dependency-free stand-in for smoke tests only.
"""
import argparse, json, os
import numpy as np


def iter_documents(a):
    if a.texts:
        with open(a.texts) as f:
            for line in f:
                if line.strip():
                    yield line
    else:
        from datasets import load_dataset
        ds = load_dataset(a.hf_dataset, a.hf_config, split=a.split, streaming=True)
        for row in ds:
            yield row[a.column]


def chunks(a):
    out = []
    for doc in iter_documents(a):
        w = doc.split()
        for i in range(0, len(w), a.chunk_words):
            c = w[i:i + a.chunk_words]
            if len(c) >= a.min_words:
                out.append(" ".join(c))
                if len(out) >= a.n:
                    return out
    return out


def encode(texts, a):
    if a.model == "hashing":                       # smoke tests only: TF-IDF hashing + random projection
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        H = TfidfTransformer().fit_transform(HashingVectorizer(n_features=2 ** 16, alternate_sign=False).transform(texts))
        P = np.random.default_rng(0).normal(size=(2 ** 16, 256)).astype(np.float32)
        X = np.asarray(H @ P, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(a.model, device=a.device, trust_remote_code=a.trust_remote_code)
        X = m.encode(texts, batch_size=a.batch_size, show_progress_bar=True, convert_to_numpy=True,
                     normalize_embeddings=False).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--texts", help="text file, one document per line")
    src.add_argument("--hf-dataset", help="Hugging Face dataset id (streamed)")
    ap.add_argument("--hf-config"); ap.add_argument("--split", default="train"); ap.add_argument("--column", default="text")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--chunk-words", type=int, default=128); ap.add_argument("--min-words", type=int, default=32)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    texts = chunks(a)
    order = np.random.default_rng(a.seed).permutation(len(texts))
    texts = [texts[i] for i in order]
    X = encode(texts, a)
    np.save(f"{a.out}_X.npy", X)
    with open(f"{a.out}_texts.txt", "w") as f:
        for t in texts:
            f.write(t.replace("\n", " ").replace("\r", " ") + "\n")
    meta = {"model": a.model, "source": a.texts or f"{a.hf_dataset}:{a.hf_config}:{a.split}:{a.column}",
            "chunk_words": a.chunk_words, "min_words": a.min_words, "n": len(texts), "dim": int(X.shape[1]),
            "seed": a.seed, "normalized": "L2"}
    json.dump(meta, open(f"{a.out}_meta.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
