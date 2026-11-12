# Running the held-out tests on a GPU server

```bash
git clone https://github.com/austinmyc/umap-scale.git && cd umap-scale
pip install -e ".[gpu,singlecell,llm]"      # faiss-gpu-cu12, scanpy, sentence-transformers, datasets, hdbscan
```

## 1. Data (once)

```bash
# LLM text embeddings (Wikipedia passages, BERTopic's default embedder); ~3M x 384 float32 = 4.6 GB
python -m umapscale.llm_embed --hf-dataset wikimedia/wikipedia --hf-config 20231101.en --column text \
    --n 3000000 --model sentence-transformers/all-MiniLM-L6-v2 --out data/wiki_minilm
# Optional second embedder (LLM-based); add it as another dataset line in the llm configs
python -m umapscale.llm_embed --hf-dataset wikimedia/wikipedia --hf-config 20231101.en --column text \
    --n 1000000 --model Qwen/Qwen3-Embedding-0.6B --trust-remote-code --out data/wiki_qwen3

# Single-cell and image embeddings for the original large test (see configs/large_gpu.yaml)
python -m umapscale.data prepare-h5ad <file.h5ad> data/mouse_brain --label-key <cell_type_column>
python -m umapscale.data prepare-embeddings <clip_features.npy> data/imagenet_clip
```

## 2. Sweeps (resumable; rerun the same command after an interruption)

```bash
python -m umapscale.sweep configs/large_gpu_llm_m2.yaml
python -m umapscale.sweep configs/large_gpu_llm_m5.yaml
python -m umapscale.topics --prefix data/wiki_minilm --n 100000 300000 --knn-backend faiss-gpu \
    --parallel --out results/raw/topics_wiki_minilm.jsonl
python -m umapscale.sweep configs/large_gpu.yaml
python -m umapscale.sweep configs/large_gpu_m10.yaml
```

The 3M-point grid is the most expensive part (12 budgets + 1000 + grow, 3 seeds). If time is short, drop 3M from
the llm configs first, then the third seed. Do not change any other setting: the predictions are frozen.

## 3. Scoring

```bash
for laws in laws/frozen_laws.json laws/frozen_laws_v2.json laws/frozen_v3.json; do
  python -m umapscale.evaluate results/raw/llm_wiki_minilm_m2.jsonl --target best --laws $laws
done
python -m umapscale.evaluate results/raw/llm_wiki_minilm_m5.jsonl --target best --laws laws/frozen_v3.json
```

Commit `results/raw/*.jsonl` from the server (they are small) or copy them back, so the analysis can be run here.
