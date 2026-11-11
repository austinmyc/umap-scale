# umap-scale

How many epochs does UMAP's layout optimization actually need? This repository tests scaling laws for choosing the
epoch budget of UMAP (umap-learn 0.6) from dataset size and intrinsic dimension, on held-out datasets up to 1M+ points.

- [`PREDICTIONS.md`](PREDICTIONS.md): **pre-registered** laws, evaluation protocol and falsifiable predictions,
  frozen before any test data was generated. Do not refit these on test data.
- [`laws/frozen_laws.json`](laws/frozen_laws.json): machine-readable coefficients.
- [`results/dev/`](results/dev/): development measurements the laws were fitted on.

## Background

umap-learn picks 500 layout epochs for n ≤ 10,000 points and 200 otherwise. On four development datasets, the epochs
needed to reach 95% of the 500-epoch neighbor recall with the UMAP 0.6 Adam optimizer were 24–85, so the default
spends 3–15× more than needed. The required budget grows slowly with n (roughly n^0.17 on the development data). The
open questions are whether this holds far beyond the 70k-point development range, and how intrinsic dimension
enters. `PREDICTIONS.md` freezes six competing answers.

## Install

```bash
pip install -e .              # CPU
pip install -e ".[gpu]"       # + FAISS on GPU for the neighbor graph
pip install -e ".[singlecell]"  # + scanpy/anndata to prepare .h5ad files
```

Data goes in `./data` (or set `UMAPSCALE_DATA`). It is git-ignored.

## Run

```bash
# 1. smoke test (a couple of minutes)
python -m umapscale.data download pendigits
python -m umapscale.sweep configs/smoke.yaml
python -m umapscale.evaluate results/raw/smoke.jsonl

# 2. synthetic manifolds with known intrinsic dimension (falsification test)
python -m umapscale.sweep configs/synthetic.yaml

# 3. held-out 1M+ datasets on a GPU server
python -m umapscale.data prepare-h5ad mouse_brain.h5ad data/mouse_brain --label-key <cell_type_column>
python -m umapscale.data prepare-embeddings imagenet_clip.npy data/imagenet_clip
python -m umapscale.sweep configs/large_gpu.yaml

# scoring and figures (any results files)
python -m umapscale.evaluate results/raw/large.jsonl --out results/large_eval.json
python -m umapscale.plots results/raw/large.jsonl --out figs/
```

Sweeps are **resumable**: each finished run is appended to the output `.jsonl`, and re-running the same command skips
completed runs. Per-subset features (n, intrinsic dimension) go to `<output>.features.jsonl`.

### Performance notes

- The neighbor graph is the main cost at 1M+ points. `knn_backend: faiss-gpu` builds it on the GPU (exact search) and
  hands it to umap-learn, whose layout optimizer runs on the CPU.
- umap-learn runs single-threaded when a random seed is fixed. `parallel: true` in a config drops the seed so the layout
  uses all cores; runs then differ slightly between repeats, and seeds only label repeats.
- `extra_budgets: [1000]` adds the 1000-epoch reference check required by `PREDICTIONS.md` for n ≥ 1M.

## Layout of the code

| Module | Purpose |
|---|---|
| `umapscale/data.py` | downloads, loaders (Fashion-MNIST, MNIST, PMLB, prepared `.npy`), preparation of `.h5ad` / embeddings |
| `umapscale/synthetic.py` | manifolds with exactly known intrinsic dimension |
| `umapscale/intrinsic.py` | intrinsic-dimension estimators (MLE, TwoNN) |
| `umapscale/knn.py` | neighbor graph: umap-learn, scikit-learn, FAISS CPU/GPU |
| `umapscale/run.py` | one UMAP run; optional negative-sample schedule and Adam block size |
| `umapscale/metrics.py` | neighbor recall, kNN accuracy, global distance correlation |
| `umapscale/sweep.py` | resumable grid runner driven by `configs/*.yaml` |
| `umapscale/laws.py` | frozen-law prediction and post-hoc fitting |
| `umapscale/evaluate.py` | scores the frozen laws exactly as specified in `PREDICTIONS.md` |
| `umapscale/plots.py` | collapse plot and epochs-vs-n figures |

## Optimizer variants (exploratory, not pre-registered)

`run.py` also implements two changes to the UMAP 0.6 Adam layout, selectable as sweep `variants`:
a front-loaded negative-sample schedule (`neg_schedule: ramp10to1`) and smaller Adam update blocks (`block_size`).
On Fashion-MNIST (70k points, 2 seeds) each raised recall at 50 epochs by about 4–5% at equal cost; on Pendigits
(5.5k, 1 seed) neither helped. Treat them as hypotheses to test, not results.
