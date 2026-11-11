# umap-scale

How many epochs does UMAP's layout optimization actually need? Experiments and scaling laws for choosing the epoch
budget of UMAP (umap-learn 0.6) from dataset size and intrinsic dimension, tested on held-out datasets up to 1M+ points.

- [`PREDICTIONS.md`](PREDICTIONS.md): pre-registered laws and evaluation protocol (frozen before testing).
- [`laws/frozen_laws.json`](laws/frozen_laws.json): machine-readable coefficients.
- [`results/dev/`](results/dev/): development measurements the laws were fitted on.

Code for running sweeps (CPU and GPU) follows in the next commit.
