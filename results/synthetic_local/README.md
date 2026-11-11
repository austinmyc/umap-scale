# Controlled synthetic sweep (local, CPU)

Uniform manifolds with known intrinsic dimension d in {2, 4, 8, 12, 16, 24}, n in {2.5k ... 40k}, 3 realizations,
UMAP 0.6 Adam, 13 budgets. Produced by `configs/synthetic_local.yaml` and
`python -m umapscale.synthetic_analysis results/raw/synthetic_local.jsonl --out results/synthetic_local`.

- `report_ref500.json`: analysis with the 500-epoch reference for every case, computed before the 1000-epoch
  reference runs (`configs/synthetic_local_reference.yaml`) finished. At d = 2 and 4 the loss still falls between
  500 and 1000 epochs (+6.3% and +3.1% more reduction at n = 10k), so by the rule in `PREDICTIONS.md` those cases will
  be re-scored against 1000 epochs. Treat the low-d numbers here as provisional.

Headline findings at this stage: including dimension halves the cross-validated error of ln(epochs needed)
(0.94 -> 0.41 RMS); estimated dimension is nearly as good as the true value; low d needs far more epochs; and the
measured size exponents (+0.22 at d = 16, +0.24 at d = 24) falsify both pre-registered dimension laws' exponent
predictions (reduced quadratic predicted negative above d ~ 13.8; the 1/d form predicted a decreasing exponent).
