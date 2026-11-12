# Pre-registration v3: the grow rule (primary) and a spectral law (secondary)

**Frozen on 2026-11-12, before any large or unseen dataset was run.** Machine-readable: [`laws/frozen_v3.json`](laws/frozen_v3.json).
The v1 and v2 laws ([`PREDICTIONS.md`](PREDICTIONS.md), [`PREDICTIONS_v2.md`](PREDICTIONS_v2.md)) stay frozen and are
scored in the same test. Definitions of q, E95_best, pass, n and d are those of v2. The default rule is 500 epochs if
n ≤ 10,000, otherwise 200.

## Why v3

Development work after v2 (`results/stopping/README.md`) showed:

1. **Output dimension matters for fixed budgets.** With 5 or 10 output dimensions, uniform synthetic manifolds need
   122–362 epochs to reach E95_best, against 22–49 in 2-D. A budget law fitted in 2-D under-budgets them by up to ~16×.
2. **Stopping a default-length run cannot work in 2-D at high intrinsic dimension.** UMAP 0.6's Adam schedule spends
   its first min(100, N/8) epochs at a high learning rate. On uniform manifolds with d ≥ 12, no point along a 500-epoch
   run gets within 61–77% of the best recall of a fresh 35-epoch run.
3. **Starting short and extending while it helps works**, in any output dimension, with no dataset features.

## Primary: the grow rule

UMAP 0.6's Adam learning-rate schedule for a run of N epochs, as implemented in `umap/layouts.py`
(good initialization, base rate 1), with w(N) = ⌊min(100, N/8)⌋:

- α_N(j) = 0.4·(1 − j/w)² + 0.1 for j < w
- α_N(j) = 0.15·(1 − (j − w)/(N − w)) + 0.05 for j ≥ w

**Stages.** L = (25, 25, 50, 100, 200), cumulative T = (25, 50, 100, 200, 400). At global epoch t, with j the epoch
index within its stage:

- stage 1: α(t) = α₂₅(t)
- stage k ≥ 2: α(t) = ρ·α_{L_k}(j), **ρ = 0.3**

The embedding and the Adam moment estimates carry over between stages; nothing restarts.

**Monitor.** After each stage, r_k = (1/|S|) Σ_{i∈S} |N^in(i) ∩ N^emb(i)| / k_mon. S is a fixed random sample of 2,000
points (seed 0). N^in is the point's neighbors in UMAP's own input kNN table (excluding itself, so k_mon = 14 for
n_neighbors = 15), and N^emb is the same number of nearest neighbors in the current embedding, in any output dimension.

**Stop and return.** After stage k ≥ 2, stop if r_k < (1 + τ)·r_{k−1}, **τ = 0.03**; otherwise continue, up to stage 5.
Return the embedding of the stage with the largest r. Cost = epochs run (T at the stopping stage).

Implementation: `umapscale.stopping.Grow`; sweep key `grow: {stages: [25, 25, 50, 100, 200], rho: 0.3, tau: 0.03}`.

**Known differences from the description.** Stage 1 is not bit-identical to a fresh 25-epoch run (recall 0.100 vs
0.094 in one check), probably because the random state is consumed differently. In parallel mode (no fixed
random_state) runs are not reproducible.

## Secondary: spectral law (2-D output only)

ln E = 2.5372 + 0.0933·ln n − 0.0346·ln λ_c + 0.0863·ln K, i.e. **E = 12.64 · n^0.093 · λ_c^−0.035 · K^0.086**

- λ_c: smallest eigenvalue above 10⁻⁶ among the 12 smallest of the symmetric normalized Laplacian of UMAP's fuzzy graph.
- K: number of connected components of that graph.

Fit on the 198 development cases under E95_best, each source weighted equally. In the large test both are computed by
the sweep (`spectral_features: true`) on the graph built from the subset as embedded. The development features were
computed with exact kNN, and on dequantized data for Letter and Pendigits; the test uses UMAP's or FAISS's kNN on the
data as embedded. Leave-one-dataset-out on development data: RMS log error 0.47 vs 0.52 for the v2 power law;
q ≥ 0.9 in 91% of cases.

## Development performance of the grow rule (corrected metric)

See the end of this file. These numbers were measured after freezing, with the monitor-metric bug fixed. Earlier
development scores compared 14 input neighbors with 15 embedding neighbors and read about 3% high. τ and ρ were chosen
from those earlier runs, and the bias affects all τ values alike.

## Test plan

- `configs/large_gpu.yaml`: two real datasets (mouse brain single-cell, ImageNet CLIP) at 100k, 300k, 1M and full size,
  plus three synthetic manifolds at 100k–1M. 2-D output; 3 seeds; fixed-budget grid as in v2; one grow run per seed.
- `configs/large_gpu_m10.yaml`: the two real datasets at 100k and 300k with 10-D output (grow rule and fixed grid).
- Score: `python -m umapscale.evaluate <runs.jsonl> --target best --laws laws/frozen_v3.json`; also run with the v1 and
  v2 law files.

## Falsifiable predictions

1. **Grow passes on real data in 2-D**: median q ≥ 0.94 and q ≥ 0.90 in at least 90% of real-data runs, with mean
   cost ≤ 60% of the default rule.
2. **Grow passes on real data in 10-D** under the same criterion (`large_gpu_m10.yaml`).
3. **Grow is at least as reliable as every v1 and v2 law**: its share of real-data runs with q ≥ 0.90 is no lower
   than that of any law, at every size.
4. **Grow's cost does not grow faster than the v2 power law** across 100k–1M: the median epochs run grows by at most
   a factor (10)^0.133 ≈ 1.36 between 100k and 1M.
5. **Synthetic manifolds**: grow reaches q ≥ 0.90 on the d = 16 and clustered d = 8 manifolds. The uniform d = 4
   manifold is a known hard case (long improvement tail); failure there is reported, not counted against prediction 1.
6. **Spectral law**: RMS log error on the 2-D real test cases at most that of v2_power_law (no worse). A gain larger
   than 0.05 would support using graph spectra; anything smaller is reported as no evidence for it.

Exploratory (not pass/fail): the share of runs where grow exceeds the best fixed-budget run (q > 1), and wall-clock
time of grow vs the default and vs the best law.
