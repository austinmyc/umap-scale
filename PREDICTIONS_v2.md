# Pre-registration v2: epoch-budget laws under the best-quality target

**Frozen on 2026-11-12, before any 1M+ point or other unseen large dataset was run.**
Machine-readable laws: [`laws/frozen_laws_v2.json`](laws/frozen_laws_v2.json). Fitting script: `python -m umapscale.fit_v2`
(cross-validation output in [`laws/candidates_v2.json`](laws/candidates_v2.json)).
The v1 laws in [`PREDICTIONS.md`](PREDICTIONS.md) stay frozen and are scored in the same test, as competitors.

## Why a second registration

The v1 synthetic tests (see `PREDICTIONS.md`, deviations D1–D3, and `results/synthetic_local/README.md`) showed:

1. The v1 recall target was degenerate on many manifolds. Recall peaks and then *declines* with more epochs at d ≥ 8, so
   "95% of the 500-epoch recall" can be reached at almost any budget. v2 uses **E95_best**, defined below.
2. Both v1 dimension laws were falsified on their size exponent. Across 90 uniform and 90 clustered synthetic cases,
   intrinsic dimension does not improve prediction of E95_best on unseen d. On the clustered family, a full quadratic is
   slightly better than the constant law for unseen d (RMS 0.27–0.29 vs 0.36 in ln E). The pooled v2 fit shows no gain.
3. The size exponent is close to 0 for d ≥ 8 on both synthetic families and about +0.3 for d = 2. On the real development
   data it is about +0.15 to +0.17.

## Definitions (fixed)

- Software, n, d (Levina–Bickel MLE), recall15: exactly as in `PREDICTIONS.md`. Optimizer: UMAP 0.6 Adam
  (`optimizer="adam", compatibility_layout=False`).
- **E95_best**: for each case, average recall15 over seeds at each budget in
  {5, 10, 15, 25, 35, 50, 70, 100, 140, 200, 300, 500} (+1000 where run). Let R* be the best recall at any budget.
  E95_best is the smallest budget whose recall reaches 0.95·R*, log-interpolated between grid points.
- **Score of a budget e**: q(e) = recall(e) / R*, log-interpolated. Over-training is penalized when recall declines.
- **Pass** (unchanged from v1): median q ≥ 0.94 and q ≥ 0.90 in at least 90% of test cases.
- **Cost**: total epochs relative to the default rule (500 if n ≤ 10,000, else 200).

## Frozen laws (L = ln n, I = ln d, E = e^(·))

Fit on 18 real development cases, 90 uniform synthetic cases and 90 clustered synthetic cases, with each source
weighted equally.

| Law | Assumption | Formula | E at 1M, d = 10 |
|---|---|---|---|
| default_rule | current umap-learn | 500 if n ≤ 10k else 200 | 200 |
| v2_constant | budget independent of data | ln E = 3.7202 (E = 41.3) | 41 |
| v2_power_law | only n matters | ln E = 2.4893 + 0.1332·L (E = 12.05·n^0.133) | 76 |
| v2_power_law_plus_dim | n plus a level in ln d | ln E = 2.7154 + 0.1526·L − 0.2059·I | 77 |
| v1 power_law (competitor) | only n, real data only | E = 10.10·n^0.167 | 101 |

Predicted budgets (epochs):

| n | d | v2_constant | v2_power_law | v2_power_law_plus_dim |
|---|---|---|---|---|
| 100k | 5 / 10 / 20 | 41 | 56 | 63 / 55 / 47 |
| 300k | 5 / 10 / 20 | 41 | 65 | 74 / 64 / 56 |
| 1M | 5 / 10 / 20 | 41 | 76 | 89 / 77 / 67 |
| 3M | 5 / 10 / 20 | 41 | 88 | 106 / 92 / 79 |

Cross-validation on the fitting data (from `fit_v2`):

| Law | Leave-dataset-out: median q / q ≥ 0.9 / epochs | Leave-source-out | Leave-dimension-out |
|---|---|---|---|
| v2_constant | 0.946 / 0.90 / 41 | 0.937 / 0.82 / 43 | 0.946 / 0.90 / 41 |
| v2_power_law | 0.945 / 0.91 / 42 | 0.931 / 0.79 / 43 | 0.943 / 0.90 / 42 |
| v2_power_law_plus_dim | 0.947 / 0.88 / 42 | 0.936 / 0.78 / 44 | 0.949 / 0.89 / 46 |

All closed-form laws are at the pass boundary on their own fitting data. Under the same validation, the default rule uses 379 epochs on average.
Shifting the intercept up to clear 90% of training cases (the p90 variants) made things *worse* (q ≥ 0.9 in 80–85%),
because the larger budget over-trains the high-d manifolds. No p90 variant is frozen.

## Frozen procedure: adaptive doubling

Run budgets 25, 50, 100, 200, 500 (each a fresh run with the same seed). Stop at the first budget a whose next doubling
b improves recall by less than 2%, i.e. recall(b) < 1.02·recall(a), and return the budget-a embedding. Cost = sum of all
budgets run. This needs no law and no dimension estimate. It uses recall15, which needs exact or approximate kNN of the
input, already computed by UMAP.

On the fitting data: q ≥ 0.90 in 100% of cases, median q 0.984. Mean cost was 272 epochs vs 379 for the default:
177 vs 380 on uniform, 352 vs 380 on clustered, and 353 vs 367 on real data. So **with restarts it saves little on real
data**. A warm-start variant would continue training from the previous embedding instead of restarting, at a cost equal
to the last budget run (about 152 on average on the fitting data). It changes the Adam schedule, has **not** been tested,
and is not part of this registration.

## Test plan (unseen data only)

- At least three large datasets not used above, each at n ∈ {100k, 300k, 1M} (plus 3M where feasible). At least one
  should be real non-image data (e.g. single-cell, text embeddings).
- 3 seeds per case. Budgets as in the definition of E95_best, with 1000 added at n ≤ 300k. faiss kNN. Config:
  `configs/large_gpu.yaml`.
- Score: `python -m umapscale.evaluate <runs.jsonl> --target best --laws laws/frozen_laws_v2.json`, and the same with
  `laws/frozen_laws.json` for the v1 competitors. Report every law, including failures.

## Falsifiable predictions

1. **Constant vs power law** (the main discriminating test). If real large datasets behave like the development real
   data (E95_best rising with n, exponent around 0.15), v2_constant fails at 1M (q < 0.90 in more than 10% of cases) and
   v2_power_law passes. If they behave like the synthetic manifolds with d ≥ 8 (flat in n), v2_constant passes, and
   v2_power_law also passes at about 1.8× its cost. On the development data the constant already failed 4 of 18 real
   cases (q 0.88–0.90), so we expect the first outcome.
2. **Size exponent**: the exponent of E95_best on n, fit per dataset over 100k–1M, lies in [0, 0.25] for every test
   dataset. An exponent above 0.3 on any real dataset with MLE d ≥ 5 falsifies all v2 laws.
3. **Dimension**: v2_power_law_plus_dim does not beat v2_power_law by more than 0.05 in RMS log error. If it does,
   dimension helps on real data even though it did not on synthetic data.
4. **Default rule**: passes, at 2–4× the cost of v2_power_law.
5. **Adaptive doubling**: q ≥ 0.90 in at least 90% of cases.
6. **Flat plateau check** (the concern about one 1M result where 71 and 102 epochs both sit on a flat recall plateau):
   at each size we report q at 0.5×, 1× and 2× each law's budget. A law is credited only if q(0.5× budget) < 0.95 in at
   least half the cases, i.e. its budget is not trivially on the plateau. Otherwise we report that the test cannot tell
   the laws apart at that size.

## Known limitations

- Fitting n spans 1.4k–70k; the 1M predictions are 14–700× extrapolations. Prediction 1 is chosen because the two v2
  laws diverge most there.
- q is measured against the best budget on the grid; a finer grid could change E95_best by up to one grid step
  (about 40% between grid points).
- Clustered synthetic d = 24 has MLE d about 15; laws are fed MLE d, as they would be in practice.
