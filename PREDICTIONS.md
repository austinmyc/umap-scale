# Pre-registered epoch-budget laws for UMAP layout optimization

**Frozen on 2026-11-12, before any synthetic-manifold or 1M+ point data was generated.**
The machine-readable version is [`laws/frozen_laws.json`](laws/frozen_laws.json). These coefficients must not be refit on test data;
any later refit is reported separately as a new, post-hoc law.

## Question

How many layout epochs does UMAP need, as a function of dataset properties known *before* running it?
umap-learn uses a fixed rule (500 epochs for n ≤ 10,000, otherwise 200). On four development datasets that rule
spends 3–15× more epochs than needed to reach 95% of the 500-epoch quality with the UMAP 0.6 Adam optimizer.

## Definitions (fixed)

- **Software**: umap-learn 0.6.0 (PyPI release; corrected from "0.6.0rc1", see note at the end). `adam` = `optimizer="adam", compatibility_layout=False`;
  `compatibility` (legacy SGD) = `compatibility_layout=True`. `n_neighbors=15, min_dist=0.1, n_components=2`.
- **Quality**: neighbor recall = mean fraction of each point's 15 nearest neighbors in input space that are among its
  15 nearest neighbors in the embedding, over 5,000 random evaluation points.
- **Target E95**: smallest epoch budget whose recall reaches 95% of the same subset's 500-epoch recall, log-interpolated
  between budgets {5, 10, 25, 50, 100, 200, 500}.
- **n**: number of points embedded. **d**: intrinsic dimension = Levina–Bickel MLE (k = 15, MacKay–Ghahramani averaging,
  4,000 query points, seed 0) computed on the exact subset being embedded; integer-valued features are dequantized with
  U(−0.5, 0.5) noise before standardizing. Labels are never used to choose a budget.
- **Development data used to fit**: Fashion-MNIST and MNIST (PCA to 50 dims), Letter and Pendigits (standardized);
  random subsets of 1.4k–70k points; one seed. Raw results are in [`results/dev/`](results/dev/).

## Competing laws (L = ln n, I = ln d, E = e^(·))

Each law encodes a different assumption. All coefficients below are final.

### UMAP 0.6 Adam

- **default_rule** (current umap-learn rule): `500 if n <= 10000 else 200`
- **constant** (budget independent of data): `ln E = +3.8914·1`
- **power_law** (only n matters): `ln E = +2.3127·1 +0.1671·L`
- **power_law_p90** (only n matters, budget set to clear the target in ~90% of cases): `ln E = +2.6229·1 +0.1671·L`
- **theory_1_over_d** (size exponent ~ a/d (graph-diffusion argument)): `ln E = +2.7208·1 +0.1464·L -0.1631·L/d`
- **reduced_quadratic** (size exponent linear in ln d (sign change allowed)): `ln E = -1.4872·1 +0.5562·L +2.0922·I -0.2122·L*I`
- **reduced_quadratic_ridge1** (as above, ridge 1.0 on standardized basis): `ln E = +2.4171·1 +0.1018·L +0.2145·I +0.0044·L*I`

Predicted epochs (rows: intrinsic dimension d; columns: n):


**constant**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 49 | 49 | 49 | 49 | 49 |
| 8 | 49 | 49 | 49 | 49 | 49 |
| 12 | 49 | 49 | 49 | 49 | 49 |
| 16 | 49 | 49 | 49 | 49 | 49 |
| 24 | 49 | 49 | 49 | 49 | 49 |

**power_law**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 47 | 69 | 83 | 102 | 122 |
| 8 | 47 | 69 | 83 | 102 | 122 |
| 12 | 47 | 69 | 83 | 102 | 122 |
| 16 | 47 | 69 | 83 | 102 | 122 |
| 24 | 47 | 69 | 83 | 102 | 122 |

**power_law_p90**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 64 | 94 | 113 | 139 | 167 |
| 8 | 64 | 94 | 113 | 139 | 167 |
| 12 | 64 | 94 | 113 | 139 | 167 |
| 16 | 64 | 94 | 113 | 139 | 167 |
| 24 | 64 | 94 | 113 | 139 | 167 |

**theory_1_over_d**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 40 | 51 | 58 | 65 | 73 |
| 8 | 49 | 65 | 74 | 87 | 100 |
| 12 | 52 | 70 | 81 | 95 | 110 |
| 16 | 53 | 73 | 85 | 100 | 116 |
| 24 | 55 | 76 | 88 | 105 | 122 |

**reduced_quadratic**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 46 | 84 | 112 | 153 | 205 |
| 8 | 51 | 66 | 75 | 86 | 97 |
| 12 | 53 | 57 | 59 | 61 | 63 |
| 16 | 56 | 52 | 50 | 48 | 46 |
| 24 | 59 | 45 | 39 | 34 | 30 |

**reduced_quadratic_ridge1**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 41 | 52 | 59 | 67 | 75 |
| 8 | 49 | 63 | 71 | 81 | 92 |
| 12 | 54 | 70 | 79 | 91 | 103 |
| 16 | 58 | 76 | 86 | 98 | 111 |
| 24 | 64 | 84 | 96 | 110 | 125 |

### Legacy SGD (compatibility_layout=True)

- **default_rule** (current umap-learn rule): `500 if n <= 10000 else 200`
- **constant** (budget independent of data): `ln E = +4.9065·1`
- **power_law** (only n matters): `ln E = +3.0993·1 +0.1913·L`
- **power_law_p90** (only n matters, budget set to clear the target in ~90% of cases): `ln E = +3.2401·1 +0.1913·L`
- **theory_1_over_d** (size exponent ~ a/d (graph-diffusion argument)): `ln E = +3.8875·1 +0.1513·L -0.3150·L/d`
- **reduced_quadratic** (size exponent linear in ln d (sign change allowed)): `ln E = +2.3036·1 +0.1970·L +0.7771·I -0.0423·L*I`
- **reduced_quadratic_ridge1** (as above, ridge 1.0 on standardized basis): `ln E = +3.3601·1 +0.0762·L +0.2724·I +0.0148·L*I`

Predicted epochs (rows: intrinsic dimension d; columns: n):


**constant**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 135 | 135 | 135 | 135 | 135 |
| 8 | 135 | 135 | 135 | 135 | 135 |
| 12 | 135 | 135 | 135 | 135 | 135 |
| 16 | 135 | 135 | 135 | 135 | 135 |
| 24 | 135 | 135 | 135 | 135 | 135 |

**power_law**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 129 | 201 | 248 | 312 | 384 |
| 8 | 129 | 201 | 248 | 312 | 384 |
| 12 | 129 | 201 | 248 | 312 | 384 |
| 16 | 129 | 201 | 248 | 312 | 384 |
| 24 | 129 | 201 | 248 | 312 | 384 |

**power_law_p90**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 149 | 231 | 285 | 359 | 443 |
| 8 | 149 | 231 | 285 | 359 | 443 |
| 12 | 149 | 231 | 285 | 359 | 443 |
| 16 | 149 | 231 | 285 | 359 | 443 |
| 24 | 149 | 231 | 285 | 359 | 443 |

**theory_1_over_d**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 95 | 112 | 122 | 133 | 144 |
| 8 | 137 | 177 | 200 | 229 | 259 |
| 12 | 154 | 206 | 236 | 274 | 315 |
| 16 | 164 | 222 | 256 | 301 | 347 |
| 24 | 174 | 239 | 279 | 329 | 383 |

**reduced_quadratic**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 105 | 144 | 168 | 199 | 231 |
| 8 | 137 | 177 | 199 | 227 | 256 |
| 12 | 161 | 199 | 220 | 245 | 271 |
| 16 | 180 | 216 | 236 | 259 | 283 |
| 24 | 210 | 243 | 260 | 280 | 300 |

**reduced_quadratic_ridge1**

| d \ n | 10,000 | 100,000 | 300,000 | 1,000,000 | 3,000,000 |
|---|---|---|---|---|---|
| 4 | 102 | 128 | 142 | 160 | 178 |
| 8 | 136 | 174 | 196 | 222 | 250 |
| 12 | 160 | 208 | 236 | 270 | 306 |
| 16 | 180 | 236 | 269 | 310 | 352 |
| 24 | 213 | 283 | 324 | 376 | 430 |
## Pre-registered evaluation

For each test case (dataset, n) and each law: run the same budget grid, compute the predicted budget E_law, and score
**q = recall(E_law) / recall(500)** by log-interpolating the measured recall curve, plus **cost = E_law**.

- **Pass criterion**: median q ≥ 0.94 **and** q ≥ 0.90 in at least 90% of test cases.
- **Ranking**: among passing laws, lowest total epochs. The default rule is reported as a reference.
- **Reference check at large n**: for n ≥ 1M, also run 1,000 epochs. If recall improves by more than 2% over 500 epochs,
  1,000 epochs becomes that case's reference and this is reported.
- **Seeds**: at least 3 per test case; q is computed on the seed-averaged recall curve.

## Falsifiable predictions

1. **Sign of the size exponent at high intrinsic dimension (Adam).** The `reduced_quadratic` law implies
   ∂ln E/∂ln n = 0.556 − 0.212·ln d, which is negative above d ≈ 13.8: at d = 16 it predicts ~56 epochs at 10k points
   but ~48 at 1M. `power_law` and `theory_1_over_d` predict a positive exponent at every d. Synthetic manifolds with
   known d ∈ {2, 4, 8, 12, 16, 24} decide between them.
2. **Sign of the dimension effect depends on n (Adam, reduced_quadratic).** ∂ln E/∂ln d = 2.092 − 0.212·ln n, positive
   below n ≈ 19,300 and negative above; at 1M points doubling d multiplies the predicted budget by ~0.56.
   `theory_1_over_d` predicts a positive dimension effect at all n.
3. **Legacy SGD may be under-trained by the default at large n.** For `compatibility`, `power_law` predicts ~312 epochs
   at 1M points and `theory_1_over_d` predicts 229–301 for d = 8–16, above the default 200. If confirmed, the default
   rule over-trains small datasets and under-trains large ones.

## Planned test data (1M+ points)

At least two of: 10x Genomics 1.3M mouse brain cells (single-cell RNA-seq, PCA-50); ImageNet-1k CLIP or DINO image
embeddings (~1.28M, labeled); SIFT1M or Deep10M (unlabeled vectors). Subsets at n ∈ {100k, 300k, 1M, full}.

## Corrections log

- **2026-11-12**: the software version was originally written as umap-learn 0.6.0rc1. All development runs actually used
  umap-learn 0.6.0, the PyPI release. Only the version label changed; no law, coefficient or criterion was altered.

## Deviations log

- **D1 (2026-11-12, before any synthetic run): loss-based target for synthetic manifolds.** On uniform synthetic
  manifolds with d ≥ 8, neighbor recall peaks after ~25 epochs and then declines (e.g. d = 16, n = 10k:
  0.110 / 0.079 / 0.086 at 25 / 100 / 500 epochs), so the recall-based E95 sits at the smallest budget for every n and
  cannot test the laws. The synthetic test therefore also uses a loss-based target: E95_loss = first budget reaching 95%
  of the reduction in UMAP's layout objective (weighted edge attraction + 5 × mean repulsion over random pairs) from the
  initial layout (budget 0) to the 500-epoch layout. This is the quantity the convergence argument in the theory is
  about. Both targets are reported; the frozen laws are unchanged and are scored against both. Prediction error is also
  reported as ln(E_law / E95_measured), which discriminates between laws even where quality curves are flat.
- **D2 (2026-11-12): synthetic generator changed before any synthetic run.** Uniform latent cube (no clusters, so
  cluster separation cannot change with d), 64 random frequencies, frequency 1.0. At frequency 2.0 the estimated
  intrinsic dimension exceeded the true d for d ≥ 12 at 10k points (d = 16 → MLE 29), i.e. the manifold was not
  d-dimensional at the scale UMAP sees; at 1.0 the MLE tracks d (12 → 10.7, 16 → 14.7, 24 → 22.9).
