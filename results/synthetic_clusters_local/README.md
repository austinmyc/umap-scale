# Clustered synthetic manifolds (local CPU sweep)

Config: `configs/synthetic_clusters_local.yaml`. 10 Gaussian blobs in a d-dimensional latent space, with separation and
width fixed in absolute distance, mapped by the same random Fourier-feature map as the uniform family.
True d ∈ {2, 4, 8, 12, 16, 24} × n ∈ {2.5k … 40k} × 3 realizations; UMAP 0.6 Adam; 1,215 runs. Full output: `report.json`.

## E95_best (epochs to 95% of the best recall at any budget), mean over realizations

| d | 2.5k | 5k | 10k | 20k | 40k | size exponent (90% CI) |
|---|---|---|---|---|---|---|
| 2 | 52 | 50 | 102 | 66 | 155 | +0.32 [+0.27, +0.39] |
| 4 | 39 | 44 | 43 | 48 | 55 | +0.10 [+0.05, +0.12] |
| 8 | 38 | 42 | 44 | 35 | 36 | −0.05 [−0.08, −0.01] |
| 12 | 43 | 45 | 42 | 50 | 55 | +0.08 [+0.06, +0.10] |
| 16 | 50 | 52 | 48 | 63 | 60 | +0.07 [+0.05, +0.12] |
| 24 | 53 | 51 | 51 | 61 | 53 | +0.02 [−0.01, +0.06] |

## Findings

- Same picture as the uniform family: 35–65 epochs suffice everywhere except d = 2, where the need grows with n
  (exponent about 0.3).
- Global structure (Spearman correlation of distances) is at its best value within 5–10 epochs at every d.
  More epochs only refine local neighborhoods.
- The loss target behaves differently from quality. At d ≥ 12 the loss converges *faster* as n grows (exponent −0.3 to
  −0.5), while the recall need stays flat. The loss is therefore not a good stand-in for quality when setting budgets.
- Dimension: under leave-one-d-out, a full quadratic in (ln n, ln d) has RMS log error 0.27–0.29 vs 0.36 for a constant
  law. That is a small gain, and it disappears when pooled with the uniform and real data (see `PREDICTIONS_v2.md`).
- MLE underestimates d = 24 as about 15 on this family (clusters are thin in some directions).
