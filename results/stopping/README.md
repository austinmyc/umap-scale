# In-training stopping rules (UMAP 0.6 Adam), local CPU

Code: `src/umapscale/stopping.py` (monitor, rules, Grow controller), `src/umapscale/stopping_analysis.py` (scoring).
Score q = recall15 of the returned embedding / best recall15 over fixed-budget runs (or the trajectory, if higher).
Cost = epochs run / umap's default (500 if n <= 10k, else 200). One seed. 2-D: 42 cases (18 real, 12 uniform,
12 clustered synthetic at 10k and 40k); 5-D and 10-D: 12 cases each (6 real, 6 synthetic at 10k).

## 1. Stopping a default-length run (offline replay of 42 trajectories)

No rule works. Stopping mid-schedule leaves the learning rate high: a 500-epoch run stopped at 40 reaches q = 0.82,
while a fresh 40-epoch run reaches about 0.95. Residual intrinsic dimension of a 2-D embedding is pinned at 2.00 ± 0.03
and fires on noise. Full table: `screen_m2.txt`.

## 2. Stop + anneal (rule fires at t*, learning rate decays linearly to 0 over max(10, 0.25 t*) epochs)

| Rule | 2-D: q>=0.9 / cost | 5-D | 10-D |
|---|---|---|---|
| recall plateau 0.005 | 79% / 30% | 100% / 21% | 83% / 26% |
| displacement 0.02 | 67% / 33% | 100% / 29% | 100% / 29% |
| neighbor stability 0.4 | 81% / 65% | 92% / 44% | 75% / 46% |
| covariance residual 0.005 | 76% / 63% | 67% / 29% | 83% / 40% |

All 2-D failures are uniform manifolds with d >= 8: along a default-length run, recall never gets within 61–77% of a
fresh 35-epoch run, because the long warm-up at high learning rate scrambles neighborhoods. No stopping time can fix that.

## 3. Grow: short schedule first, warm-started extensions while they help

Stage 1 is a ~fresh 25-epoch run; each extension (25, 50, 100, 200 epochs) continues the same embedding and Adam
state with UMAP's schedule for that length scaled by 0.3. Stop when monitor recall improves by less than `tol`;
return the best stage end. Rows for tol > 0.02 are offline replays of the same runs using the recorded stage scores
(monitor recall), so they are approximate.

| tol | 2-D: q>=0.9 / min q / cost / median epochs | 5-D | 10-D |
|---|---|---|---|
| 0.02 (run) | 98% / 0.89 / 63% / 150 | 100% / 0.92 / 32% / 100 | 83% / 0.81 / 38% / 100 |
| 0.03 | 98% / 0.89 / 49% / 100 | 100% / 0.91 / 28% / 100 | 83% / 0.77 / 25% / 100 |
| 0.05 | 95% / 0.88 / 35% / 100 | 100% / 0.91 / 25% / 100 | 75% / 0.77 / 21% / 50 |

- Grow is the only method that handles the 2-D high-d uniform cases (q >= 0.89 on all of them).
- It often exceeds the best fixed-budget run (q up to 1.08; Letter 20k 1.05): a low-learning-rate continuation
  finds layouts that no single UMAP schedule length reaches.
- Remaining failures are 10-D uniform manifolds with a long, slow tail (best at 500+ epochs) that flattens for one
  doubling and then improves again; any doubling rule with a relative tolerance stops there.
- tol was chosen on these same cases. It must be frozen before the large-scale test.

**Correction (after freezing v3).** The monitor compared 14 input neighbors with 15 embedding neighbors. Scores in
sections 2–3 above read about 3% high. Corrected results for the frozen grow rule (τ = 0.03, ρ = 0.3) are in the appendix of
`PREDICTIONS_v3.md`: real data reach q ≥ 0.90 in 100% of cases at 2, 5 and 10 output dimensions (cost 33–49% of
default); uniform synthetic manifolds fail in 4 of 12 cases in 2-D and 2 of 4 in 10-D.
