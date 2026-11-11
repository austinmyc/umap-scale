# Controlled synthetic sweep (local, CPU)

Uniform manifolds with known intrinsic dimension d in {2, 4, 8, 12, 16, 24}, n in {2.5k ... 40k}, 3 realizations,
UMAP 0.6 Adam, budgets 0-500 plus 1000-epoch references for d <= 8 (`configs/synthetic_local.yaml`,
`configs/synthetic_local_reference.yaml`). Analysis: `python -m umapscale.synthetic_analysis results/raw/synthetic_local.jsonl`.

- `report_ref500.json`: first analysis, 500-epoch reference everywhere (superseded for d <= 8).
- `report.json`: current analysis; the 1000-epoch reference is used where it gives >2% more loss reduction
  (as pre-registered): 14 of 15 d = 2 cases, 12 of 15 d = 4 cases, 5 of 15 d = 8 cases.

## Findings

1. **Dimension carries real information.** Over d = 2-24, adding dimension cuts the cross-validated RMS error of
   ln(epochs needed, loss target) from 1.21 (n only) to 0.53-0.63; the reduced quadratic generalizes best to an unseen
   d (0.73 vs 1.43). Estimated dimension (MLE/TwoNN) is nearly as good as the true value.
2. **Both pre-registered dimension laws are falsified on the size exponent.** Measured exponents at d = 12/16/24 are
   +0.18/+0.22/+0.24 with tight bootstrap intervals; the frozen reduced quadratic predicted a negative exponent above
   d ~ 13.8 and the 1/d form predicted a decreasing one. No frozen law meets the pass criterion on this data
   (61-70% of cases reach q >= 0.90 vs the required 90%); the default rule does (91%).
3. **Two regimes.** For d >= 8, a reference-free criterion (doubling the budget adds < 5% more loss reduction) gives a
   tight 35-70 epochs, growing slowly with n, matching the real datasets. For d = 2 no budget up to 1000 satisfies it.
4. **The loss keeps falling where quality has stopped improving.** At d = 2, neighbor recall saturates by ~100-200 epochs
   and global distance correlation by ~50 (0.98); at d = 4-8 global correlation declines with more epochs
   (0.72 -> 0.65 at n = 40k). Loss-based targets therefore overstate useful epochs at low d, while the 500-epoch recall
   target fails at high d (recall peaks at ~25 epochs, then declines).
5. **Proposed next target** (not yet applied): epochs to reach 95% of the best quality achieved at any budget, which is
   defined in both regimes.
