# OURS (`NonsequentialArmijoAdaptiveGD`) vs. Algorithm 1

Compared: Algorithm 1 as printed in the manuscript (PROMPT.md, Appendix B) against
`src/methods/gd_non_sequential_armijo_adaptive.jl` (**A**) and
`src/methods/line_search_helpers/non_sequential_armijo.jl` (**H**), on branch
`cv/experiments-quasi-likelihood`. Date: 2026-11-07. **OURS's behavior was changed in exactly one place, on 2026-11-08 (row 17, fixed at Vivak's request after the manuscript confirmed the bug). The comparison below describes the code before that fix except where marked.**

Notation: the paper's outer index `k` starts at 0 and its inner index `j` at 0. In the code
the outer counter `iter` (called `k` inside `inner_loop!`) starts at 1, so code `k` = paper `k`+1,
and `grad_val_hist[iter]` is `‖Ḟ(θ_k)‖`. The code's inner counter `j` is incremented *before* a
step, so code `j` = paper `j`+1 inside the loop body.

## Summary

* **One real discrepancy** (row 17): the Lipschitz estimate at the terminal inner iterate was
  off by one when the inner loop took exactly one step (`j_k = 1`). It changed the trajectory,
  and `j_k = 1` is common (about 23% of inner loops). **Fixed**; the solver now equals the paper transcription exactly.
* Everything else matches Algorithm 1 once the printing errors in the paper are read as intended
  (rows 2, 5, 13, 15). The remaining differences are cosmetic (rows 18–19).
* `test/methods/ours_vs_algorithm1.jl` checks all of this mechanically: an independent literal
  transcription of Algorithm 1 (with the corrected `while`, 0-based `j`, and `√20·τ_low`)
  reproduces the solver's iterates, `j_k`, event flags, acceptance flags, `δ`, `τ_low`, `τ_upp`
  for 40 outer iterations from 3 random starts on a `QLLogisticSin` problem (392 checks pass).

## Line-by-line table

| # | Paper (Algorithm 1) | Code | Difference | Which is right |
|---|---|---|---|---|
| 1 | `τ_low⁰, τ_upp⁰ ← ‖Ḟ(θ₀)‖/√2, √10‖Ḟ(θ₀)‖` | A:504–505 | none | same |
| 2 | `while ‖Ḟ(θ_k)‖ > ε` | A:513–514 `iter < max_iterations && grad_val_hist[iter+1] > threshold` | extra `max_iterations` guard (budget handled by the recorder, §3) | code (needed in practice). The threshold test is on **outer iterates only** (`grad_val_hist[iter+1]`), never on inner iterates; hits at ε are therefore recorded only at outer iterates (rejected iterations repeat the previous norm). |
| 3 | `j, ψ₀ᵏ ← 0, θ_k` | A:367, 369 (`j = 0`, `norm_∇F_ψ = grad_val_hist[k]`); `ψ` buffer is `x`, which equals `θ_k` on entry (`x` is reset to `iter_hist[iter]` on rejection, H:101) | none | same |
| 4 | `L̂ ← EstimateLipschitz(j,k,ψ_j,ψ_{j−1})` at `j = 0` (needs `ψ_{−1}`, which does not exist) | A:379–381 call at the top of each loop pass; at code `j==1` (paper `j=0`) returns `1.0` if `k==1`, else the previous estimate (A:216–222) | paper is undefined at `j=0`; code defines it as "no information yet": carry the last estimate | code (the paper's call is not well-defined). |
| 5 | inner `while ‖ψ−θ‖ > 10 or ‖Ḟ(ψ)‖ ∉ (τ_low,τ_upp) or j == 100` | A:371–373 `norm(ψ−θ) <= radius && τ_lower < n < τ_upper && j < max_iteration` | paper's condition is printed **inverted** (it would stop immediately and never loop) | code (matches the paper's text/event definitions: stop on ball, on leaving the gradient interval, or at `j = 100`). |
| 6 | `C₁ = ‖Ḟ‖³ + .5‖Ḟ‖²L̂ + 1e-16`, `C₂ = ‖Ḟ‖ + .5L̂ + 1e-16` | `compute_step_size`, A:287–288 | none | same |
| 7 | `α_j ← min((τ_low)²/C₁, 1/C₂) + 1e-16` | A:289; called with `τ_lower`, `norm_∇F_ψ`, `local_lipschitz_estimate` at A:384 | none | same |
| 8 | `ψ_{j+1}, j ← ψ_j − δ_k α_j Ḟ(ψ_j), j+1` | A:393 `ψjk .-= (δk*αjk) .* store.grad`; `j += 1` at A:376 (before the step); `α0k` saved at A:389 | order of `j` increment (cosmetic) | same |
| 9 | gradient at the new point | A:396–399: saves `prev_norm_step`, `prev_∇F_ψ`, then `grad!` and `norm_∇F_ψ` | none (1 gradient per inner step) | same |
| 10 | `L̂ ← EstimateLipschitz(j,k,ψ_j,ψ_{j−1})` after each step | A:379–381 at the **start of the next pass**, and once after the loop at A:408 | in-loop estimates are identical (computed one pass later); the **post-loop call is the problem**, see row 17 | see 17 |
| 11 | `if F(ψ_j) ≥ F(θ_k) − ρ δ_k α₀ᵏ ‖Ḟ(θ_k)‖²` reject | A:526–530, H:67 accept iff `F < ref − ρ δ α0k ‖g_k‖²`, where `ref = F(θ_k)` (A:509, A:540) and `‖g_k‖ = grad_val_hist[iter]` | accept = complement of reject (strict/non-strict consistent) | same. (The docstrings of A omit the square on `‖Ḟ‖`; the code and paper square it.) |
| 12 | reject: `θ_{k+1}, δ_{k+1} ← θ_k, .5δ_k`; τ unchanged | H:100–103: `x .= iter_hist[iter]`, `δk *= .5`; A:543–545 restores `store.grad` and `grad_val_hist[iter+1] = grad_val_hist[iter]` | none; no extra evaluations on rejection | same |
| 13 | `elseif ‖Ḟ(ψ_j)‖ ≤ τ_low`: accept, `δ` unchanged, `τ_low' = ‖Ḟ(θ_{k+1})‖/√2`, `τ_upp' = √20 τ_low'` | H:104–107: `τ_lower = n/√2`, `τ_upper = √10·n` | `√20·(n/√2) = √10·n`: **same number**; not a mismatch (confirmed numerically, test uses `√20·τ_low`) | same |
| 14 | `elseif ‖Ḟ(ψ_j)‖ ≥ τ_upp`: accept, `δ' = min{1.5δ, δ̄}`, τ updated as in 13 | H:108–112 | none | same |
| 15 | `else`: accept, `δ' = min{1.5δ, δ̄}`, `τ' ← τ_low^k τ_upp^k` (**missing comma**) | H:113–115: `δk = min(1.5δk, δ_upper)`, τ unchanged | paper has a typo; intended `τ' = (τ_low^k, τ_upp^k)` | code |
| 16 | `θ_{k+1} = ψ_j` on acceptance; reference `F(θ_{k+1}) = F(ψ_j)` | A:537–541 (`iter_hist[iter+1] .= x`, `reference_value = Fx`, `grad_val_hist[iter+1] = norm_∇F_ψ`) | none | same |
| 17 | `EstimateLipschitz` at the **terminal** inner iterate (feeds `L̂₀^{k+1}`) | A:408 `update_local_lipschitz_approximation(j, k, …)` with the *final code `j`* | **Off by one when `j_k = 1`.** The function's docstring (A:152–159, 213–223) defines `j == 1` as "no information yet" (returns `1.0`, or the previous estimate). That is correct for the in-loop call at the top of pass 1, but the post-loop call passes the final `j = j_k`, i.e. the number of steps taken. When `j_k = 1` one step was taken, so `ψ_1, ψ_0` and both gradients are available and the paper's `EstimateLipschitz(1,k,ψ_1,ψ_0)` has data, but the code returns `1.0` (k=1) or the stale carried value (k>1) instead of `‖Ḟ(ψ_1)−Ḟ(ψ_0)‖/‖ψ_1−ψ_0‖`. For `j_k ≥ 2` the code is consistent. | **Bug, confirmed against the manuscript** (see "Manuscript check" below). **Fixed 2026-11-08** (Vivak's decision): the post-loop call now passes `j + 1`. |
| 18 | events: ball / lower-gradient / upper-gradient / max-iteration | A:403–406: `!(‖ψ−θ‖ ≤ 10)`, `!(τ_low < n)`, `!(n < τ_upp)`, `!(j < max_iteration)`; A:530 stores `accepted` as flag 5 | ball test hard-codes `10` instead of the `radius` keyword (identical at default); flags are not mutually exclusive (e.g. ball and upper can both be true) | code is consistent at defaults. When tabulating Table 4 decide whether proportions are over flags or over loops (see ANALYSIS.md caveat). |
| 19 | `δ_0 ∈ (0, δ̄]`, `ρ ∈ (0,1)` | constructor asserts `0 < δ0 ≤ δ_upper` (A:100–105); `ρ ∉ (0,1)` is not asserted | cosmetic | n/a (experiment uses `ρ = 1e-4`) |

(Item numbering: 11–15 in the "Known issues" list of the prompt correspond to rows 5, 4, 13 and 15 above; the
`√10` vs `√20` question is row 13 and is **not** a mismatch.)

## How common is the row-17 issue?

Measured with the new `record_inner_loop!` hook on `QLLogisticSin` / `QLLogisticMonomial`
(n = 10, m = 100, 4 random starts each, run to 1e-9 and counted up to the 1e-3 hit):

* inner loops with `j_k = 1`: **23.7%** (V1: 18/76), **22.8%** (V2: 38/167);
* triggering flags in the same sample (loops up to the 1e-3 hit): lower ≈ 72%, upper ≈ 22%,
  ball ≈ 5%, max-iteration 0%; ≈ 85% of outer iterations accepted. (Paper, 1e-3: 64.19 / 30.18 /
  1.36 / 4.34 %, ≈ 12% rejected: same order of magnitude. This is a tiny sample, not a replication.)
* When the loops are run all the way to 1e-9 the late loops are dominated by max-iteration (≈ 80% of
  all loops), as expected near the solution; this is why the tolerance-specific analysis matters.
* Trajectories of "code rule" vs "paper-consistent rule" (transcription, 40 outer iterations, 6 starts)
  **diverge within one or two outer iterations after the first `j_k = 1` loop in all 6 runs**, so the
  difference is not negligible for the paper's results.

## Manuscript check (2026-11-08, by a session with the manuscript source)

`EstimateLipschitz` is the function box Algorithm 2 (`alg:update`, `04-algorithm.tex`, subsection "A Greedy Step Size Strategy"); there is no numbered equation. Cases, with the paper's 0-based `j`:

* `j == 0, k == 0`: `L̂₀⁰ ← 1`.
* `j == 0, k > 0`: `L̂₀ᵏ ← L̂_{j_{k−1}}^{k−1}` (terminal estimate of the previous inner loop).
* `j > 0, θ_k ≠ θ_{k−1}`: `L̂ⱼᵏ ← ‖Ḟ(ψⱼ)−Ḟ(ψ_{j−1})‖ / ‖ψⱼ−ψ_{j−1}‖` ("aggressive").
* `j > 0, θ_k = θ_{k−1}`: `max(that ratio, L̂_{j−1}ᵏ)` ("conservative").

It is defined at `j = 0` without using `ψ_{−1}`, and Algorithm 1 calls it with `j = 0` before the loop and with the incremented `j` after every step. So "no information" is exactly `j = 0`, and after one step (`j = 1`) the one-step ratio is required. The code's in-loop calls use code-`j = paper j + 1` and are right; the post-loop call passes the number of steps taken (the paper's `j`) into a function that expects the code's `j`, so `j_k = 1` is wrong. **Verdict: code has an off-by-one for `j_k = 1`; minimal fix is `j + 1` in the post-loop call (A:408); no change for `j_k ≥ 2`.** No other differences in the five cases (the "rejected" case's `max` with the previous estimate within the same inner loop matches; no safeguard in either).

## Other notes

* The earlier baseline error at `test/methods/gd_non_sequential_armijo_adaptive.jl:551`
  (`BoundsError` in `inner_loop!`, `triggering_event_hist` empty) was a test-setup problem
  fixed in commit `9dd2167` (test-only); no library change.
* The only change made to `gd_non_sequential_armijo_adaptive.jl` in this phase is the new line
  `record_inner_loop!(progData, iter, j_k)` after the inner loop (and keeping `j_k` instead of discarding the
  return value). It is a no-op for ordinary problems and supplies `j_k` to `ours_events.csv`; the
  transcription test confirms the iterates are unchanged.
