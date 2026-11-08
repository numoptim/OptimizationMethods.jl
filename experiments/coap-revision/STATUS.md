# COAP revision experiments: STATUS
Branch: cv/experiments-quasi-likelihood · Last updated: 2026-11-07 (UTC) · Current phase: 1

## Checklist
- [x] Phase 1.1 Julia 1.13 baseline
- [x] Phase 1.2a `TimedProblem` timing layer + `record_iterate!` hook, hooked into `FixedStepGD`, tests in `test/methods/experiment_recorder.jl`
- [x] Phase 1.2b Add `record_iterate!` hook to the other solvers used (BBGD, DIM, LIPAPP, NEST, WEIGHT, OURS, ARM, WATCH, SPEC), each at its own termination test; tests per solver (1e-3 hit == separate run); measure timing overhead (<2%, n=90, m=1000)
- [x] Phase 1.3 Evaluation-budget stopping for every solver used, with tests
- [x] Phase 1.4a ADATR docstring + stub
- [x] Phase 1.4b ADATR implementation
- [x] Phase 1.4c ADATR tests (§2.3)
- [x] Phase 1.4d ASTR1 docstring + stub
- [x] Phase 1.4e ASTR1 implementation
- [x] Phase 1.4f ASTR1 tests (§2.3)
- [x] Phase 1.5 OURS vs Algorithm 1 (`OURS_vs_Algorithm1.md`) (one possible bug found, Open question 1)
- [x] Gate 1: full test suite passes on Julia 1.13
- [ ] Phase 2.1 config/problems.csv, starts.csv, algorithms.csv
- [ ] Phase 2.2 src/run_job.jl
- [ ] Phase 2.3 src/jobs.jl (+ timing pilot, sizing arithmetic)
- [ ] Phase 2.4 chtc/ (.sub files, submit_all.sh, run_job.sh, build_env.sh)
- [ ] Phase 2.5 src/collect.jl
- [ ] Gate 2: local pilot (3 problems x 2 starts x 100 algorithms) + spot checks
- [ ] Phase 3.1 analysis/profiles.jl
- [ ] Phase 3.2 analysis/tables.jl
- [ ] Phase 3.3 results/paper outputs + README
- [ ] Gate 3: analysis end to end on pilot data
- [ ] Phase 4.1 CHTC_RUNBOOK.md
- [ ] Phase 4.2 ANALYSIS.md
- [ ] Phase 4.3 .gitignore + pilot-sized data
- [ ] Gate 4: re-read runbook as newcomer

## Environment notes (cloud container)
- Julia 1.13.3 via juliaup: `export PATH=$HOME/.juliaup/bin:$PATH`. If the container is fresh: install with `curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.13`, then `julia -e 'using Pkg; Pkg.Registry.add("General")'`, then `julia --project -e 'using Pkg; Pkg.instantiate()'`.
- Run tests: `julia --project -e 'using Pkg; Pkg.test()'` (about 1.5 min, fully green at Phase 1.1).

## Open questions for Vivak
1. (Resolved by Vivak, 2026-11-08: FIX) **OURS Lipschitz off-by-one when `j_k = 1`** (`OURS_vs_Algorithm1.md`, row 17). Confirmed against the manuscript (Algorithm 2 `alg:update`: "no information" is exactly paper `j = 0`). Fixed by passing `j + 1` in the post-loop call of `inner_loop!` (src/methods/gd_non_sequential_armijo_adaptive.jl:408); identical for `j_k ≥ 2`. **Consequence for the paper:** OURS numbers will differ from the original submission (about 23% of inner loops had `j_k = 1`), so the revision should say the implementation was corrected to match Algorithm 2.
2. Bibliography (from the prompt): the manuscript's `gratton2022firstorder` cites arXiv:2203.01757 (now "Complexity and performance for two classes of noise-tolerant first-order algorithms"); ASTR1 is arXiv:2203.01647. Not acted on.
3. (Resolved by Vivak, 2026-11-07) The flaky `test/methods/gd_watchdog_barzilai_borwein.jl` (`WatchdogSafeBarzilaiBorweinGD`, unused in the experiments) was removed from `test/test.txt`; the file is left in place. Gate 1 now only needs the remaining suite green.

## Decisions
- `Project.toml` `[compat] julia = "1.13"` set (per prompt). `Manifest.toml` is gitignored, so not committed.
- Per-file baselines were obtained by running each test file in its own process in a scratch env (the stock `Pkg.test()` stops at the first erroring file, `cutest_wrapper.jl`).

- CUTEst test removed from `test/test.txt` (file `test/problems/cutest_wrapper.jl` left in place, just not run), per Vivak: not relevant to the experiments.
- CHTC uses Julia 1.13.2 (Vivak); `[compat] julia = "1.13"` allows both.

- Timing/recorder design (Phase 1.2a): `TimedProblem(inner)` in `src/experiment_recorder.jl` wraps a problem, shares its NLPModels counters, and times `obj`/`grad!`/`objgrad!` with `time_ns()`; solver loops are unchanged except one line `record_iterate!(progData, iter, x, gra_norm)` after the gradient norm used by the termination test (no-op for plain problems). A tolerance is hit when `!(gra_norm > ε)`, the same test the solvers use (NaN counts as hit, matching solver stopping). Wrapper chosen over editing each solver's calls so solver math is untouched.
- `LeastSquares` uses `NLSCounters`; wrapper reads `.counters.neval_*` through that.

- SPEC safeguards (struct has no defaults): `α_lower = 1e-10`, `α_default = 1` (Raydan 1997 style), `window_size = 10`, `δ = 0.5`, `ρ = 1e-4`, 100 line-search iterations.
- OURS threshold test is on outer iterates only; recorder hits follow that.

- Budget stopping (Phase 1.3): `TimedProblem(...; max_grad, max_obj, max_time_ns)`; its `obj`/`grad!`/`objgrad!` throw `BudgetExhausted` *before* the call that would start at `#∇F ≥ max_grad` (resp. `#F ≥ max_obj`, or elapsed ≥ cap), so counts stop exactly at the cap and the stop works mid line search and mid inner loop with no solver edits. `run_solver!(solver, optData, timed)` returns `(x, status)` with status `:finished | :budget_grad | :budget_obj | :timeout`; on exhaustion `x` is the last completed iterate (`timed.last_x`, all-NaN with `last_iter == -1` if none). Consequence: a method at `#F = F_max` that needs no further F (none of ours) would keep taking gradient steps; all F-using methods call F every iteration so this is equivalent to stopping at `#F ≥ F_max`. Harness maps `:finished` to `converged_1e-9` / `nan_or_inf` / etc. from the hits (Phase 2.2). `max_iterations` is set high in the harness so it never binds.

- Papers fetched and read (2026-11-07): ADATR PDF from the DIAL URL (first fetch reset the connection, second worked); ASTR1 arXiv:2203.01647 PDF. Both agree with the prompt summaries: ADATR (2.1)-(2.5) and AdaTrust2 `b̂_k = min{b̂max, max{bmin, b_k/2}}`, `b̂max = ||∇f(x0)||`, `bmin = 1e-4`; ASTR1 Algorithm 2.1, weights (3.1) with μ=1/2 (`θ(ς+Σg²)^{1/2}`, per-coordinate) and (4.1) with μ=ν=1/10 (`θ(k+1)^{1/10} max{ς, max_j|g_ij|}`), ς=1/100, θ as the learning-rate parameter (footnote 4). Termination: the algorithms have no rule; ε-test on `||g_k||` added (ASTR1 comment 4).
- Names: `AdaptiveTrustRegionGD`/`adaptive_trust_region_gd` (ADATR; constructor `init_scaling = b0 = 1/η`, `α`, `b_min`), `ASTR1GD`/`astr1_gd` (`variant ∈ {:adagrad, :maxg}`, `θ = 1/η`, `ς = 0.01`). Update rules are pure functions `adaptive_trust_region_update`, `astr1_adagrad_weights!`, `astr1_maxg_weights!`.
- ADATR: result clamped to `[b_min, b_k]` in addition to the paper's formula, so (2.5) holds even if `b̂max < b_min`. Constructor requires `b0 ≥ b_min` (fine for η ≤ 1e4; grid has η ≥ 1e-4 ⇒ b0 ≤ 1e4, and η ≤ 10 ⇒ b0 ≥ 0.1).
- Test problem scaling (§2.3 item 4): the default `LeastSquares(T, nequ=50, nvar=10)` has L≈85 and ||∇f(x0)||≈150; with b0=1 (η=1) WNGrad's first step overshoots, b jumps to ~7e7 and the method cannot reach 1e-6 in 20,000 iterations (known WNGrad behaviour, not a bug). The convergence tests therefore use `A = randn(50,10)/√50` (eig(A'A) ≈ [0.3, 2.1]), response `A*β + 0.1*noise`, which is "well-conditioned" and passes with η=1 for both new methods. **Consequence for the experiments:** WNGrad-type methods (ADATR, ADAGRAD) are sensitive to b0/θ relative to L; the η grid covers it, but expect poor reliability for small b0 on large-L problems.
- Memory note for Phase 2: all solvers preallocate `iter_hist` with `max_iterations+1` vectors; with `max_iterations` large enough never to bind (~G_max=200,000 for gradient-only methods) and n=90 this is ~144 MB per run. Decide in Phase 2.2/2.3 (cap `max_iterations` per method from G_max, and use the pilot for `request_memory`).

- Phase 1.5: added a hook `record_inner_loop!(progData, k, j_k)` (no-op for plain problems; `TimedProblem.inner_loops` stores `(k, j_k)`) right after `inner_loop!` in `nonsequential_armijo_adaptive_gd`, because `j_k` was discarded and `ours_events.csv` needs it. Behavior of OURS is unchanged (transcription test passes). Event flags come from `triggering_event_hist[k]` = `[ball, lower, upper, maxiter, accepted]`.

- Phase 1.5 fix (Vivak's decision): OURS now follows the manuscript's `EstimateLipschitz` (Algorithm 2) at the terminal inner iterate. Behavior change is limited to inner loops with exactly one step; all other quantities unchanged. The pre-fix behaviour is retained only as a switch in the test transcription (`code_terminal = true`) to show it diverges from the paper after the first `j_k = 1` loop.

## Development plan
Phases 1-4 as in `PROMPT.md` §4. One step per run; a step is "done" when its checks pass and it is committed. Verification: test files listed in the Log; gates push to origin.

## Log
### 2026-11-08: OURS fix and Gate 1
- Manuscript check (session with the manuscript source) confirmed the off-by-one; applied the fix (`j + 1`, with a comment). `test/methods/ours_vs_algorithm1.jl` (396 pass) now requires the solver to equal the paper transcription exactly (iterates, `j_k`, events, acceptance, `δ`, `τ`), plus a direct test that an inner loop of one step sets the estimate to the one-step ratio. Existing OURS test files still pass (inner loop 42, method 82).
- Removed the flaky unused watchdog-BB test from `test/test.txt` (Vivak).
- Full `Pkg.test()` green on Julia 1.13.3. Gate 1 ticked.
- Next: Phase 2.1 (config/problems.csv, starts.csv, algorithms.csv).
### 2026-11-07: Phase 1.5
- Compared `NonsequentialArmijoAdaptiveGD` with Algorithm 1 line by line: `experiments/coap-revision/OURS_vs_Algorithm1.md` (19 rows). Confirmed the known issues: inverted inner `while` (paper typo), missing comma in the `\Else` branch (paper typo), `EstimateLipschitz` at `j=0` undefined in the paper (code defines it as carry-over), `√20·τ_low = √10·‖g‖` so no mismatch. Found one possible real bug (Open question 1).
- New test `test/methods/ours_vs_algorithm1.jl` (392 pass): an independent transcription of Algorithm 1 reproduces the solver (iterates, `j_k`, event flags, acceptance, `δ`, `τ`) over 40 outer iterations from 3 starts; the paper-consistent variant of the Lipschitz rule is identical until the first `j_k = 1` loop and diverges right after.
- Full suite: only the unrelated flaky `gd_watchdog_barzilai_borwein.jl` failed (Open question 3); everything else passes (27,561 to 27,570 passed, 1 failed in two runs).
- Stopped here per §4 (real bug found, so stop and ask). Gate 1 not ticked. Next after Vivak's answer: Gate 1, then Phase 2.1.
### 2026-11-07: Phase 1.4 (all six sub-steps, at Vivak's request in one run)
- 1.4a/1.4d: docstrings + stubs committed first (`src/methods/gd_adaptive_trust_region.jl`, `src/methods/gd_astr1.jl`), then implementations (1.4b/1.4e), then tests (1.4c/1.4f), separate commits.
- Tests (`test/methods/gd_adaptive_trust_region.jl`: 10,138 pass; `test/methods/gd_astr1.jl`: 5,523 pass), wired in `test/test.txt`: hand-computed three iterations for both variants of each method (rtol 1e-12, plain arithmetic in the test); invariants (ADATR: `b_k ≥ b_min`, `||d_k|| = Δ_k`, α=0 ⇒ δ_k non-increasing; ASTR1: `|s_ik| ≤ Δ_ik` with weights recomputed independently from stored iterates, adagrad `w` non-decreasing); `neval_obj == 0` and `neval_grad == stop_iteration + 1` on `LeastSquares`; convergence to 1e-6 within 20,000 iterations for η=1; start at stationary point stops at iteration 0; `max_iterations` respected.
- The new methods were also added to the recorder/budget test matrix (`experiment_recorder_solvers.jl`, now 658 pass).
- Full `Pkg.test()` green on Julia 1.13.3.
- Next: Phase 1.5 (OURS vs Algorithm 1), then Gate 1.
### 2026-11-07: Phase 1.3
- Implemented budgets as above. Tests (`test/methods/experiment_recorder_solvers.jl`, now 486 pass): for all 10 solver configs, a non-binding budget leaves the result unchanged; `max_grad` in {1, 2, 3, half} gives exactly that many gradients, status `budget_grad`, finite iterate equal to the solver's `iter_hist` entry at `last_iter`; `max_obj` in {1, 2, half} for ARM, WATCH, SPEC, OURS gives exactly that many F evaluations (mid line-search / inner loop); the F-free methods are asserted to use 0 F; a 1 ns wall cap gives `timeout`.
- Full `Pkg.test()` green on Julia 1.13.3.
- Next: Phase 1.4a (ADATR docstring + stub).
### 2026-11-07: Phase 1.2b (partial)
- Hooked BBGD, DIM, LIPAPP, NEST, WEIGHT, ARM, WATCH, SPEC, OURS: one call at iteration 0 (immediately before the first loop test) and one at the end of each loop body. New test `test/methods/experiment_recorder_solvers.jl`: 170 pass (hit == separate run at that threshold: #F, #grad, iteration, iterate; wrapper result identical).
- OURS tests the threshold only on outer iterates (`grad_val_hist[iter+1]`), never on inner iterates; hits are recorded there only.
- SPEC struct has no safeguard defaults; use Raydan-style `α_lower = 1e-10`, `α_default = 1` in Phase 2.1 (record in Decisions then).
- Overhead (n=90, m=1000, all four V; `src/overhead_check.jl`, best of 7): FIXED -9.3% to +5.0%, ARM -0.9% to +0.5%. Runs are noise-dominated (differences of both signs). Analytic bound: wrapper adds two `time_ns()` calls (~50 ns) per F/grad call vs ~0.3 ms per gradient, i.e. ~0.015% < 2%. Conclusion: overhead is under 2%.
- Full `Pkg.test()` on Julia 1.13.3: all tests pass, including the two new recorder files.
- Next: Phase 1.3 (evaluation-budget stopping).
### 2026-11-07: Phase 1.2a
- Added `src/experiment_recorder.jl` (`TimedProblem`, `ToleranceHit`, `start!`, `record_iterate!`), exported; hooked `fixed_step_gd` (iteration 0 and each iteration).
- New test `test/methods/experiment_recorder.jl` (wired in `test/test.txt`): 58 pass. Checks hits ordered, counts/times non-decreasing, each of the four hits equals a separate run at that threshold (iteration, #F, #grad, iterate), wrapper result identical to plain run, counters shared. `test/methods/gd_fixed.jl` still passes.
- Next: Phase 1.2b (hook the other solvers; overhead measurement).
### 2026-11-07: Phase 1.1 baseline
- Installed juliaup, Julia 1.13.3. Added the General registry manually (`Pkg.Registry.add("General")`), as none was installed in the cloud image. Set compat `julia = "1.13"`, resolved and instantiated.
- `Pkg.test()` stock run: 1836 pass, 4 fail, 1 error, and aborts at `problems/cutest_wrapper.jl` (CUTEst not installed), so method tests never ran.
- Per-file baseline (each file separately, Julia 1.13.3). All pass except:
  - `problems/quasi_likelihood.jl`: 448 pass, **4 fail** (line 301, Float16 `obs_obs_t` sum vs `design'*design`, atol too tight for Float16).
  - `problems/cutest_wrapper.jl`: **error**, `Package CUTEst not found`.
  - `methods/gd_non_sequential_armijo_adaptive.jl` (OURS): 304 pass, **1 error** at test line 551, `BoundsError` in `inner_loop!` (src/methods/gd_non_sequential_armijo_adaptive.jl:403, access to a 0-element `Vector{Vector{Int64}}`, likely `triggering_event_hist` unset in that test). Relevant to Phase 1.5; investigate there, don't change OURS behavior.
  - `methods/gd_non_sequential_armijo_fixed_mnewton.jl`: 740 pass, **6 fail**.
  - Everything else passes (problems: least_squares 220, logistic 257, poisson 694, ql_sin 42, ql_centered_exp 56, ql_centered_log 63, ql_monomial 56; all other method and helper files pass).
- Next: Phase 1.2 (timing layer and tolerance first-hit recorder).

### 2026-11-07: test suite rerun without CUTEst
- Removed `problems/cutest_wrapper.jl` from `test/test.txt`. Stock `Pkg.test()` now runs all files: **11438 pass, 10 fail/error**, all pre-existing and unchanged: `quasi_likelihood.jl` 4 fails (line 301, Float16 tolerance); `gd_non_sequential_armijo_fixed_mnewton.jl` 6 fails (lines 1653, 1668, 1672, 1821, 1836); `gd_non_sequential_armijo_adaptive.jl` 1 error (line 551, BoundsError, OURS; see Phase 1.5).
- Gate 1 therefore needs these to be either unrelated-and-listed or fixed; decision at Gate 1: they are unrelated to experiments (not run by the harness), so they will be listed rather than fixed, except the OURS error, which is investigated in Phase 1.5.
- Next: Phase 1.2.

### 2026-11-07: test fixes (Vivak's go-ahead)
- `test/problems/quasi_likelihood.jl`: dropped `Float16` from `real_types` (3 places); the 4 failures are gone.
- OURS `BoundsError` was a **test bug, not a source bug**. In `test/methods/gd_non_sequential_armijo_adaptive.jl` (the "method" testset, 4 call sites) `inner_loop!(..., k)` is run on `optDatakm1`, built with `max_iterations = k-1`, so its `triggering_event_hist` has only k-1 entries and `[k]` is out of bounds. It only fired when the random run had a rejected first iterate (data-dependent, so flaky). Fix: `push!(optDatakm1.triggering_event_hist, zeros(Int64, 5))` before each call. `src/` unchanged. The solver sizes the history to `max_iterations` and indexes `[iter]`, so it is consistent.
- Note for Phase 1.5: `triggering_event_hist[k]` is length 5; entries 1-4 are the events (ball, lower, upper, max-iteration; `true` means that event ended the inner loop) and entry 5 is `achieved_descent` (written in the solver). Entries 1-4 are set inside `inner_loop!` from the *final* ψ_j, so several can be 1 at once.
- 3 full `Pkg.test()` runs: only failures are the 6 in `gd_non_sequential_armijo_fixed_mnewton.jl` (ignored per Vivak; method not used in experiments). Total pass count varies run to run (11553-11587) because tests use random sizes.
- Gate 1 criterion therefore: full `Pkg.test()` passes (see next entry).

### 2026-11-07: removed mnewton-fixed test from suite
- Per Vivak, removed `methods/gd_non_sequential_armijo_fixed_mnewton.jl` from `test/test.txt` (file kept on disk, not run; method not used in experiments).
- Full `Pkg.test()` on Julia 1.13.3: **10809 pass, 0 fail, 0 error, rc=0**. The suite is fully green, so the Gate 1 baseline is "all tests pass".
- Next: Phase 1.2.
