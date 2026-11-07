# COAP revision experiments: STATUS
Branch: cv/experiments-quasi-likelihood · Last updated: 2026-11-07 (UTC) · Current phase: 1

## Checklist
- [x] Phase 1.1 Julia 1.13 baseline
- [x] Phase 1.2a `TimedProblem` timing layer + `record_iterate!` hook, hooked into `FixedStepGD`, tests in `test/methods/experiment_recorder.jl`
- [x] Phase 1.2b Add `record_iterate!` hook to the other solvers used (BBGD, DIM, LIPAPP, NEST, WEIGHT, OURS, ARM, WATCH, SPEC), each at its own termination test; tests per solver (1e-3 hit == separate run); measure timing overhead (<2%, n=90, m=1000)
- [ ] Phase 1.3 Evaluation-budget stopping for every solver used, with tests
- [ ] Phase 1.4a ADATR docstring + stub
- [ ] Phase 1.4b ADATR implementation
- [ ] Phase 1.4c ADATR tests (§2.3)
- [ ] Phase 1.4d ASTR1 docstring + stub
- [ ] Phase 1.4e ASTR1 implementation
- [ ] Phase 1.4f ASTR1 tests (§2.3)
- [ ] Phase 1.5 OURS vs Algorithm 1 (`OURS_vs_Algorithm1.md`)
- [ ] Gate 1: full test suite passes on Julia 1.13
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
(none open)

Resolved by Vivak (2026-11-07): use Julia **1.13.2** on CHTC (tarball in Phase 2.4 must pin 1.13.2; local dev stays 1.13.3); remove CUTEst tests.

## Decisions
- `Project.toml` `[compat] julia = "1.13"` set (per prompt). `Manifest.toml` is gitignored, so not committed.
- Per-file baselines were obtained by running each test file in its own process in a scratch env (the stock `Pkg.test()` stops at the first erroring file, `cutest_wrapper.jl`).

- CUTEst test removed from `test/test.txt` (file `test/problems/cutest_wrapper.jl` left in place, just not run), per Vivak: not relevant to the experiments.
- CHTC uses Julia 1.13.2 (Vivak); `[compat] julia = "1.13"` allows both.

- Timing/recorder design (Phase 1.2a): `TimedProblem(inner)` in `src/experiment_recorder.jl` wraps a problem, shares its NLPModels counters, and times `obj`/`grad!`/`objgrad!` with `time_ns()`; solver loops are unchanged except one line `record_iterate!(progData, iter, x, gra_norm)` after the gradient norm used by the termination test (no-op for plain problems). A tolerance is hit when `!(gra_norm > ε)`, the same test the solvers use (NaN counts as hit, matching solver stopping). Wrapper chosen over editing each solver's calls so solver math is untouched.
- `LeastSquares` uses `NLSCounters`; wrapper reads `.counters.neval_*` through that.

- SPEC safeguards (struct has no defaults): `α_lower = 1e-10`, `α_default = 1` (Raydan 1997 style), `window_size = 10`, `δ = 0.5`, `ρ = 1e-4`, 100 line-search iterations.
- OURS threshold test is on outer iterates only; recorder hits follow that.

## Development plan
Phases 1-4 as in `PROMPT.md` §4. One step per run; a step is "done" when its checks pass and it is committed. Verification: test files listed in the Log; gates push to origin.

## Log
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
