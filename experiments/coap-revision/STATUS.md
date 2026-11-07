# COAP revision experiments: STATUS
Branch: cv/experiments-quasi-likelihood · Last updated: 2026-11-07 (UTC) · Current phase: 1

## Checklist
- [x] Phase 1.1 Julia 1.13 baseline
- [ ] Phase 1.2 Timing layer and tolerance first-hit recorder (FixedStepGD + LeastSquares tests)
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

## Open questions for Vivak
1. Julia version: `juliaup` installed 1.13.3 (latest 1.13 patch available now; the prompt said 1.13.2). Chose 1.13.3 locally. CHTC tarball (Phase 2.4) will use 1.13.3 too unless you prefer 1.13.2.
2. Pre-existing failures (not fixed, see Log): `test/problems/cutest_wrapper.jl` needs the CUTEst package (not a test dep); this aborts the whole `Pkg.test()` run, so Gate 1 ("full suite passes") cannot literally be met unless that line is removed from `test/test.txt` on this branch. Plan: for Gate 1, run with `cutest_wrapper.jl` excluded and say so. Tell me if you want otherwise.

## Decisions
- `Project.toml` `[compat] julia = "1.13"` set (per prompt). `Manifest.toml` is gitignored, so not committed.
- Per-file baselines were obtained by running each test file in its own process in a scratch env (the stock `Pkg.test()` stops at the first erroring file, `cutest_wrapper.jl`).

## Development plan
Phases 1-4 as in `PROMPT.md` §4. One step per run; a step is "done" when its checks pass and it is committed. Verification: test files listed in the Log; gates push to origin.

## Log
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
