ALL DONE
# COAP revision experiments: STATUS
Branch: cv/experiments-quasi-likelihood · Last updated: 2026-11-08 (UTC) · Current phase: done (Gates 1-4 passed; CHTC steps still untested on CHTC itself)

## For Vivak
- **Branch:** `cv/experiments-quasi-likelihood` (isolated, not to be merged into `main`).
- **First three CHTC commands** (full copy-paste sequence in [`CHTC_RUNBOOK.md`](CHTC_RUNBOOK.md); run from `~/OptimizationMethods.jl/experiments/coap-revision` after step 1):
  1. `git clone --branch cv/experiments-quasi-likelihood --single-branch https://github.com/numoptim/OptimizationMethods.jl.git`
  2. `bash chtc/make_tarballs.sh`
  3. `condor_submit -i chtc/build.sub`   (then `bash build_env.sh` inside the job, `exit`)
- **Size:** 120,000 runs in **535 jobs** (one submit file). Worst-case job time **max 19.6 h, median 14.3 h** (upper bounds from the timing pilot; most runs stop earlier; the CHTC pilot, runbook step 3, measures the real values). Worst-case total 6,600 core-hours. 3000 MB / 4 GB per job.
- **Open questions** (details in the section below): (2) bibliography entry `gratton2022firstorder` in the manuscript; (4) V3 sign: resolved, please fix the manuscript to `exp(−|μ−1|^4.5)`; (5) things that can only be confirmed on CHTC (CPU level, depot size, memory).
- **Decisions of yours reflected in the code:** OURS terminal Lipschitz estimate fixed to match Algorithm 2 (OURS results will differ from the original submission); CUTEst and the flaky watchdog-BB tests removed from the suite.
- **OURS vs Algorithm 1:** [`OURS_vs_Algorithm1.md`](OURS_vs_Algorithm1.md). After the CHTC run, give `results/merged/` to a fresh session with [`ANALYSIS.md`](ANALYSIS.md).
- **Not tested here (no HTCondor in this container):** the `.sub` files and the build/run scripts on CHTC itself. Everything else was tested locally: the runner and the collector on a 600-run pilot, the shell plumbing of `run_job.sh` with stub tarballs, the `submit_all.sh` skip logic, the runtime summarizer, and the docs' paths (`python3 test/check_docs.py`). The CHTC pilot in the runbook is the real test.

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
- [x] Phase 2.1 config/problems.csv, starts.csv, algorithms.csv
- [x] Phase 2.2 src/run_job.jl
- [x] Phase 2.3 src/jobs.jl (+ timing pilot, sizing arithmetic)
- [x] Phase 2.4 chtc/ (.sub files, submit_all.sh, run_job.sh, build_env.sh)
- [x] Phase 2.5 src/collect.jl
- [x] Gate 2: local pilot (3 problems x 2 starts x 100 algorithms) + spot checks
- [x] Phase 3.1 analysis/profiles.jl
- [x] Phase 3.2 analysis/tables.jl
- [x] Phase 3.3 results/paper outputs + README
- [x] Gate 3: analysis end to end on pilot data
- [x] Phase 4.1 CHTC_RUNBOOK.md
- [x] Phase 4.2 ANALYSIS.md
- [x] Phase 4.3 .gitignore + pilot-sized data
- [x] Gate 4: re-read runbook as newcomer

## Environment notes (cloud container)
- Julia 1.13.3 via juliaup: `export PATH=$HOME/.juliaup/bin:$PATH`. If the container is fresh: install with `curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.13`, then `julia -e 'using Pkg; Pkg.Registry.add("General")'`, then `julia --project -e 'using Pkg; Pkg.instantiate()'`.
- Run tests: `julia --project -e 'using Pkg; Pkg.test()'` (about 1.5 min, fully green at Phase 1.1).

## Open questions for Vivak
1. (Resolved by Vivak, 2026-11-08: FIX) **OURS Lipschitz off-by-one when `j_k = 1`** (`OURS_vs_Algorithm1.md`, row 17). Confirmed against the manuscript (Algorithm 2 `alg:update`: "no information" is exactly paper `j = 0`). Fixed by passing `j + 1` in the post-loop call of `inner_loop!` (src/methods/gd_non_sequential_armijo_adaptive.jl:408); identical for `j_k ≥ 2`. **Consequence for the paper:** OURS numbers will differ from the original submission (about 23% of inner loops had `j_k = 1`), so the revision should say the implementation was corrected to match Algorithm 2.
2. Bibliography (from the prompt): the manuscript's `gratton2022firstorder` cites arXiv:2203.01757 (now "Complexity and performance for two classes of noise-tolerant first-order algorithms"); ASTR1 is arXiv:2203.01647. Not acted on.
3. (Resolved by Vivak, 2026-11-07) The flaky `test/methods/gd_watchdog_barzilai_borwein.jl` (`WatchdogSafeBarzilaiBorweinGD`, unused in the experiments) was removed from `test/test.txt`; the file is left in place. Gate 1 now only needs the remaining suite green.

4. (Resolved by Vivak, 2026-11-08) **V3 sign:** the code's V3(μ) = exp(−|μ−1|^4.5) is correct; Vivak will correct the manuscript (Appendix A / §7.2.1) to the minus sign. No code change. (Why it is self-consistent: V3 ≤ 1 on (0,1), so the generator's `vmax = 1.0` noise bound is valid.)

5. **To confirm on CHTC (runbook step 3):** (a) that `Microarch == "x86_64-v3"` has many slots (otherwise switch the level with the one `sed` command in the runbook); (b) the depot tarball size (above 100 MB CHTC must give SQUID access); (c) memory and disk use versus the requests (3000 MB, 4 GB) and the real job runtimes (worst case 19.6 h). Chosen from local measurements and CHTC's Julia guide; none could be tested on CHTC from here.

## Decisions
- `Project.toml` `[compat] julia = "1.13"` set (per prompt). `Manifest.toml` is gitignored, so not committed.
- Per-file baselines were obtained by running each test file in its own process in a scratch env (the stock `Pkg.test()` stops at the first erroring file, `cutest_wrapper.jl`).

- CUTEst test removed from `test/test.txt` (file `test/problems/cutest_wrapper.jl` left in place, just not run), per Vivak: not relevant to the experiments.
- ~~CHTC uses Julia 1.13.2 (Vivak)~~ **Superseded 2026-11-08 (Vivak): CHTC now uses Julia 1.13.3**, the version everything was developed and tested on (local = CHTC). `chtc/make_tarballs.sh`, `build.sub`, `build_env.sh`, `run_job.sh`, `collect.sh`, `submit_all.sh`, the `.sub` files (via `src/jobs.jl`) and the runbook were updated; `config/jobs.csv` is unchanged. `[compat] julia = "1.13"` allows any 1.13.

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

- Phase 2.1 data generation: the branch's constructors already generate data as Appendix A: x_i ~ U[0,1]^n, θ* ~ U[logit(0.3)/n, logit(0.7)/n], ε ~ U[−0.3/√vmax, 0.3/√vmax], y = μ + V(μ)^{1/2} ε with μ = g(xᵀθ*), and `get_design` is the first RNG draw. Differences from Appendix A: (i) V3 sign, see Open question 4; (ii) the noise bounds use fixed `vmax` = 2.3 (V1), 2.0 (V2), 1.0 (V3), 2.8 (V4), valid but looser than max V on [0.3,0.7] (≈2.26, 1.20, 1.00, 1.18), so for V2 and V4 the noise is smaller than "Ṽ ≈ max" would give; kept as is. (iii) The problem structs do not store θ*, so `true_design` regenerates it from the same seed (verified equal to the stored design for all 120 problems).
- Seeds: data seed of problem p = 1,000,000 + p (set with `Random.seed!` immediately before construction; constructors use the global RNG); start (p, s) seed = 2,000,000 + 100p + s, generated with `Random.Xoshiro(seed)` as 20·rand − 10. `problems.csv` also stores design/response checksums (sum and sum of squares) so `run_job.jl` can detect RNG drift between Julia 1.13.2 (CHTC) and 1.13.3 (local).
- Design check results (recorded in `problems.csv` columns `checks_*`, all 120 problems): y_i ∈ (0,1) ✓ (range 0.149–0.850); g(x_iᵀθ*) ∈ (0.3,0.7) ✓ (range 0.430–0.584); |V^{1/2}ε| < 0.3 ✓ (max 0.296, V3).
- Algorithm table: ids 1–100 in order FIXED (1–7), DIM (8–14), BBGD (15–28; long then short), LIPAPP (29–35), NEST (36–42), WEIGHT (43–49), OURS (50), ARM (51), WATCH (52–58), SPEC (59–72; long then short), ADATR (73–86; α=0 then 0.9), ASTR1 (87–100; adagrad then maxg); η in the order 1e-4, 1, 2, 4, 6, 8, 10. Hyperparameter columns for runs.csv: `eta, long_stepsize, adatr_alpha, astr1_variant`.
- `experiments/coap-revision/Project.toml` uses `[sources]` to dev the package at `../..`; its `Manifest.toml` is gitignored (decide in Phase 2.4 whether CHTC needs a committed manifest).

- **Run statuses.** The prompt's six plus one: `stalled` = the solver stopped by itself without reaching 1e-9 (failed line search or `max_iterations`), distinct from `nan_or_inf`. `converged_1e-9` iff the 1e-9 hit was recorded; `nan_or_inf` if the final gradient norm or iterate is non-finite. A NaN gradient norm is never recorded as a hit (the solvers' own test `!(norm > ε)` would count NaN as converged); recorder changed to `gra_norm <= ε`.
- **Budgets/iterations.** `max_iterations = G_max + 10` (never binds: every iteration of every method uses at least one gradient). `G_max = F_max = 200,000` for the full experiment; the Gate-2 pilot used 1,000 for both (a full-budget local pilot would take hours on the n=90, m=1000 problems).
- **Sizing arithmetic (timing pilot, `results/pilot/eval_times.csv`, 1,000 evals each).** t_F from 0.43 ms (n=10, m=100) to 14.7 ms (V1, n=90, m=1000); t_grad from 4 µs to 0.19 ms; c = t_F/t_grad from 30 to 277. Worst-case run time T_run = F_max t_F + G_max t_grad: gradient-only methods <= 38 s per run (largest problem), F-using methods <= about 50 min (V1, n=90, m=1000). Jobs are tiles (algorithms x starts) of one problem and one kind (gradient-only: 77 algorithms; F-using: 23 algorithms), sized so that the sum of T_run <= 20 h (limit 24 h; 72 h hold). Result: **535 jobs** (120 gradient-only jobs of 770 runs, 415 F-using jobs of 23-230 runs), worst-case job time max 19.6 h, median 14.3 h; one submit file (<= 10,000), 535 < 1,000 queued, so the 5-minute mean-runtime rule does not apply. Per-run safety cap = max(30 s, 2 T_run). Worst-case total is 6,595 core-hours (6,240 for F-using methods); typical runs stop earlier, so the real total will be lower. G_max/F_max were not lowered.
- **Resources.** Peak RSS 1,525 MB for the largest problem at G_max = 200,000 (histories preallocated): `request_memory = 2400MB` (+50% rounded). `request_disk = 4GB` (Julia ~1 GB unpacked, depot < 0.5 GB, tarballs, output). `request_cpus = 1`, `JULIA_NUM_THREADS=1`. These are estimates to be confirmed by the CHTC pilot (runbook step 3). **Raised to 3000MB after the CHTC pilot peaked at 2198 MB (only 8% below 2400MB; memory overruns put jobs on hold).**
- **CPU pinning.** `requirements = (Microarch == "x86_64-v3")` (HTCondor machine attribute `Microarch`; levels x86_64-v1..v4). Check availability with `condor_status -af Microarch | sort | uniq -c` and `condor_status -constraint 'Microarch == "x86_64-v3"' -af CpuModel | sort | uniq -c`. The CPU model of every run is recorded in `runs.csv` (`cpu_model`), so homogeneity can be audited afterwards. If v3 is scarce, change `MICROARCH` in `src/jobs.jl` and regenerate.
- **CHTC environment.** Following CHTC's Julia guide (portable binary + portable depot, built in an interactive build job): `chtc/make_tarballs.sh` (Julia 1.13.3 tarball + `git archive` code tarball), `chtc/build.sub` + `chtc/build_env.sh` (instantiate + precompile into `depot.tar.gz`), `chtc/run_job.sh` (unpacks and runs one job; output `job_<id>` is remapped to `results/raw/job_<id>` on the submit node). The experiments `Manifest.toml` is committed (`git add -f`) so the CHTC depot uses the exact versions tested here (resolved with Julia 1.13.3, the same version CHTC uses). The depot tarball may exceed 100 MB; then CHTC asks to contact chtc@cs.wisc.edu for SQUID access (documented in the runbook). The package is precompiled in the depot but, because it is a path dependency extracted to a per-job directory, Julia may recompile it once per job (about 1 min).
- **Not tested here:** the `chtc/` scripts and `.sub` files cannot be run in this container (no HTCondor). They were checked with `bash -n`; the real test is the CHTC pilot (runbook step 3), and Gate 4 re-reads them.
- **Job output.** `run_job.jl` refuses to overwrite an existing job directory; it writes `.job_<id>.tmp` and renames at the end; errors per run are caught (status `error`, message in `errors.log`).

- **Analysis environment.** The analysis has its own environment `experiments/coap-revision/analysis/` (CSV, DataFrames, Statistics, CairoMakie), so the CHTC depot does not carry plotting libraries. Run: `julia --project=analysis src/analysis/run_analysis.jl [--merged DIR] [--out DIR] [--no-plots]` (defaults `results/merged` -> `results/paper`; about 1 min). Code: `src/analysis/{common,profiles,tables,plots,run_analysis}.jl`; tests: `julia --project=. test/test_analysis.jl` (hand-computed synthetic cases; the main environment suffices because the tests do not plot).
- **Analysis definitions** (also in `results/paper/README.md`, generated): reliable at ε = hit and F(x_hit) < F(x0). Best configuration of a method at ε: highest reliability, ties by smaller median #∇F of reliable runs, then smaller id; chosen per tolerance. Performance profiles: cost of a solver on a pair is the metric at the first ε hit if reliable, else ∞; pairs that no solver of the set solves are dropped (reported in `profile_summary.csv`); for #F the metric is #F + 1 because gradient-only methods have #F = 0 and the ratio would be undefined. Weighted cost = #F·c + #∇F with c per run = (time_obj_s/n_obj)/(time_grad_s/n_grad) (c = 0 when n_obj = 0). 95% margins: 1.96 sd/√n, unclamped, 0 when n < 2. Events (Table 4): per OURS run, proportions of inner loops with each flag over the outer iterations up to the first ε hit, averaged over runs; rejection rate = share of those outer iterations not accepted. Flags are not mutually exclusive (they nearly are in practice).
- **Outputs.** `results/paper/`: per tolerance and metric Appendix-D tables `reliability_<ε>_<metric>.csv` (16), `profile_{main,all}_<metric>_<ε>.csv` (32) with wide pgfplots columns `tau,<method>...`, `events_<ε>.csv` (4), `all_algorithms_reliability.csv`, `algorithms_table.csv`, `best_configurations.csv`, `profile_summary.csv`, `success_summary.csv`, 10 PNG previews and a generated `README.md` mapping files to manuscript items (Appendix E). The OURS curve is drawn thick black in the PNGs.
- Pilot-sized data committed: `results/pilot_merged/` (600 runs) and `results/pilot_paper/` (analysis of it). Real outputs go to the gitignored `results/merged/` and `results/paper/` (commit `results/paper/` by hand after the real run if wanted).

- **Phase 4.** `CHTC_RUNBOOK.md` (steps 1-7 as in the prompt, with helper scripts `chtc/collect.sh`, `resubmit.sh`, `job_runtimes.sh`; the submit node has no Julia, so `collect.sh` uses the portable Julia + depot from `$HOME/coap-julia`); `submit_all.sh` skips jobs that already have output (the 14 pilot jobs are real jobs of the list and are kept); `ANALYSIS.md` for a fresh session; `src/analysis/check_merged.jl` (structure check of the merged CSVs); `test/check_docs.py` (every path and script in the two docs exists; Gate 4). `.gitignore` covers `results/raw/`, `results/merged/`, `results/pilot_raw/`, `results/pilot_logs/`, `results/raw_invalid/`, `chtc/*.tar.gz`, `chtc/logs/` and the generated submit lists; pilot-sized data (`results/pilot_merged/`, `results/pilot_paper/`, `results/pilot/eval_times.csv`) is committed. The Manifest.toml files of the experiments and analysis environments are committed (`git add -f`).

## Development plan
Phases 1-4 as in `PROMPT.md` §4. One step per run; a step is "done" when its checks pass and it is committed. Verification: test files listed in the Log; gates push to origin.

## Log
### 2026-11-13: collect.jl out-of-memory fix
- `chtc/collect.sh` on the full results was killed (OOM): the CHTC pilot had 2.7M OURS inner-loop rows for ~30 OURS runs, so the full experiment has ~100M, and `collect.jl` held them all in memory. `collect.jl` now streams `ours_events.csv` job by job (merged file byte-identical to the old output on the local pilot) and writes `ours_events_summary.csv` (per OURS run and tolerance counts over the loops up to the hit); the analysis reads the summary (`events_table(summary, tol)`), not the big file. Checked on the local pilot: merged files identical, `events_<ε>.csv` equal up to 1e-14 rounding, analysis tests and `check_merged.jl` pass.
### 2026-11-08: Phase 4 and Gate 4
- 4.1 `CHTC_RUNBOOK.md`; helper scripts `chtc/collect.sh`, `resubmit.sh`, `job_runtimes.sh` (tested on synthetic `condor_history` lines: passes at 20 h, fails with a clear message at 27.8 h); `submit_all.sh` rewritten to skip finished jobs (tested with a fake results directory: 533 of 535 queued when 2 are done); `build.sub` now lists `transfer_output_files = depot.tar.gz`.
- 4.2 `ANALYSIS.md` and `src/analysis/check_merged.jl` (8 checks pass on the pilot data).
- 4.3 `.gitignore`, pilot-sized data committed.
- **Gate 4:** `python3 test/check_docs.py` (all paths and scripts mentioned in the two docs exist; a negative test confirmed it fails on a bogus path); `bash -n` on all scripts; `run_job.sh` plumbing tested with stub tarballs and a fake `julia` (the job directory appears at the top level of the scratch directory, where `transfer_output_files` expects it); read-through of the runbook as a newcomer: each step names its commands, the expected output and what to do on failure. Placeholders left for Vivak: `<NETID>`, `<SUBMIT_HOST>`, `<CLUSTER_ID>`.
- All checklist items done: `ALL DONE`.
### 2026-11-08: Phase 3 and Gate 3
- Implemented the analysis (3.1 profiles, 3.2 tables, 3.3 outputs, PNG previews, generated README). Unit tests `test/test_analysis.jl` (5 test sets: weighted cost and join, best configuration with tie-break, profile ratios and curves with hand-computed values, Appendix-D columns and margins, event proportions up to the hit) pass.
- **Gate 3:** `run_analysis.jl --merged results/pilot_merged --out results/pilot_paper` runs end to end (67 files, about 1 min). PNGs inspected: profiles and reliability bars render correctly. With only 6 problem-start pairs the pilot margins are wide, so the numbers are a smoke test, not results.
- On the pilot (budget 1,000), OURS' inner-loop event mix up to the 1e-3 hit is ball 2.8%, lower 75.8%, upper 21.4%, max-iteration 0%, rejection 12.1% (2 runs, 41 loops): same ballpark as the paper (1.36 / 64.19 / 30.18 / 4.34 %, about 12% rejected), not a replication.
- Next: Phase 4.1 (`CHTC_RUNBOOK.md`), 4.2 (`ANALYSIS.md`), 4.3 (`.gitignore` and pilot data), Gate 4.
### 2026-11-08: Phase 2.2-2.5 and Gate 2
- 2.2: `src/runner.jl` (algorithm factory, `run_one`, uncounted F evaluations at x0/final/hit iterates with counters restored) and `src/run_job.jl`. Smoke test (n=10, m=100, all 100 algorithms, budget 300): no errors, budgets respected exactly, F-count 0 for gradient-only methods.
- 2.3: `src/timing_pilot.jl` (results/pilot/eval_times.csv), `src/jobs.jl` -> `config/jobs.csv` (535 jobs, 120,000 runs) and `config/pilot_jobs.csv` (6 jobs: V1 n=10 m=100, V3 n=50 m=1000, V2 n=90 m=1000; 2 starts; 100 algorithms; budget 1,000). Sizing in Decisions.
- 2.4: `chtc/` files, `chtc/jobs_1.sub` (535 jobs), `chtc/pilot.sub` + `chtc/pilot_jobs.txt` (14 jobs covering the smallest, a middle and the largest problem; worst case 15.2 h), `submit_all.sh` (`--dry-run` supported).
- 2.5: `src/collect.jl [--pilot] [--jobs F] [--raw D] [--out D]`: exact counts, one row per (problem, start, algorithm), exactly 4 tolerance rows per run, allowed statuses, no NaN/Inf/missing in counts and times, hit data monotone across tolerances, #F = 0 for FIXED/DIM/BBGD/LIPAPP/NEST/WEIGHT/ADATR/ASTR1, budgets respected, status vs 1e-9 hit consistent; writes `resubmit_jobs.txt`; exits nonzero on failure. `test/test_collect.jl` (8 checks: clean data passes; missing file, duplicate row, bad status, NaN time, F evaluations for FIXED, non-monotone hits, status/hit mismatch all fail).
- **Gate 2 (local pilot):** `run_job.jl` on the 6 pilot jobs (600 runs, 0 errors, 4 jobs in parallel, about 4 min); `collect.jl --pilot` -> VALID. Statuses: 303 `converged_1e-9`, 259 `budget_grad`, 38 `budget_obj`, no `error`/`timeout`/`nan_or_inf`/`stalled`. Pooled reliability (all 100 algorithms): 1e-3: 0.535, 1e-5: 0.463, 1e-7: 0.438, 1e-9: 0.423 (budget 1,000, so not comparable with the paper). Spot checks (`src/spot_check.jl`): OURS, ARM and ADATR re-run by calling the solvers directly with threshold 1e-3 reproduce the recorded 1e-3 hit exactly (iteration, #F, #grad, F at the hit iterate).
- Full OptimizationMethods test suite still green (library change: NaN rule in `record_iterate!`, test added).
- Next: Phase 3.1 (`src/analysis/profiles.jl`), developed on the pilot data in `results/pilot_merged/`.
### 2026-11-08: Phase 2.1
- Added `experiments/coap-revision/{Project.toml, src/instances.jl, src/make_config.jl, test/test_config.jl, config/{problems,starts,algorithms}.csv}`. `julia --project=. src/make_config.jl` regenerates the three CSVs; `julia --project=. test/test_config.jl` (405 checks) validates counts (120 problems, 1,200 starts, 100 algorithms = 58+14+14+14, 120,000 runs), uniqueness of seeds, the paper-design checks, and that regenerating a sample of problems/starts reproduces the recorded checksums/values.
- Variance functions positivity (checked over μ ∈ (0,1)): V1 ≥ 0.737, V2 ≥ 1, V3 (code) ≥ 0.368, V4 ≥ 1; all positive.
- Found the V3 sign mismatch (Open question 4, resolved by Vivak: the code is correct, the manuscript will be corrected).
- Next: Phase 2.2 (`src/run_job.jl`).
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
