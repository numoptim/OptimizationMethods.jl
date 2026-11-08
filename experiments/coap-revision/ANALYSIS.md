# Analysis guide (for a fresh session)

You have: this branch (`cv/experiments-quasi-likelihood`) and the merged results of the CHTC runs in
`experiments/coap-revision/results/merged/{runs,hits,ours_events}.csv`. You do **not** have the manuscript or the
raw job output. Your task: rebuild every analysis output from the merged CSVs, sanity-check it, and
report. Do not change the experiment code. All commands run from `experiments/coap-revision`.

## 1. Setup and rebuild (copy-paste)

```bash
# Julia 1.13 (skip if already installed)
curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.13
export PATH=$HOME/.juliaup/bin:$PATH
julia -e 'using Pkg; Pkg.Registry.add("General")'

cd experiments/coap-revision
ls -lh results/merged/                                                  # runs.csv hits.csv ours_events.csv must be here
julia --project=analysis -e 'using Pkg; Pkg.instantiate()'              # a few minutes (CairoMakie)
julia --project=analysis src/analysis/check_merged.jl                   # structural check; must print Test Summary with no failures
julia --project=analysis src/analysis/run_analysis.jl                   # about 1-3 minutes; writes results/paper/
julia --project=. test/test_analysis.jl                                 # unit tests of the analysis functions
```

`run_analysis.jl` options: `--merged DIR` (default `results/merged`), `--out DIR` (default `results/paper`),
`--no-plots` (skip PNGs). A worked example on a tiny pilot dataset is committed in `results/pilot_merged/`
(input) and `results/pilot_paper/` (output); run `julia --project=analysis src/analysis/run_analysis.jl --merged results/pilot_merged --out /tmp/pilot_paper` first if you want to see what to expect.

## 2. What the outputs are, and which manuscript items they replace

`results/paper/README.md` is generated with the outputs and lists every file. Summary:

| Output (`results/paper/`) | Replaces |
|---|---|
| `profile_main_<metric>_<ε>.csv`, `profile_main_<metric>.png` | **Fig. 3 and its discussion**: Dolan-Moré performance profiles of OURS, ARM, WATCH, SPEC, ADATR, ASTR1, LIPAPP on #F (`nobj`), #∇F (`ngrad`), weighted cost (`cost`) and wall time (`time`), one curve file per tolerance |
| `profile_all_<metric>_<ε>.csv`, `profile_all_<metric>.png` | the same for all 12 methods (appendix) |
| `reliability_<ε>_<metric>.csv`, `reliability.png` | **Reliability and performance tables** (columns as in the old tables: `algorithm.name, param, total.x, avg.reliability, sd.reliability, moe_plus.x, moe_minus.x, total.y, avg.<metric>, sd.<metric>, moe_plus.y, moe_minus.y`) |
| `events_<ε>.csv`, `events_1e-3.png` | **Table 4** (OURS triggering events, percent of inner loops, plus rejection rate) |
| `algorithms_table.csv`, `all_algorithms_reliability.csv` | **Supplement algorithm table** (the 100 algorithms; reliability at each tolerance) |
| `best_configurations.csv`, `profile_summary.csv`, `success_summary.csv` | supporting numbers: which configuration is best, ρ(1) and ρ(∞), mean/median costs of successful runs |

The paper's own numbers at 1e-3 (best configuration per method, 1,200 pairs, different random instances and
budget) for a plausibility check of `reliability_1e-3_ngrad.csv`: ARMIJO reliability 0.578 (mean #∇F 1,437,
mean #F 9,463); BBGD 0.472 (452, 0); DIMINISHING 0.613 (961, 0); FIXEDGD 0.568 (55, 0); OURS triggering
events (percent) ball 1.36, lower 64.19, upper 30.18, max-iteration 4.34, about 12% of outer iterations rejected.
Expect the same order of magnitude, not equality. If a method is wildly off (reliability near 0 or 1 where the
paper has about 0.5), look at `runs.csv` statuses for that method before reporting.

## 3. Column schemas

### `runs.csv`: one row per run (problem, start, algorithm); 120,000 rows
`problem_id` (1-120, `config/problems.csv`), `n` (variables 10/50/90), `m` (observations 100/1000), `V` (variance
function V1-V4), `replicate` (1-5), `start_id` (1-10, `config/starts.csv`), `algorithm_id` (1-100, `config/algorithms.csv`),
`method`, hyperparameters `eta` (initial step grid 1e-4,1,2,4,6,8,10; empty for OURS and ARM), `long_stepsize` (BBGD, SPEC),
`adatr_alpha` (ADATR 0/0.9), `astr1_variant` (adagrad/maxg), `status` (see below), `iterations` (completed
iterations), `n_obj`, `n_grad` (NLPModels counters of the whole run; the post-run evaluations done for
the analysis are never counted), `time_total_s`, `time_obj_s`, `time_grad_s` (wall seconds of the solver call, and the
part inside objective / gradient calls), `F0` (F at the start), `F_final`, `gradnorm0`, `gradnorm_final` (norm of the
gradient at the last completed iterate), `julia_version`, `hostname`, `cpu_model`, `job_id`.

`status`: `converged_1e-9` (reached 1e-9), `budget_grad` (#∇F reached 200,000), `budget_obj` (#F reached 200,000),
`timeout` (per-run wall-clock safety cap of 2x the worst-case run time), `nan_or_inf` (non-finite gradient or iterate),
`stalled` (the solver stopped by itself without reaching 1e-9: failed line search), `error` (exception; message in
the job's `errors.log`, not in the merged files).

### `hits.csv`: 4 rows per run (tolerances 1e-3, 1e-5, 1e-7, 1e-9); 480,000 rows
`problem_id, start_id, algorithm_id`, `tol`, `hit` (true/false; **explicit false rows**, then the numeric
fields are empty), `iteration`, `n_obj`, `n_grad`, `time_total_s`, `time_obj_s`, `time_grad_s` (all cumulative at the
first time the method's own termination test would have fired at `tol`), `gradnorm` (at that iterate), `F_at_hit`
(F at the hit iterate, evaluated after the run and not counted), `reliable` (= hit **and** `F_at_hit < F0`: the paper's
definition of reliability, per tolerance).

### `ours_events.csv`: one row per OURS inner loop
`problem_id, start_id, algorithm_id` (always 50), `k` (outer iteration, 1-based), `j_k` (inner-loop length), `ball`,
`lower`, `upper`, `maxiter` (which triggering events fired; a loop can have more than one), `accepted` (whether the
outer iterate was accepted by the non-sequential Armijo test; empty if the budget ended before the test).

## 4. Caveats you must keep in mind

* **Tolerances come from one run.** Every algorithm runs to 1e-9 (or its budget); the 1e-3, 1e-5, 1e-7 results are the first
  times that tolerance was reached in that same run (`hits.csv`). A run that stops by budget has hits only for the
  tolerances it reached.
* **Budget hits are failures at the tolerances they did not reach.** Do not drop `budget_*` rows. `reliable` is false
  when the tolerance was not hit.
* **How OURS's tolerance test works.** OURS tests `‖Ḟ‖` only at accepted or rejected **outer** iterates, never at the
  inner-loop iterates (as the algorithm is written). Hits for OURS are therefore recorded at outer iterates, and
  a rejected outer iterate repeats the previous gradient norm. The event proportions in `events_<ε>.csv` use the inner
  loops up to the outer iteration of the first hit of ε.
* **OURS differs from the original submission in one place.** The terminal local-Lipschitz estimate after an inner loop of
  exactly one step now follows the manuscript's Algorithm 2 (it used a stale value before). About 23% of inner loops are
  affected, so OURS numbers are expected to differ from the old paper's. See `OURS_vs_Algorithm1.md`.
* **Weighted cost** = #F·c + #∇F with c = (mean time per F)/(mean time per ∇F), computed **per run** from that run's timers
  (`time_obj_s/n_obj` over `time_grad_s/n_grad`); runs with no F evaluations have c = 0. F is computed by numerical
  quadrature (about 30-280 times the cost of a gradient), so the weighted cost is dominated by #F for F-using methods.
* **Timing noise.** Wall times depend on the CPU, the load of the machine and Julia's garbage collector. Jobs are pinned to
  one CPU level (`Microarch`), but check `cpu_model` and `hostname` for heterogeneity (`combine(groupby(runs, :cpu_model), nrow)`).
  The timing layer adds below 2% overhead. Wall-time profiles are the noisiest ones; #F and #∇F are exact.
* **#F = 0 for gradient-only methods** (FIXED, DIM, BBGD, LIPAPP, NEST, WEIGHT, ADATR, ASTR1). In the #F profile the metric is #F + 1.
* **A NaN gradient norm is never a hit** (the solvers' own stopping tests would treat NaN as converged; the recorder does not).
  Such runs are `nan_or_inf` and unreliable.
* **Reliability can include saturated points.** A run can "succeed" at a point where the logistic mean saturates (tiny
  gradient far from the data's solution) as long as `F < F0`; this is the paper's definition, kept as is.
* **F values are accurate to the quadrature tolerance** (`quadgk`); differences of F below about 1e-8 are noise.
* **Instances are new.** Seeds are recorded (`config/problems.csv`, `config/starts.csv`) but differ from the old paper's
  unrecorded ones. V3 is `exp(-|μ-1|^4.5)` (the manuscript is being corrected to this sign).
* **Sample size.** 1,200 pairs per algorithm; margins are 95% normal-approximation margins (1.96 sd/√n), unclamped.
* **Choices made in the analysis** (all in `results/paper/README.md`): best configuration per method **per tolerance**
  (highest reliability, ties by smaller median #∇F of successful runs); profile pairs that no solver of the set
  solves are dropped (counted in `profile_summary.csv`); events are averaged over runs.

## 5. What to report back

1. Confirm that `check_merged.jl` and the unit tests passed and give the status counts per method.
2. Say which configuration is best for each method at each tolerance (`best_configurations.csv`).
3. Look at `profile_main_*.png`, `reliability.png` and `events_1e-3.png`; describe in a few sentences how OURS compares with ARM, WATCH,
   SPEC, ADATR, ASTR1 and LIPAPP on each metric and tolerance, and flag anything surprising (a method with reliability 0, a large
   `n_pairs_dropped`, many `stalled`/`timeout` runs, a hostname/CPU imbalance).
4. Commit `results/paper/` to the branch and push (the raw and merged CSVs are git-ignored).
