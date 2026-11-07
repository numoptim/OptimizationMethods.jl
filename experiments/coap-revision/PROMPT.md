# Prompt: COAP revision experiments on `cv/experiments-quasi-likelihood`

_For a cloud Claude Code session (Sonnet 5.5, medium effort) connected to `numoptim/OptimizationMethods.jl`. Written 2026-11-06._

---

You are preparing the numerical experiments for a journal revision of "An Event-Driven Optimization Method for a Class of Data Science Problems" (Varner and Patel), under major revision at *Computational Optimization and Applications* (COAP-D-26-00352).

**Your environment.**
- You run in the cloud with only the `numoptim/OptimizationMethods.jl` repository.
- You do **not** have the manuscript or the old experiment repository; both are private. Everything you need from them is in the **Appendix** of this prompt.
- You may use the internet to fetch the two papers in §2. If a fetch fails, use the summaries in §2, and record the failure in `STATUS.md` under Open questions.

**Who reads your work.** Vivak (the PI) will **not** read or review your code. He will:
1. read `STATUS.md`;
2. run the CHTC jobs by following your `CHTC_RUNBOOK.md`;
3. give the merged results to a fresh session, which follows your `ANALYSIS.md`.

So every piece of code must pass the checks below, every decision must be written down, and every instruction for Vivak must be copy-paste ready.

## 0. Run in chunks (read this first, every time)

This work is done over **many short runs**, to limit usage. Each run is started fresh, by a schedule or by Vivak, with this prompt. In each run:

1. **Resume.** Fetch the branch. If `experiments/coap-revision/STATUS.md` exists, read it and continue from the first unchecked checklist item. Don't redo checked items. Don't rely on memory of earlier runs; STATUS.md and the git log are the only record.
2. **Do one chunk.** A chunk is **one numbered step of §4** (for example "Phase 1.2"), or a single gate check. Don't start a second step in the same run, even if the first went quickly.
   - If a step is too big for one run, split it into sub-items in the checklist (for example "1.4a ADATR docstring", "1.4b ADATR implementation"), do one, and leave the rest unchecked.
3. **Stop cleanly.** Before stopping, always:
   - leave the tests passing, or record exactly what's failing in the Log;
   - update STATUS.md (checklist, Log entry with date and what's next);
   - commit and push.

   Never stop with uncommitted work.
4. **Finished?** When every checklist item is checked, write `ALL DONE` on the first line of STATUS.md. Any later run that sees it stops immediately without doing anything.
5. **Blocked?** If you're blocked on a decision only Vivak can make, write `BLOCKED: <one line>` on the first line of STATUS.md, then push and stop. Later runs that see it also stop immediately until Vivak removes the line.

Long local computations, such as the Gate 2 pilot, can run in the background within the run. If one can't finish in a single run, save partial results, record how to resume, and continue next run.

## 0a. Ground rules

**Branch**
- Work on **`cv/experiments-quasi-likelihood`**: `git fetch origin && git switch cv/experiments-quasi-likelihood`.
  - If the cloud environment forces you onto a different branch, base it on `origin/cv/experiments-quasi-likelihood`, and put the branch name at the top of `STATUS.md`.
- This branch is isolated and disposable. It will **not** be merged into `main`.
  - Don't rebase it, merge `main` into it, or touch other branches.
  - Commit after each step that passes its checks, and push after every phase gate (§4).

**`STATUS.md` is the single source of truth.** Create it first, at `experiments/coap-revision/STATUS.md`, with this structure, and keep it current with every commit:

```
# COAP revision experiments: STATUS
Branch: <name> · Last updated: <date, UTC> · Current phase: <n>

## Checklist
- [x] / [ ] one line per step of §4 (Phase 1.1 … Phase 4.3), in order
## Open questions for Vivak
(numbered; what you chose in the meantime)
## Decisions
(every parameter or design choice not fixed by this prompt, with a reason)
## Development plan
(the linear plan: phases 1–4 from §4, each step with what "done" means and how it was verified)
## Log
(dated entries: what was done, which checks ran, results)
```

- The checklist is the first thing Vivak reads: what is done and what is still open.
- Don't use other tracking files. The deliverables in §4 are separate, but decisions and progress go here.

**Design**
- The branch uses the **old design**: NLPModels.jl problems and NLPModels counters. Keep it. Don't port anything to OptimizationModels.jl or OptimizationProblems.jl.
- For new methods, put the update rule in a small pure function of `(gradient, state)`, separate from the NLPModels calls, so it can be ported later.
- Follow the branch's conventions:
  - one file per method in `src/methods/`, with a mutable `...{T} <: AbstractOptimizerData{T}` struct and a solver function;
  - Blue style;
  - full docstrings with ```` ```math ```` blocks;
  - tests in `test/methods/`, wired through `test/test.txt`.

**Julia**
- Install Julia **1.13** (latest patch, currently 1.13.2) with `juliaup`.
- Set `julia = "1.13"` in `[compat]`, re-resolve, and run the full test suite once before any change. Record the baseline (pass/fail per file) in the Log.
- Don't fix unrelated failures that already existed; list them.

**Code locations.** All experiment and analysis code goes under `experiments/coap-revision/`, with its own `Project.toml` that `dev`s the package at `../..`.

**When unsure.** If something here is ambiguous or impossible:
1. add it to Open questions;
2. make the most conservative choice and record it under Decisions;
3. keep going.

Stop only if a choice would make the experiments invalid. In that case, say so at the top of `STATUS.md` and push.

## 1. Experiment protocol

**Problems.** These are the same quasi-likelihood suite as the paper (Appendix A):
- 4 variance functions × $n \in \{10, 50, 90\}$ × $m \in \{100, 1000\}$ × 5 datasets each, giving **120 problems**;
- 10 starting points per problem, uniform on $[-10, 10]^n$, giving **1,200 problem–start pairs**.

Use the problem types and parameters in Appendix C (the old scripts' choices).
- Every dataset and every starting point must come from an **explicit, recorded seed**. Store the seeds in `config/problems.csv` and `config/starts.csv`.
- The old scripts did not record seeds, so the instances will differ from the paper's. That's expected, because everything is rerun.
- Check that the generated data satisfy the paper's design: $y_i \in (0, 1)$, and $g(x_i^\top\theta^*) \in (0.3, 0.7)$. Record the check.

**Algorithms.** Rerun every method from the paper, and add three new ones.
- The initial-step grid used everywhere is $\eta \in \{10^{-4}, 1, 2, 4, 6, 8, 10\}$.
- Hyperparameters of the existing methods are given **exactly** in Appendix C. Use those values.

| Abbrev. | Struct / solver on this branch | Grid |
|---|---|---|
| FIXED | `FixedStepGD` | step ∈ η |
| DIM | `DiminishingStepGD` with `stepdown_100_step_size` | scaling ∈ η |
| BBGD | `BarzilaiBorweinGD` | {long, short} × η |
| LIPAPP | `LipschitzApproxGD` | η |
| NEST | `NesterovAcceleratedGD` | η |
| WEIGHT | `WeightedNormDampingGD` | damping = 1/η |
| OURS | `NonsequentialArmijoAdaptiveGD` | δ0 = 1, δ_upper = 1, ρ = 1e-4 |
| ARM | `BacktrackingGD` | α = 1 |
| WATCH | `WatchdogFixedSimpleGD` | α ∈ η |
| **SPEC** (new) | `SafeBarzilaiBorweinNLSMaxValGD`: spectral gradient + GLL nonmonotone line search | {long, short} × η, window 10 |
| **ADATR** (new) | Grapiglia–Stella, first-order (§2.1) | {α = 0, α = 0.9} × η |
| **ASTR1** (new) | Gratton–Jerad–Toint, first-order (§2.2) | {adagrad, maxg} × η |

- **SPEC:** use δ = 0.5, ρ = 1e-4 and 100 line-search iterations, matching ARM. For the step safeguards, use the struct's defaults (or Raydan 1997 if there are none), and record them under Decisions.
- **Totals:** 58 + 14 + 14 + 14 = **100 algorithms**, × 1,200 pairs = **120,000 runs**. Write the table, with an integer `algorithm_id`, to `config/algorithms.csv`.

**One run, four tolerances.**
- Run every algorithm with threshold $10^{-9}$ and an evaluation budget (§3).
- During the run, record the **first time** the method's own termination test would have fired at each $\epsilon \in \{10^{-3}, 10^{-5}, 10^{-7}, 10^{-9}\}$: the same check it uses to stop, with `threshold` replaced by ε.
  - For OURS, check whether the code tests $\|\dot F\|$ at accepted outer iterates only, or also at inner iterates. Use what the code does, and document it.
- At each first hit, record:
  - the cumulative #F and #∇F (NLPModels counters);
  - the cumulative wall time, time inside objective calls, and time inside gradient calls;
  - the iteration index and $\|\dot F\|$;
  - a copy of the iterate.
- After the run, evaluate $F$ at each hit iterate, at $x_0$ and at the final iterate, with the counters saved and restored, so these evaluations are **never counted**.
- **Reliability** at ε means the method hit ε and $F(x_{\text{hit}}) < F(x_0)$. This is the paper's definition, applied per tolerance.

**Cost measured inside the run.**
- Add a thin timing layer around the objective/gradient calls the solvers use (`time_ns()`), not edits to each solver's math.
- From these timings, analysis computes c = (mean time per F) / (mean time per ∇F) per run, and the weighted cost #F·c + #∇F.
- Check that the overhead is under 2% on the largest problem (n = 90, m = 1000), and log the measurement.

**OURS inner loops.** For every OURS run, save one record per inner loop:
- the outer index k;
- which events fired (ball, lower-gradient, upper-gradient, max-iteration), from `triggering_event_hist`;
- whether descent was accepted;
- the inner-loop length $j_k$.

These feed the paper's triggering-event table (paper results: 1.36% ball, 64.19% lower, 30.18% upper, 4.34% max-iteration, and about 12% rejected).

## 2. New methods

For each method:
1. **First** commit the docstring, with the algorithm statement and equation numbers from the source paper, plus a stub struct and solver that throw `error("not implemented")`.
2. **Then** implement it.
3. **Then** write the tests.

Fetch the PDFs and convert them to text with `pdftotext -layout`; install `poppler-utils` if needed. Read the algorithm sections yourself. **The papers govern** if they differ from the summaries below.

### 2.1 ADATR: Grapiglia and Stella (2022)
"An adaptive trust-region method without function evaluations", *Comput. Optim. Appl.* 82:31–60, doi:10.1007/s10589-022-00356-0.
- PDF: https://research.dial.uclouvain.be/bitstreams/84b36909-7b06-466c-9607-cf54972f4260/download. Algorithm in §2, eqs. (2.1)–(2.5); settings in §3.

**Algorithm.**
- Trust region: $\|d\| \le \Delta_k = \delta_k\|\nabla f(x_k)\|$, with $\delta_k = 1/b_k$ and $\omega_0 = \|\nabla f(x_0)\|$.
- Update (2.4):
  - if $\|\nabla f(x_{k+1})\| \le \alpha\,\omega_k$, set $(b_{k+1}, \omega_{k+1}) = (\hat b_k, \|\nabla f(x_{k+1})\|)$, where $\hat b_k \in [b_{\min}, b_k]$;
  - otherwise, set $(b_{k+1}, \omega_{k+1}) = (b_k + \|\nabla f(x_{k+1})\|^2/b_k,\ \omega_k)$.

**First-order version.**
- Use $B_k = 0$, so $d_k = -\delta_k\nabla f(x_k)$ and $\|d_k\| = \Delta_k$.
- Document that the paper's own experiments used BFGS with Steihaug–Toint, and that we use the first-order case so every compared method is first-order.
- With α = 0, the update coincides with WNGrad. Keep it anyway as a check.

**Parameters.**
- Flexible-variant rule (AdaTrust2, §3): $\hat b_k = \min\{\hat b_{\max}, \max\{b_{\min}, b_k/2\}\}$ (here $\|d_k\| > \Delta_k/2$ always holds), with $\hat b_{\max} = \|\nabla f(x_0)\|$ and $b_{\min} = 10^{-4}$.
- Grid: $b_0 = 1/\eta$ and α ∈ {0, 0.9}.
- Cost: one gradient per iteration and **zero objective evaluations**.

### 2.2 ASTR1: Gratton, Jerad and Toint
"Complexity of a class of first-order objective-function-free optimization algorithms", arXiv:2203.01647 (v3, June 2023).
- PDF: https://arxiv.org/pdf/2203.01647. Algorithm 2.1; weights in eq. (3.1) and §4 eq. (4.1); variants in §5, Table 1.

**Algorithm.**
- Trust region: $\Delta_{i,k} = |g_{i,k}|/w_{i,k}$.
- With $B_k = 0$ and $\gamma_k = 1$, the step is $s_{i,k} = -g_{i,k}/w_{i,k}$.
- Weights, scaled by $\theta = 1/\eta$ (the paper's step-size parameter, footnote 4):
  - **adagrad:** $w_{i,k} = \theta(\varsigma + \sum_{j \le k} g_{i,j}^2)^{1/2}$, with $\varsigma = 1/100$;
  - **maxg:** $w_{i,k} = \theta(k+1)^{1/10}\max\{1/100, \max_{j \le k}|g_{i,j}|\}$.
- Grid: {adagrad, maxg} × η.
- Cost: one gradient per iteration and **zero objective evaluations**.

**Bibliography note.** Add this to Open questions; don't act on it. The manuscript's `gratton2022firstorder` cites arXiv:2203.01757, which is now titled "Complexity and performance for two classes of noise-tolerant first-order algorithms" (v3, Jan 2025). ASTR1 is in arXiv:2203.01647.

### 2.3 Tests for each new method (all must pass before Phase 2)
1. **Hand-computed iterations.** On $f(x) = \tfrac12 x^\top \mathrm{diag}(1, 10)\,x$, compute three iterations by hand in plain arithmetic inside the test. Compare them to the solver with rtol 1e-12, for both variants.
2. **Invariants.**
   - ADATR: $b_k \ge b_{\min}$; $\|d_k\| = \Delta_k$; with α = 0, $\delta_k$ is non-increasing.
   - ASTR1: $|s_{i,k}| \le \Delta_{i,k}$; for adagrad, $w$ is non-decreasing.
3. **Zero objective evaluations.** On `LeastSquares`, `neval_obj == 0`, and the exact `neval_grad` count is asserted.
4. **Convergence.** On a well-conditioned 50×10 `LeastSquares`, $\|\nabla f\| \le 10^{-6}$ within 20,000 iterations for η = 1.
5. **Edge cases.** A start at a stationary point stops immediately, and `max_iterations` is respected.

## 3. Budget and CHTC sizing

The paper used 10,000 iterations at 1e-3 (Appendix D has its averages). Here:
- Stop on **evaluation counts**: #∇F ≥ G_max or #F ≥ F_max. Set `max_iterations` high enough that it never binds.
- Start from **G_max = F_max = 200,000**, then size with the pilot:
  1. **Local timing pilot:** for each (n, m, V), time 1,000 F evaluations and 1,000 ∇F evaluations, giving $t_F$ and $t_\nabla$. Write `results/pilot/eval_times.csv`.
  2. The worst-case run time is $T_{\text{run}} = F_{\max}t_F + G_{\max}t_\nabla$. Methods without objective evaluations only pay the gradient part.
  3. **CHTC limits:**
     - jobs are held after **72 h**, so target a worst-case job time of **at most 24 h**;
     - with more than 1,000 jobs per submit file, the **mean job time must be at least 5 minutes**;
     - at most **10,000 jobs per submit file**, and at most 50,000 queued in total.

     Choose the job granularity (for example, one problem × one method family × all 10 starts) to meet these. If it can't fit, lower G_max/F_max uniformly. Log the arithmetic under Decisions.
  4. Add a per-run wall-clock **safety** cap of 2 × $T_{\text{run}}$, only against hangs. If it fires, the status is `timeout`.

Run statuses: `converged_1e-9`, `budget_grad`, `budget_obj`, `timeout`, `nan_or_inf`, `error`. Catch exceptions per run, so one failure never kills a job.

## 4. Phases and gates

Do the phases in order, and don't start one until the previous gate passes. At each gate:
1. update the checklist and Log in `STATUS.md`;
2. commit;
3. **push**.

**Phase 1: library**
1. Julia 1.13 baseline (§0).
2. **Timing layer and tolerance first-hit recorder**, shared by all solvers through a minimal hook in each iteration loop.
   - Test with `FixedStepGD` on `LeastSquares`: hits arrive in order and counts are non-decreasing.
   - The 1e-3 hit must exactly equal a separate run with `threshold = 1e-3`: same #F, #∇F and iterate.
3. **Evaluation-budget stopping** for every solver used, with tests. Line-search methods must stop cleanly mid-search.
4. **ADATR and ASTR1** (§2), docstring first.
5. **OURS vs Algorithm 1.** Compare `NonsequentialArmijoAdaptiveGD` line by line with Algorithm 1 (Appendix B).
   - Write the comparison to `experiments/coap-revision/OURS_vs_Algorithm1.md`, as a table: paper line, code line, difference, which one is right.
   - **Don't change OURS's behavior.** If you find a real bug, put it under Open questions and stop.
   - Known issues to confirm:
     - the paper's inner `while` condition is printed inverted (it should loop while ‖ψ−θ‖ ≤ 10 **and** ‖Ḟ(ψ)‖ ∈ (τ_low, τ_up) **and** j < 100);
     - the paper is missing a comma in the last `\State` of the `\Else` branch;
     - `EstimateLipschitz` is called at j = 0 with $\psi_{-1}$;
     - the √10 vs √20 question is **not** a mismatch, since √20 · ‖g‖/√2 = √10 · ‖g‖. Confirm the code does the same.

   **Gate 1:** the full test suite passes on Julia 1.13.

**Phase 2: experiment harness** (`experiments/coap-revision/`)
1. `config/problems.csv`, `config/starts.csv`, `config/algorithms.csv` (§1).
2. `src/run_job.jl JOB_ID` writes `results/raw/job_<id>/{runs,hits,ours_events}.csv` (schemas in §5). It writes to a temporary file and renames at the end.
3. `src/jobs.jl` builds `config/jobs.csv` from the §3 sizing.
4. `chtc/`:
   - `.sub` files, each with at most 10,000 jobs;
   - `submit_all.sh`;
   - `run_job.sh`, which unpacks Julia 1.13.2 and a pre-instantiated depot, then runs the job;
   - `build_env.sh`, which builds that tarball. Follow CHTC's current Julia guide at https://chtc.cs.wisc.edu.
   - A `requirements` line pinning one CPU type, using the correct ClassAd attribute from CHTC's docs (for example `Microarch` or `CPUModel`). Pick a common type, and include the `condor_status` command Vivak can use to check availability.
   - `request_cpus = 1`, plus `request_memory` and `request_disk` measured in the pilot, plus 50%.
5. `src/collect.jl [--pilot]` merges into `results/merged/{runs,hits,ours_events}.csv` and validates. It checks:
   - the exact row counts: 120,000 runs, and 4 hit rows per run with explicit `hit = false`;
   - every (problem, start, algorithm) appears exactly once;
   - no NaN/Inf in count or time columns;
   - only allowed status values;
   - hit counts are non-decreasing across tolerances;
   - #F = 0 for FIXED, DIM, BBGD, LIPAPP, NEST, WEIGHT, ADATR and ASTR1.

   It prints a summary, writes `results/merged/resubmit_jobs.txt`, and exits nonzero on any failure.

   **Gate 2, local pilot:**
   - run 3 problems (smallest, middle, largest) × 2 starts × all 100 algorithms;
   - `collect.jl --pilot` must pass;
   - spot-check 3 runs (OURS, ARM, ADATR) against direct solver calls.

**Phase 3: analysis code**, written and tested on the pilot data.
1. `src/analysis/profiles.jl`: Dolan–Moré performance profiles per tolerance, on four metrics: #F, #∇F, weighted cost, and wall time.
   - Use each method's best configuration: highest reliability at that tolerance, ties broken by median #∇F.
   - **Main set:** OURS, ARM, WATCH, SPEC, ADATR, ASTR1, LIPAPP.
   - **Appendix set:** all methods.
2. `src/analysis/tables.jl` produces:
   - reliability per method and tolerance, with 95% margins (format: Appendix D);
   - OURS triggering-event proportions and the rejection rate;
   - mean and median #F, #∇F and time for successful runs.
3. Outputs go to `results/paper/`:
   - **pgfplots-ready CSVs**, one per plot or table, since the paper inlines its data;
   - PNG previews;
   - `results/paper/README.md`, mapping each file to the paper figure or table it replaces.

   **Gate 3:** the analysis runs end to end on the pilot data.

**Phase 4: handoff**
1. **`experiments/coap-revision/CHTC_RUNBOOK.md`**, for Vivak, copy-paste ready:
   1. clone the branch on the CHTC submit node;
   2. build the environment tarball;
   3. **CHTC pilot:** about 20 jobs covering the smallest and largest problems, then `collect.jl --pilot`, plus one command that prints the max and mean job runtime from the HTCondor logs, checked against 24 h;
   4. `submit_all.sh`;
   5. monitor, and release or resubmit held jobs from `resubmit_jobs.txt`;
   6. collect, and copy `results/merged/` off CHTC;
   7. what to do if `collect.jl` fails.
2. **`experiments/coap-revision/ANALYSIS.md`**, for a **fresh** session that gets only the merged CSVs and this branch:
   - exact commands to rebuild every output;
   - column schemas;
   - caveats (timing noise, budget hits, how OURS's tolerance test works);
   - which paper figures/tables each output replaces (Appendix E).
3. `.gitignore` covers `results/raw/` and `results/merged/`. Commit only pilot-sized data.

   **Gate 4:** re-read `CHTC_RUNBOOK.md` as a newcomer, and confirm every command and path exists.

## 5. Data schemas (`results/merged/`)
- **`runs.csv`** (one row per run): `problem_id, n, m, V, replicate, start_id, algorithm_id, method, <hyperparameter columns>, status, iterations, n_obj, n_grad, time_total_s, time_obj_s, time_grad_s, F0, F_final, gradnorm0, gradnorm_final, julia_version, hostname, cpu_model, job_id`
- **`hits.csv`** (4 rows per run): `problem_id, start_id, algorithm_id, tol, hit, iteration, n_obj, n_grad, time_total_s, time_obj_s, time_grad_s, gradnorm, F_at_hit, reliable`
- **`ours_events.csv`** (one row per OURS inner loop): `problem_id, start_id, algorithm_id, k, j_k, ball, lower, upper, maxiter, accepted`

## 6. Definition of done
- Gates 1–4 pass, and everything is pushed.
- The top of `STATUS.md` has a short **"For Vivak"** block with:
  - the branch name;
  - the first three CHTC commands;
  - the number of jobs and the worst-case and median job times;
  - the open questions;
  - a link to `OURS_vs_Algorithm1.md`.
- Every checklist item is checked off or explained.

---

## Appendix A: Problem generation (from the manuscript, §7.2.1)

$$\min_{\theta\in\mathbb{R}^n} F(\theta) = -\sum_{i=1}^m \int_0^{g(x_i^\top\theta)} \frac{y_i-\mu}{V(\mu)}\,d\mu,$$

- $g$ is the logistic function;
- $n \in \{10, 50, 90\}$ and $m \in \{100, 1000\}$;
- 5 independent datasets per (n, m, V), giving 120 problems;
- 10 starts per problem, uniform on $[-10, 10]^n$.

Variance functions:
1. $V_1(\mu) = 1 + \mu + \sin(2\pi\mu)$
2. $V_2(\mu) = |\mu|^{4.5} + 1$
3. $V_3(\mu) = \exp(|\mu - 1|^{4.5})$
4. $V_4(\mu) = \log(|\mu - 1|^{4.5} + 1) + 1$

Data generation (for each $V_k$):

| Quantity | Distribution |
|---|---|
| True coefficients | $\theta^* \sim \mathcal U[g^{-1}(0.3)/n,\ g^{-1}(0.7)/n]^n$ |
| Features | $x_i \sim \mathcal U[0,1]^n$, independently |
| Errors | $\epsilon_i \sim \mathcal U[-0.3/\tilde V_k^{1/2},\ 0.3/\tilde V_k^{1/2}]$, independently, where $\tilde V_k \approx \max_{\mu\in[0.3,0.7]} V_k(\mu)$ |
| Responses | $y_i = g(x_i^\top\theta^*) + V_k(g(x_i^\top\theta^*))^{1/2}\epsilon_i$ |

These distributions ensure $y_i \in (0, 1)$, $g(x_i^\top\theta^*) \in (0.3, 0.7)$, and $V_k(\mu)^{1/2}\epsilon_i \in (-0.3, 0.3)$ for $\mu \in (0.3, 0.7)$. Check whether the branch's problem constructors already generate data this way, and record what they do.

## Appendix B: Algorithm 1 as printed in the manuscript

```latex
\Require $F, \dot F, \theta_{0}, \epsilon > 0$
\Require Choose $\rho \in (0, 1), \overline{\delta} \in (0, \infty), \delta_0 \in (0, \overline{\delta}]$
\State $k \leftarrow 0$
\State $\tau_{low}^0, \tau_{upp}^0 \leftarrow \|\dot{F}(\theta_0)\|/\sqrt{2}, \sqrt{10}\|\dot{F}(\theta_0)\|$
\While{$\| \dot F(\theta_k) \| > \epsilon$}
    \State $j, \psi_0^k \leftarrow 0, \theta_k$
    \State $\hat{L}_j^k \leftarrow \textbf{EstimateLipschitz}(j, k, \psi_{j}^k, \psi_{j-1}^k)$
    \While{$\| \psi_j^k - \theta_k \| > 10$ \textbf{or} $\| \dot F(\psi_j^k)\| \not\in (\tau_{low}^k, \tau^k_{upp})$ \textbf{or} $j == 100$}   % printed inverted
        \State $C_{j,1}^k \leftarrow \|\dot{F}(\psi_j^k)\|^3 + .5\|\dot{F}(\psi_j^k)\|^2 \hat{L}_j^k + 10^{-16}$
        \State $C_{j,2}^k \leftarrow \|\dot{F}(\psi_j^k)\| + .5 \hat{L}_j^k + 10^{-16}$
        \State $\alpha^k_{j} \leftarrow \min( (\tau^{k}_{low})^2/C_{j,1}^k, 1/C_{j,2}^k ) + 10^{-16}$
        \State $\psi_{j+1}^k, j \leftarrow \psi_j^k - \delta_k \alpha_j^k \dot{F}(\psi^k_j), j+1$
        \State $\hat{L}_j^k \leftarrow \textbf{EstimateLipschitz}(j, k, \psi_{j}^k, \psi_{j-1}^k)$
    \EndWhile
    \If{$F(\psi_j^k) \geq F(\theta_k) - \rho \delta_k \alpha_0^k \|\dot{F}(\theta_k)\|^2$}
        \State $\theta_{k+1}, \delta_{k+1} \leftarrow \theta_k, .5 \delta_k$                      % reject
    \ElsIf{$\| \dot F(\psi^k_j)\| \leq \tau_{low}^k$}
        \State $\theta_{k+1}, \delta_{k+1} \leftarrow \psi_j^k, \delta_k$
        \State $\tau_{low}^{k+1}, \tau_{upp}^{k+1} \leftarrow \|\dot{F}(\theta_{k+1})\|/\sqrt{2}, \sqrt{20}\tau_{low}^{k+1}$
    \ElsIf{$\| \dot F(\psi^k_j)\| \geq \tau_{upp}^k$}
        \State $\theta_{k+1}, \delta_{k+1} \leftarrow \psi_j^k, \min\{1.5\delta_k, \overline{\delta}\}$
        \State $\tau_{low}^{k+1}, \tau_{upp}^{k+1} \leftarrow \|\dot{F}(\theta_{k+1})\|/\sqrt{2}, \sqrt{20}\tau_{low}^{k+1}$
    \Else
        \State $\theta_{k+1}, \delta_{k+1} \leftarrow \psi_j^k, \min\{1.5\delta_k, \overline{\delta}\}$
        \State $\tau_{low}^{k+1}, \tau_{upp}^{k+1} \leftarrow \tau_{low}^{k} \tau_{upp}^{k}$     % missing comma
    \EndIf
    \State $k \leftarrow k+1$
\EndWhile
\State \Return $\theta_k$
```

The paper uses ρ = 1e-4, δ0 = 1 and δ̄ = 1.

## Appendix C: Old experiment settings (from `numoptim/experiments-stepsizeselection`, which ran on this branch)

**Old run settings:** `THRESHOLD = 1e-3`, `MAX_ITERATIONS = 10000`, `STEPSIZE = [1e-4, 1, 2, 4, 6, 8, 10]`, and `x0` set per start.

**Problems** (constructor keyword arguments; `nobs` = m, `nvar` = n):
- V1: `OptimizationMethods.QLLogisticSin(T; nobs, nvar)`
- V2: `OptimizationMethods.QLLogisticMonomial(T; nobs, nvar, p = 2.25, c = 1.0)`
- V3: `OptimizationMethods.QLLogisticCenteredExp(T; nobs, nvar, p = 2.25, c = 1.0)`
- V4: `OptimizationMethods.QLLogisticCenteredLog(T; nobs, nvar, p = 2.25, c = 1.0, d = 1.0)`

**Algorithms** (constructor keyword arguments):
- `BarzilaiBorweinGD`: `init_stepsize = η`, `long_stepsize ∈ {true, false}`
- `FixedStepGD`: `step_size = η`
- `LipschitzApproxGD`: `init_stepsize = η`
- `NesterovAcceleratedGD`: `step_size = η`
- `DiminishingStepGD`: `step_size_function = OptimizationMethods.stepdown_100_step_size`, `step_size_scaling = η`
- `WeightedNormDampingGD`: `init_norm_damping_factor = 1/η`
- `NonsequentialArmijoAdaptiveGD`: `δ0 = 1.0`, `δ_upper = 1.0`, `ρ = 1e-4`
- `BacktrackingGD`: `α = 1.0`, `δ = 0.5`, `ρ = 1e-4`, `line_search_max_iteration = 100`
- `WatchdogFixedSimpleGD`: `α = η`, `δ = 0.5`, `ρ = 1e-4`, `line_search_max_iterations = 100`, `window_size = 1`, `inner_loop_max_iterations = 100`

**Solvers:** `barzilai_borwein_gd`, `fixed_step_gd`, `lipschitz_approximation_gd`, `nesterov_accelerated_gd`, `diminishing_step_gd`, `weighted_norm_damping_gd`, `nonsequential_armijo_adaptive_gd`, `backtracking_gd`, `watchdog_fixed_simple_gd`.

The old script reset counters with `OptimizationMethods.NLPModels.reset!(progData.counters)` and timed each run with `time()`.

## Appendix D: Paper results at 1e-3 (best configuration per method, 1,200 pairs)

| Method (param id) | Reliability | Mean #∇F (successful runs) | Mean #F (successful runs) |
|---|---|---|---|
| ARMIJO (1) | 0.578 | 1,437 | 9,463 |
| BBGD (10) | 0.472 | 452 | 0 |
| DIMINISHING (4) | 0.613 | 961 | 0 |
| FIXEDGD (10) | 0.568 | 55 | 0 |

Reliability CSV columns: `algorithm.name, param, total.x, avg.reliability, sd.reliability, moe_plus.x, moe_minus.x, total.y, avg.<metric>, sd.<metric>, moe_plus.y, moe_minus.y`

Triggering-event CSV columns: `Events, Average.Proportion.on.Inner.Loops, Margin.of.Error`. Rows: Bounding Ball Event 1.36, Lower Gradient Interval 64.19, Upper Gradient Interval 30.18, Max Iteration 4.34.

## Appendix E: What the outputs replace in the manuscript

- **Fig. 3 and its discussion:** replaced by performance profiles (#F, #∇F, weighted cost, CPU time) at each tolerance; main set in the text, all methods in the appendix.
- **Reliability and performance tables:** per method, per tolerance.
- **Table 4 (triggering events):** OURS event proportions and rejection rate.
- **Supplement algorithm table:** the full list of 100 algorithms (`config/algorithms.csv`).
