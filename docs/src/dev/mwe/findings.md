# Phase 1 findings — November 10, 2026

Prototype implemented; maintainer review remains pending. Original main checkout unchanged. Registry versions: OptimizationModels 0.1.8, OptimizationProblems 0.1.5, BenchmarkTools 1.8.0. Full resolution is in the manifest.

## Q1 — Keys

The proposed keys work for all three methods. Additional keys are :trial (uncommitted line-search point), :D (Nesterov weight), and :stop_spec (inner-loop budget access). Allocation rejects collisions. :obj remains the latest evaluation, while :obj_x identifies the accepted objective. Recommendation: formally document ownership and evaluation-point semantics; do not promise :grad always belongs to :x.

## Q2 — Performance

The measured small no-history driver is about 0.9% above the handwritten baseline; history is about 6.2% above baseline. Large-case differences are within a narrow range and three samples cannot establish a speedup. The arithmetic function barrier saves 40 allocations per 20-update run relative to the unbarriered implementation at both sizes, but problem evaluation allocations dominate (about 1.81 GB per large run). Recommendation: retain a small arithmetic barrier; profile upstream gradient evaluations before redesigning the store for performance. No claims about isolated dispatch costs or general hardware performance.

Julia 1.13.1 | arm64-apple-darwin27.0.0 | CPU: Apple M5 | Julia threads: 1 | BLAS threads: 4
20 updates/run; setup, initialization and history included; minimum of 3 samples, evals=1. Per-update values amortize complete run.
| parameters | observations | case | ms/run | μs/update | bytes/run | allocations/run |
|---:|---:|:---|---:|---:|---:|---:|
| 10 | 100 | handwritten | 1.05875 | 52.9375 | 1078960 | 21113 |
| 10 | 100 | driver | 1.068167 | 53.40835 | 1083616 | 21327 |
| 10 | 100 | history | 1.124292 | 56.2146 | 1099808 | 21551 |
| 10 | 100 | no_barrier | 1.083083 | 54.15415 | 1084896 | 21367 |
| 1000 | 10000 | handwritten | 714.5965 | 35729.825 | 1811060000 | 2310116 |
| 1000 | 10000 | driver | 710.02675 | 35501.3375 | 1811072768 | 2310331 |
| 1000 | 10000 | history | 709.716 | 35485.8 | 1811267424 | 2310577 |
| 1000 | 10000 | no_barrier | 716.372958 | 35818.647899999996 | 1811074048 | 2310371 |


Bytes and allocations per update are the reported run values divided by 20; these include initialization/setup overhead. Benchmarks use the same problem, starting point, step size and 20 updates for each comparison; counters reset between samples. Random seed 2026, Float64. No Julia/Python comparison.

## Q3 — History

Push-based columns and a store-resident History spec keep recording optional. deepcopy covers arrays and mutable Counter objects. Forced terminal recording replaces any same-iteration snapshot, avoiding duplicates after failures or every>1. Tests check schedule and independent snapshots. Recommendation: retain push-based storage for the prototype; typed or preallocated history columns require further measurement. History never invokes evaluations.

## Q4 — Driver fit

All three methods share the driver. Fixed/backtracking consume one gradient per accepted update, Nesterov two. Nesterov matches the reference B, D, y and z recurrence and retains the gradient at y while recording the gradient norm at x. Recommendation: retain the generic loop; reserve complete-step gradient budgets before invoking it.

## Q5 — Failure signaling

step! sets stop_reason. Rejected line-search trials are isolated in :trial; failure leaves the accepted point and its gradient unchanged. :iter counts accepted updates, not failed attempts. A zero reduction budget still attempts the initial candidate. Objective budgets are checked before each evaluation. Recommendation: agree on accepted-step versus attempted-step counts before the production port. :ls_iterations measures trial attempts. Nonfinite accepted state terminates with :nonfinite; initialization handling is also checked.

## Q6 — Types

Float32 and Float64 x/store pairs are exercised; hyperparameters can promote intermediate arithmetic before writing into the fixed buffer type. x0 must be a floating Vector matching the gradient buffer. Recommendation: retain parametric method structs and explicit conversion policy at allocation. Mixed problem-data precision, BigFloat, GPU arrays and arbitrary vector types remain uninvestigated; upstream currently requires Vector{T}.

## Q7 — Package friction / O1 / O2

allocate and allocate! have distinct roles and are usable together. Depending directly on OptimizationProblems for generics couples methods to implementations: moving declarations to OptimizationModels is a Phase 2 recommendation, not implemented here. Counters alias the problem: reuse accumulates budget usage, and concurrent stores are not independent. Recommendation: consider store-owned counters upstream; until then document cumulative budgets and prohibit shared-problem concurrent solves. For these full-batch/full-coordinate methods batch_equivalent and block_equivalent each advance by one; they must not be added together.

## Verification and limits

See dev/mwe/test-results.txt for actual test totals. All three algorithms are tested against independently written loops on logistic regression. Checks cover counter counts, stopping, schedules, snapshots, failure and type handling. BenchmarkTools completed both requested sizes. Documentation is Markdown with module docstrings; the original Documenter site is deliberately absent from the orphan branch, so its build has not been verified. Original algorithm references remain in Nesterov's docstring and the copied roadmap. Findings have NOT yet been reviewed with Vivak; acceptance requiring his review remains outstanding.
