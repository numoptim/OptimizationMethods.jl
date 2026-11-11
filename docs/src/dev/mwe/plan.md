# Phase 1 plan

Standalone orphan experiment based on main 76d7294; original checkout preserved.
Follow core → fixed GD → benchmarks → backtracking → Nesterov → findings.
Import evaluations from OptimizationProblems; OptimizationModels owns the problem and Counter types.
Fixed immutable configurations; all mutable state in Dict{Symbol,Any}. History deep-copies selected keys, never evaluates functions. Accepted-step count is :iter; failures leave it unchanged. Final recording is forced and deduplicated.
Evaluation budgets are absolute full-batch counts on problem.counters (batch_equivalent), checked before initialization and before complete steps. Nesterov reserves two gradients per step. Objective budgets are enforced inside line search. Counters are not reset implicitly.
Method-owned keys must not collide with an existing store. Require a fresh store for each solve and floating x matching the gradient buffer type. Function barrier isolates vector arithmetic.
Files: core.jl owns configurations, allocation, recording and driver; each method owns its state and update; line_search.jl owns trial acceptance; tests compare independent loops; benchmark.jl measures actual runs; findings.md records Q1–Q7. O1/O2 remain upstream decisions for discussion with Vivak.
