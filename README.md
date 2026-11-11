# Phase 1 shared-store experiment

This orphan branch contains the standalone MWE only. Original v0.1.x code remains in the original checkout and on main. No remote operations are required.

## Reproduce

From this worktree (Julia 1.12-compatible; measured on Julia 1.13.1):

```sh
julia -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url="https://github.com/numoptim/NumOptimRegistry"))'
julia --project=dev/mwe -e 'using Pkg; Pkg.instantiate()'
julia --project=dev/mwe dev/mwe/test/runtests.jl
julia --project=dev/mwe dev/mwe/benchmark.jl
julia --project=dev/mwe dev/mwe/example.jl
```

If Julia is not on PATH, use `/Users/anggun/.juliaup/bin/julia`. Manifest.toml pins the measured environment; no temporary-path dependencies are used.

## How the code works

A method is a recipe: its immutable struct contains fixed settings. The store is the worksheet: it contains the current answer, gradient, progress and optional snapshots. `allocate!` prepares the worksheet, `initialize!` computes the first gradient, `step!` makes one update, and `optimize!` repeats until a stopping rule applies. The history is a photo album; the algorithm never consults it.

Fixed GD subtracts step size times gradient. Backtracking tries smaller steps until the objective decreases enough, committing only an accepted trial. Nesterov carries two extra vectors and acceleration weights; it measures progress at x but prepares the next update using the gradient at y.

## Contract

Use a fresh `allocate(problem; type=T)` store and a `Vector{T}` initial point. History keys must exist after initialization. Without History there is no history key. Records deep-copy values; `every` controls frequency, with initial and final records always included. Final records replace an existing record at the same iteration. If no gradient evaluation is permitted, the initial gradient norm remains Inf (unmeasured).

StoppingCriteria supports grad_norm, max_iterations, max_grad_evals and max_obj_evals. Budgets refer to cumulative problem counter batch-equivalents, not per-run differences. Do not solve concurrently against the same problem: counters are shared. Nesterov may stop with one unused gradient evaluation because a whole step needs two. Iteration counts accepted outer steps. On line-search failure, x and its gradient remain unchanged; :obj can describe the rejected trial and :obj_x describes the accepted iterate. max_ls_iterations counts reductions; zero allows the initial trial only.

The benchmark includes store allocation and initialization and reports minimum-of-three warmed samples, each with 20 updates. Per-update values amortize that setup. It is a preliminary whole-run overhead comparison, not an isolated kernel benchmark or a statistical speedup claim.

## Development navigation

- [Architecture plan](docs/src/dev/mwe/plan.md)
- [Findings and measured results](docs/src/dev/mwe/findings.md)
- [Reference roadmap](docs/src/dev/roadmap.md)

The original Documenter site is not copied into this clean orphan branch; navigation is provided here. The Decisions Log remains unchanged pending review with Vivak.

## Wednesday, November 11

Show the example, reference tests and measurements. Discuss counter ownership, strict budget semantics, iteration count on failure, and whether history should use typed/preallocated columns. Ask where the interface generics should live. Do not present these prototype decisions as approved upstream API changes.
