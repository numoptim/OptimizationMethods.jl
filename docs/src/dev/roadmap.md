# Development Roadmap

This page is the working development plan for redesigning `OptimizationMethods.jl` around
[`OptimizationModels.jl`](https://github.com/numoptim/OptimizationModels.jl) and
[`OptimizationProblems.jl`](https://github.com/numoptim/OptimizationProblems.jl).
It records the design decisions made so far, the target API, a detailed plan for the next
phase, a sketch of later phases, and the open questions.

Each phase is implemented with the `/incremental-dev` workflow:

1. a file-level plan, kept at `docs/src/dev/<phase>/plan.md`;
2. a stub and test outline for one function at a time;
3. the implementation and tests for one function at a time.

Every step is reviewed before the next begins.

!!! note "Status (2026-09-29)"
    The code in `src/` is still the v0.1.x implementation, which is based on NLPModels.jl.
    Nothing described on this page exists yet. Phase 1 (the MWE) is next.

## Overview

The three packages have separate roles.

| Package | Role |
|:--|:--|
| OptimizationModels.jl | Defines the interface: the `OptimizationProblem` abstract type, the fields every problem must have (`name`, `counters`, `num_param`, `num_obs`), `Counter`, and `validate`. |
| OptimizationProblems.jl | Implements problems (currently generalized linear models). `allocate(problem)` returns a `Dict{Symbol, Any}` store. `obj!`, `grad!`, `objgrad!`, and `hess!` write into the store and increment `problem.counters`. |
| OptimizationMethods.jl | Implements optimization methods that work with any `OptimizationProblem`. |

The guiding principle comes from OptimizationProblems.jl: **structs fix things, and the
store holds what changes**.

- A **problem struct** fixes the optimization problem.
- A **method struct** fixes the method's hyperparameters.
- **Spec structs** fix what is recorded (`History`) and when to stop (`StoppingCriteria`).
- The **store**, a `Dict{Symbol, Any}`, holds everything that changes during a run. That means
  the problem's evaluations, the method's state, and, optionally, the recorded history.

### Why change

In v0.1.2, each method is a single mutable struct that mixes three kinds of data:

- hyperparameters (e.g., `step_size`, `δ`, `ρ`);
- algorithm state that changes during a run (e.g., `δk`, `τ_lower`, `B`, `y`, `z`);
- research history (`iter_hist`, `grad_val_hist`, `stop_iteration`), preallocated for
  `max_iterations + 1` iterations.

Three other things also stand out:

- The initial iterate is stored inside the struct.
- The stopping rules are hard-coded: a gradient norm threshold and a maximum number of
  iterations.
- Every method has its own solver function (`fixed_step_gd`, `backtracking_gd`, ...).

The history is also used by some algorithms: `backtracking_gd` and `nonsequential_armijo_gd`
read `iter_hist[iter]` to get the previous iterate. As a result, history cannot be switched off
for production use.

## Design Principles

1. **A method is an immutable struct of hyperparameters.** For example,
   `struct FixedStepGD{T} <: AbstractMethod; step_size::T; end`. Behaviour is attached through
   multiple dispatch on the method type.
2. **There is one shared store.** The problem's `allocate(problem)` creates the store, and the
   method's `allocate!(store, method, problem; x0, history)` adds the method's keys to it.
   Anything the algorithm carries from one iteration to the next (e.g., `:x`, `:x_prev`,
   `:grad_norm`) is a top-level key.
3. **Algorithms never read history.** Anything an algorithm needs, such as the previous
   iterate, is stored explicitly in the store.
4. **History is a spec plus records.** An immutable `History` spec says which top-level store
   keys to record and how often. The records go into `store[:history]`, whose keys mirror the
   top-level keys (`store[:history][:x]`, `store[:history][:grad_norm]`). It also holds
   `store[:history][:iter]`, the iterations at which records were taken. Without a `History`
   spec there is no `:history` key, which is the production setting.
5. **History never triggers evaluations.** It only copies what is already in the store, so the
   problem's counters stay honest. If a method never computes a quantity, it is computed after
   the run from the recorded iterates. For example, fixed step-size gradient descent never
   evaluates the objective.
6. **One generic driver runs every method.** `optimize!` owns the loop: it checks the stopping
   criteria, records history, and increments `store[:iter]`. Each method implements
   `allocate!`, `initialize!`, and `step!`, where `step!` performs one outer iteration. A method
   may still define its own `optimize!` when a bespoke loop reads better.
7. **Stopping is a spec.** An immutable `StoppingCriteria` is passed to `optimize!`. It can use
   the gradient norm, the iteration count, and evaluation budgets read from the counters. The
   driver writes the reason for stopping to `store[:stop_reason]`.
8. **The store gives access to the counters.** For now, `store[:counters]` is an alias of
   `problem.counters` (the same object), so `History` and `StoppingCriteria` only need the store.
   Whether counters should move into the store in OptimizationModels.jl and
   OptimizationProblems.jl is an open question (O2).
9. **For now, each method is self-contained.** Each method is its own struct, and shared pieces
   such as line searches are helper functions that act on the store. Composable step-size and
   direction components will be reconsidered after the full-batch methods are ported (O3).
10. **The redesign is a clean break.** After the MWE, the v0.1.x methods, the bundled problems,
    and the NLPModels.jl dependency are removed. Methods are then added back one at a time. The
    old code remains available at tag
    [`v0.1.2`](https://github.com/numoptim/OptimizationMethods.jl/tree/v0.1.2).

## Target API

The following shows the intended usage. It is a sketch for the MWE to validate, not a
specification.

```julia
using OptimizationProblems, OptimizationMethods

problem = LogisticRegression(Float64; num_param=10, num_obs=100)
method  = FixedStepGD(step_size=1e-2)
hist    = History(keys=[:x, :grad_norm, :counters], every=1)
stop    = StoppingCriteria(grad_norm=1e-6, max_iterations=1_000)

store = allocate(problem)                                   # :obj, :grad
allocate!(store, method, problem; x0=zeros(10), history=hist)
optimize!(method, problem, store; stop)

store[:x]                    # terminal iterate
store[:stop_reason]          # e.g., :grad_norm or :max_iterations
store[:history][:iter]       # iterations at which records were taken
store[:history][:grad_norm]  # gradient norms at those iterations
```

### Generic pieces (written once)

| Name | Kind | Purpose |
|:--|:--|:--|
| `AbstractMethod` | abstract type | Supertype of all method structs. |
| `History` | immutable struct | Which keys to record (`keys`) and how often (`every`). |
| `StoppingCriteria` | immutable struct | Gradient norm tolerance, maximum iterations, and evaluation budgets. |
| `allocate_common!(store, problem; x0, history)` | function | Adds the keys every method needs (see the table of store keys). |
| `optimize!(method, problem, store; stop)` | function | Runs `initialize!`, then calls `step!` until the run stops; records history along the way. |
| `record!(store)` | function | Copies the keys named by the `History` spec into `store[:history]`. Does nothing without a spec. |
| `check_stop(stop, store)` | function | Returns `:none`, or the reason for stopping. |

### Per-method pieces

| Name | Purpose |
|:--|:--|
| `allocate!(store, method, problem; x0, history=nothing)` | Calls `allocate_common!`, then adds the method's own keys. |
| `initialize!(method, problem, store)` | Does the work for iteration zero, e.g., the gradient at `x0`. |
| `step!(method, problem, store)` | Performs one outer iteration. |

### Sketches

A sketch of the driver:

```julia
function optimize!(method::AbstractMethod, problem, store; stop::StoppingCriteria)
    initialize!(method, problem, store)
    record!(store)
    store[:stop_reason] = check_stop(stop, store)
    while store[:stop_reason] == :none
        step!(method, problem, store)  # may set store[:stop_reason] itself, e.g., on failure
        store[:iter] += 1
        record!(store)
        if store[:stop_reason] == :none
            store[:stop_reason] = check_stop(stop, store)
        end
    end
    return store[:x]
end
```

A sketch of a method:

```julia
struct FixedStepGD{T} <: AbstractMethod
    step_size::T
end
FixedStepGD(; step_size) = FixedStepGD(step_size)

function allocate!(store, method::FixedStepGD, problem; x0, history=nothing)
    allocate_common!(store, problem; x0, history)
    return store
end

function initialize!(method::FixedStepGD, problem, store)
    grad!(problem; store, x=store[:x])
    store[:grad_norm] = norm(store[:grad])
    return nothing
end

function step!(method::FixedStepGD, problem, store)
    x = store[:x]
    x .-= method.step_size .* store[:grad]
    grad!(problem; store, x)
    store[:grad_norm] = norm(store[:grad])
    return nothing
end
```

The MWE settles the remaining details, such as how the final iteration is always recorded
when `every > 1`.

### Store keys (proposal, to be validated by the MWE)

| Owner | Key | Meaning |
|:--|:--|:--|
| Problem (`allocate`) | `:obj`, `:grad` (and `:hess`, ..., if requested) | The most recent evaluation. After a line search it may be at a trial point rather than at `:x`. |
| Common (`allocate_common!`, driver) | `:iter` | The outer iteration count; `0` after `initialize!`. |
| | `:stop_reason` | `:none` until the run stops. Afterwards: `:grad_norm`, `:max_iterations`, `:max_grad_evals`, or a method-specific reason such as `:line_search_failed`. |
| | `:counters` | An alias of `problem.counters`. |
| | `:history`, `:history_spec` | The recorded data and the `History` spec. Present only when a spec is given. |
| Method (names shared by all methods) | `:x` | The current iterate. |
| | `:grad_norm` | The norm of the gradient at `:x`. Read by `StoppingCriteria`. |
| Method (specific to one method; examples) | `:x_prev`, `:obj_x`, `:step_size`, `:y`, `:z`, `:B`, `:ls_iterations` | Whatever state the method carries between iterations. |

Proposed invariant: after `initialize!` and after each `step!`, `store[:x]` is the current
iterate and `store[:grad_norm]` is the norm of the gradient there. `store[:grad]` is not
guaranteed to be the gradient at `store[:x]`; Nesterov's method, for example, evaluates the
gradient at `y`.

## Phase 1: Minimal Working Example (MWE)

### Goal

Before any of the design is built into `src/`:

- show that the shared-store design can express a representative set of methods;
- settle the conventions for store keys;
- measure the overhead of the design.

### Scope and location

- **Location.** Everything lives in a standalone folder, `dev/mwe/`, with its own
  `Project.toml` and a module named `MWE`. The MWE does not touch `src/` or `test/`, so names
  such as `FixedStepGD` do not collide with the v0.1.x code.
- **Dependencies.** OptimizationModels, OptimizationProblems, LinearAlgebra, BenchmarkTools,
  and Test. OptimizationModels and OptimizationProblems are installed from the NumOptim
  registry: `] registry add https://github.com/numoptim/NumOptimRegistry`.
- **Where `allocate`, `obj!`, and `grad!` come from.** The MWE takes them from
  OptimizationProblems (see O1).
- **Tests.** The MWE's tests are run manually and are not part of CI.
- **Suggested layout.** To be confirmed at Gate 1 of `/incremental-dev`:

```
dev/mwe/
├── Project.toml
├── benchmark.jl
├── src/
│   ├── MWE.jl                      # module and exports
│   ├── core.jl                     # AbstractMethod, History, StoppingCriteria,
│   │                               # allocate_common!, check_stop, record!, optimize!
│   ├── line_search.jl              # backtracking! acting on the store
│   ├── fixed_step_gd.jl
│   ├── backtracking_gd.jl
│   └── nesterov_accelerated_gd.jl
└── test/
    ├── runtests.jl
    └── ...                         # one test file per source file
```

### Methods and what each one tests

| Method | v0.1.2 reference | What it tests |
|:--|:--|:--|
| `FixedStepGD` | [`gd_fixed.jl`](https://github.com/numoptim/OptimizationMethods.jl/blob/v0.1.2/src/methods/gd_fixed.jl) | The simplest `step!`. It is the reference point for the overhead benchmarks. |
| `BacktrackingGD` | [`gd_backtracking.jl`](https://github.com/numoptim/OptimizationMethods.jl/blob/v0.1.2/src/methods/gd_backtracking.jl), [`backtracking.jl`](https://github.com/numoptim/OptimizationMethods.jl/blob/v0.1.2/src/methods/line_search_helpers/backtracking.jl) | `obj!` at trial points overwrites `store[:obj]`, so the objective at the current iterate needs its own key (`:obj_x`). The method needs `:x_prev`. A failed line search must revert `:x` and set `store[:stop_reason] = :line_search_failed`. |
| `NesterovAcceleratedGD` | [`gd_nesterov_accelerated.jl`](https://github.com/numoptim/OptimizationMethods.jl/blob/v0.1.2/src/methods/gd_nesterov_accelerated.jl) | Extra vector and scalar state (`:y`, `:z`, `:B`). The gradient is evaluated at `y`, but `:grad_norm` must describe `x`, so there are two gradient evaluations per iteration, as in v0.1.2. |

### Tests

For each method:

- After `k` iterations, the iterates match a reference loop, written in the test, that calls
  `grad!` (and `obj!`) directly.
- After a run, the counters match the expected number of evaluations. For example,
  `FixedStepGD` run for `k` iterations makes `k + 1` full gradient evaluations.

For the generic pieces (these can be tested with a small dummy method defined in the test
file):

- `History` records the right keys at the right iterations, for both `every = 1` and
  `every > 1`.
- The initial and final iterations are always recorded.
- Recorded arrays and counters are copies, not aliases.
- Without a `History` spec, the store has no `:history` key.
- `StoppingCriteria` triggers each stop reason (`:grad_norm`, `:max_iterations`,
  `:max_grad_evals`) at the right iteration.

For `BacktrackingGD`:

- On the failure path, `:x` is reverted and `:stop_reason` is set. One way to trigger it is
  `max_ls_iterations = 0` with a step size that is too large.

### Benchmark

`benchmark.jl` runs `FixedStepGD` on logistic regression at two sizes:

- small: `num_param = 10`, `num_obs = 100`, where overhead dominates;
- large: `num_param = 1_000`, `num_obs = 10_000`, where the cost of evaluations dominates.

At each size it compares four setups:

1. a hand-written loop that calls `grad!` directly (the baseline);
2. `optimize!` without a `History` spec;
3. `optimize!` with `History(keys=[:x, :grad_norm])`;
4. setup 2 with and without a function barrier in `step!`, i.e., unpacking the store into
   local variables and passing them to an inner function whose argument types are known to
   the compiler.

It reports the time and the allocations per iteration, measured with BenchmarkTools. The
resulting table is copied into `findings.md`.

### Questions the MWE must answer

Record each answer, with a recommendation, in `docs/src/dev/mwe/findings.md`.

- **Q1. Key conventions.** Are the proposed keys and invariants enough for the three methods?
  Where do problem keys and method keys collide (e.g., `:obj` being overwritten during a line
  search), and what convention resolves the collision?
- **Q2. Overhead.** What do the `Dict{Symbol, Any}` store and the generic driver cost relative
  to the hand-written loop, at both sizes? Are function barriers needed, and if so, where?
- **Q3. History mechanics.**
  - Should records grow with `push!`, or be preallocated?
  - Where does the spec live: `store[:history_spec]` or somewhere else?
  - How are mutable values snapshotted (copying arrays, `deepcopy` of the counters)?
  - How is the final iteration always recorded when `every > 1`?
- **Q4. Driver fit.** Did all three methods fit `allocate!`, `initialize!`, and `step!`
  naturally, or did any need its own `optimize!`?
- **Q5. Failure signalling.** Is setting `store[:stop_reason]` inside `step!` a clean way for a
  method to stop the driver? What should `:iter` and `:x` be after a failed step?
- **Q6. Element types.** How do three things interact: the type of the method's
  hyperparameters, `allocate(problem; type=T)`, and OptimizationProblems' requirement that
  `x::Vector{T}`? Should method structs be parametric in `T`?
- **Q7. Friction between packages.** What was awkward about each of the following? The answers
  feed O1 and O2.
  - having both `allocate` (for the problem) and `allocate!` (for the method);
  - depending on OptimizationProblems for `grad!`;
  - aliasing `problem.counters` into the store.

### Deliverables

- `dev/mwe/`: source code, tests, and `benchmark.jl`.
- `docs/src/dev/mwe/plan.md`: the `/incremental-dev` plan for this phase, written at Gate 1 and
  added to the documentation navigation under Development.
- `docs/src/dev/mwe/findings.md`: the answers to Q1–Q7, the benchmark table, and the
  recommendations.
- The Decisions Log on this page, updated after the findings are reviewed.

### Acceptance criteria

- `julia --project=dev/mwe -e 'using Pkg; Pkg.instantiate()'` succeeds, and
  `julia --project=dev/mwe dev/mwe/test/runtests.jl` passes.
- `julia --project=dev/mwe dev/mwe/benchmark.jl` produces the comparison table, and the table
  is in `findings.md`.
- Each of Q1–Q7 has an answer and a recommendation, and they have been reviewed with Vivak.

### How to start

Run `/incremental-dev` for "Phase 1: MWE" and point it at this section. Gate 1 produces
`docs/src/dev/mwe/plan.md`. A sensible order of work:

1. `core.jl`: `History`, `StoppingCriteria`, `allocate_common!`, `check_stop`, `record!`,
   `optimize!`;
2. `FixedStepGD`;
3. `benchmark.jl` for `FixedStepGD`;
4. `line_search.jl` and `BacktrackingGD`;
5. `NesterovAcceleratedGD`;
6. `findings.md`.

## Later Phases (sketch)

Each of these phases is planned in detail only once the previous phase is complete.

- **Phase 2: Decisions spanning the packages.** Resolve O1 and O2 using the MWE findings. If needed,
   open pull requests against OptimizationModels.jl and OptimizationProblems.jl.
- **Phase 3: Clean break.** Remove the v0.1.x methods, the bundled problems, the examples, and their
   tests. Also remove the dependencies that only they need: NLPModels, QuadGK, Distributions,
   and CircularArrays. CircularArrays can be added back if the nonmonotone line search still
   needs it. Rewrite `CLAUDE.md` for the new architecture.
- **Phase 4: Core infrastructure.** Move the MWE's generic pieces into `src/`, with full docstrings and
   tests, and delete `dev/mwe/`.
- **Phase 5: Full-batch first-order methods.** Port the nine v0.1.2 gradient descent methods, one per
   increment, with shared helpers (backtracking, the non-sequential Armijo condition,
   diminishing step sizes). Release v0.2.0.
- **Phase 6: Composition.** Revisit composable step-size, line search, and direction components (O3).
- **Phase 7 and later: Further method families.** Stochastic, variance-reduced, block/coordinate, second-order
   and quasi-Newton, and proximal methods. Also the documentation manual and examples built on
   OptimizationProblems.jl (see the roadmap in the README).

## Open Questions

- **O1. Where the interface functions are declared.** Should the generic functions `allocate`,
  `obj!`, `grad!`, `objgrad!`, and `hess!` be declared in OptimizationModels.jl? In that case
  OptimizationProblems.jl extends them, and OptimizationMethods.jl depends only on
  OptimizationModels.jl. The alternative is to leave them in OptimizationProblems.jl. To be
  decided in Phase 2.
- **O2. Where the counters live.** One option is to keep `counters` in the problem struct and
  alias it into the store. The other is to move the counters into the store upstream, so that
  `allocate(problem)` creates `store[:counters]` and the problem struct truly fixes the
  problem. To be decided in Phase 2.
- **O3. Composition.** Should step-size rules, line searches, and directions become composable
  components? To be decided in Phase 6.
- **O4. Stochastic methods.** Where do the batch indices and the random number generator live:
  in store keys or in method fields? For history and stopping, does an "iteration" mean one
  step or one epoch? To be decided before the first stochastic method.
- **O5. Quasi-likelihood problems.** After the clean break, the v0.1.x quasi-likelihood problems
  are unavailable until OptimizationProblems.jl releases the Wedderburn models from its
  `wedderburn` branch.

## Out of Scope / Potential Issues

These were noticed while planning but are not part of any phase yet. They are candidates for
issues.

- `test/methods/gd_backtracking.jl` exists but is not listed in `test/test.txt`, so it never
  runs.
- OptimizationModels.jl's documentation describes `obj(x; problem, store, batch)` and a field
  named `num_params`. The code uses `obj!(problem; store, x, ...)` (in OptimizationProblems.jl)
  and `num_param`.
- OptimizationModels.jl's `validate_supertype` requires `supertype(T) == OptimizationProblem`
  exactly, which rules out intermediate abstract problem types.
- If OptimizationProblems.jl and OptimizationMethods.jl both export a function named
  `allocate`, loading both with `using` makes the name ambiguous (relevant to O1).
- Access to a `Dict{Symbol, Any}` is type-unstable, so hot loops need function barriers
  (MWE Q2).

## Decisions Log

- **2026-09-29.**
  - Adopted Design Principles 1–10.
  - Limited the MWE to the shared-store design, with `FixedStepGD`, `BacktrackingGD`, and
    `NesterovAcceleratedGD`, in `dev/mwe/`.
  - Deferred O1 (where the interface functions are declared) and O2 (where the counters live)
    until after the MWE.
  - Planned only Phase 1 in detail.
