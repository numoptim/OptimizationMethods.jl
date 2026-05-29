# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```
julia --project -e 'using Pkg; Pkg.test()'
```

**Run a subset of tests** by editing `test/test.txt` to include only the desired test file paths (relative to `test/`), then running the test suite. Each line is an `include`d file path.

**Build docs locally** (from `docs/` directory):
```
julia --project make.jl
cd build && python3 -m http.server
```

## Architecture

### Two-layer design
The package is split into **Problems** and **Methods**, which are independent and combined at runtime.

- **Problems** (`src/problems/`) implement `AbstractNLPModel{T,S}` from NLPModels.jl. Each problem file defines three things: the problem struct (e.g., `LeastSquares{T,S}`), a `Precompute` struct for expensive one-time computations (e.g., `PrecomputeLS` stores `A'A`, `A'b`), and an `Allocate` struct for pre-allocated working buffers (e.g., `AllocateLS` holds `grad`, `res`, `hess`).

- **Methods** (`src/methods/`) implement `AbstractOptimizerData{T}`. Each method file defines a mutable struct (e.g., `FixedStepGD{T}`) holding algorithm parameters plus full iterate history (`iter_hist::Vector{Vector{T}}`) and gradient norm history (`grad_val_hist::Vector{T}`), and a solver function (e.g., `fixed_step_gd`).

### Calling convention for problems
Each problem supports three levels of dispatch:
1. `obj(progData, x)` — no precomputation
2. `obj(progData, preComp, x)` — with precomputed values, no in-place storage
3. `obj!(progData, preComp, store, x)` — in-place, using preallocated storage

Solver functions always call `initialize(progData)` first to get `(precomp, store)`, then use the in-place variants throughout the iteration loop.

### Test wiring
`test/runtests.jl` reads `test/test.txt` line by line and `include`s each file. Adding or removing lines from `test.txt` controls which tests run without modifying `runtests.jl`.

### Helper modules
- `src/problems/regression_helpers/` — link functions, variance functions, and quasi-likelihood utilities shared across QL problem variants
- `src/methods/line_search_helpers/` — backtracking and non-sequential Armijo line search logic used by multiple GD methods
- `src/methods/stepsize_helpers/` — diminishing step size schedules

### Quasi-likelihood problems
The four `ql_logistic_*.jl` problems (`sin`, `centered_exp`, `centered_log`, `monomial`) all follow the same pattern: logistic link function with different variance functions. They inherit from `AbstractDefaultQL{T,S}` which provides shared struct fields and default implementations. The objective requires numerical integration (`QuadGK`); gradient and Hessian have closed forms.

### Code style
Follows [Blue style](https://github.com/invenia/BlueStyle). Docstrings are thorough and include math via ````math` blocks with `\\` escaping.
