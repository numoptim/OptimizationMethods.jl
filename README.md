# OptimizationMethods

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://numoptim.github.io/OptimizationMethods.jl/dev/)
[![CI](https://github.com/numoptim/OptimizationMethods.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/numoptim/OptimizationMethods.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/numoptim/OptimizationMethods.jl/graph/badge.svg?token=CR7AFXRO0E)](https://codecov.io/gh/numoptim/OptimizationMethods.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

`OptimizationMethods.jl` is a research-tier library for the Julia language that
implements optimization methods with a focus on problems arising in data science.

## Installation

This package is registered in [NumOptimRegistry](https://github.com/numoptim/NumOptimRegistry).
Add the registry once per Julia installation, then install the package normally:

```julia
] registry add https://github.com/numoptim/NumOptimRegistry
] add OptimizationMethods
```

Alternatively, install directly from the repository URL without adding the registry:

```julia
] add https://github.com/numoptim/OptimizationMethods.jl
```

It is also possible to clone the repository into a local directory.
In that case, refer to [Julia Pkg instructions](
    https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project
).

## License

MIT License

## Roadmap

Items marked `[x]` are available in the current release.
Items marked `[ ]` are planned.

#### Interface & Architecture
- [ ] Migrate from [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) to [OptimizationModels.jl](https://github.com/numoptim/OptimizationModels.jl) interface
- [ ] `abstract type AbstractMode` with `struct Research <: AbstractMode` and `struct Execution <: AbstractMode` as concrete subtypes
- [ ] Mode encoded as a type parameter on all optimizer structs (e.g., `FixedStepGD{T, M<:AbstractMode}`)
- [ ] Optimizer structs carry `counters::Dict{Symbol, Counter}` for optimizer-specific tracking (gradient steps, objective evaluations, inner loop iterations)
- [ ] `allocate(optimizer, store::Dict; ...)` — adds optimizer-specific keys to an existing problem store
- [ ] `allocate(optimizer; n, ...)` — fresh dict for standalone or testing use
- [ ] `allocate(optimizer::.{T, Research}, ...)` — full iterate history, gradient norm history, and step-size history
- [ ] `allocate(optimizer::.{T, Execution}, ...)` — minimal scratch space only, no history overhead
- [ ] Remove bundled problem implementations; delegate to [OptimizationProblems.jl](https://github.com/numoptim/OptimizationProblems.jl)

#### Full-Batch First-Order Methods
- [x] Fixed step-size gradient descent
- [x] Diminishing step-size gradient descent
- [x] Barzilai-Borwein gradient descent
- [x] Lipschitz approximation gradient descent (Malitsky & Mishchenko)
- [x] Weighted norm damping gradient descent (WNGrad)
- [x] Nesterov accelerated gradient descent
- [x] Gradient descent with backtracking line search (Armijo)
- [x] Gradient descent with non-monotone line search
- [x] Gradient descent with non-sequential Armijo line search
- [ ] Heavy ball method
- [ ] Proximal gradient descent
- [ ] FISTA (accelerated proximal gradient for nonsmooth objectives)

#### Stochastic First-Order Methods
- [ ] SGD with fixed/diminishing step size
- [ ] Mini-batch gradient descent
- [ ] SAG / SAGA
- [ ] SVRG
- [ ] SARAH / SpiderBoost

#### Variance Reduction
- [ ] Katyusha
- [ ] L-SVRG / SARAH variants

#### Block / Coordinate Methods
- [ ] Randomized coordinate gradient descent
- [ ] Block gradient descent
- [ ] Stochastic block gradient descent
- [ ] Stochastic block with variance reduction

#### Second-Order and Quasi-Newton
- [ ] Newton's method
- [ ] L-BFGS

#### Documentation
- [ ] Manual: method guide with theory and usage notes per method family
- [ ] Examples updated to use [OptimizationProblems.jl](https://github.com/numoptim/OptimizationProblems.jl)
