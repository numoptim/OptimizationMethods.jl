# Date: 01/23/2025
# Author: Christian Varner
# Purpose: Implementation of gradient descent using backtrackings

"""
    BacktrackingGD{T} <: AbstractOptimizerData{T}

Mutable sturcture representing gradient descent using backtracking.
    It also stores and keeps track of values during the optimization
    routine.

# Fields

- `name:String`, name of the solver for reference.
- `α::T`, initial step size used for backtracking.
- `δ::T`, backtracking decreasing factor applied to the `α` when line
    search criterion not satisfied.
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum allowable iterations for the
    backtracking procedure.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
- `iter_diff::Vector{T}`, a buffer for storing differences between subsequent
    iterate values that are used for computing the step size
- `grad_diff::Vector{T}`, a buffer for storing differences between gradient 
    values at adjacent iterates, which is used to compute the step size
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    correpsonds to the gradient norm at iteration `k`.
- `stop_iteration::Int64`, the iteration number that the solver stopped on.
    The terminal iterate is saved at `iter_hist[stop_iteration+1]`.

!!! warning
    The method does not check if the line search procedure is successful
    or not. 

# Constructors

    BacktrackingGD(::Type{T}; x0::Vector{T}, α::T, δ::T, ρ::T,
        line_search_max_iteration::Int64, threshold::T, max_iteration::Int64)
        where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `α::T`, initial step size used for backtracking.
- `δ::T`, backtracking decreasing factor applied to the `α` when line
    search criterion not satisfied.
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum allowable iterations for the
    backtracking procedure.
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct BacktrackingGD{T} <: AbstractOptimizerData{T}
    name::String
    α::T
    δ::T
    ρ::T
    line_search_max_iteration::Int64
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function BacktrackingGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iteration::Int64,
    threshold::T,
    max_iteration::Int64,
) where {T}

    d = length(x0)

    # initialization iterate history
    iter_hist = Vector{T}[Vector{T}(under, d) for i in 
        1:max_iteration + 1]
    iter_hist[1] = x0

    # initialization of gradient and dummy value for stop_iteration
    grad_val_hist = Vector{T}(under, max_iteration + 1)
    stop_iteration = -1

    return BacktrackingGD("Gradient Descent with Backtracking",
        α, δ, ρ, line_search_max_iteration, threshold, 
        max_iteration, iter_hist, grad_val_hist, stop_iteration)
end

"""
    backtracking_gd(optData::BacktrackingGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

# Reference(s)

# Method

Let ``\\theta_{k-1}`` be the current iterate, and let 
``\\alpha \\in \\mathbb{R}_{>0}``, ``\\delta \\in (0, 1)``, and
``\\rho \\in (0, 1)``. The ``k^{th}`` iterate is generated as
``\\theta_k = \\theta_{k-1} - \\delta^t\\alpha \\dot F(\\theta_{k-1})`` 
where ``t + 1 \\in \\mathbb{N}`` is the smallest such number satisfying

```math
    F(\\theta_k) \\leq F(\\theta_k) - \\rho\\delta^t\\alpha
    ||\\dot F(\\theta_{k-1})||_2^2,
```

where ``||\\cdot||_2`` is the L2-norm. 

!!! note
    Theoretically, there exists such a ``t``, but it can be made
    arbitrarily large. Therefore, the line search procedure stops
    search for such a ``t`` after `optData.line_search_max_iteration`.
    The current implementation does not check if line search
    methodology terminates successful, and tries to continue
    regardless.

# Arguments

- `optData::BarzilaiBorweinGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function backtracking_gd(
    optData::BacktrackingGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initializations
    precomp, store = initialize(progData)

    # Iteration 0
    iter = 0

    # Update iteration 
    x = copy(optData.iter_hist[iter + 1]) 
    grad!(progData, precomp, store, x)

    # Store Values
    optData.grad_val_hist[iter + 1] = norm(store.grad)

    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        iter += 1

        # backtrack
        obj!(progData, precomp, store, x)
        OptimizationMethods.backtracking!(x, optData.iter_hist[iter],
            store.grad, store.grad, reference_value, 
            optData.α, optData.δ, optData.ρ; 
            max_iteration = optData.line_search_max_iteration)
        
        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end