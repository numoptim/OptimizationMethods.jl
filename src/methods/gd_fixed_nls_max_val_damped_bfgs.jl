# Date: 2025/04/22
# Author: Christian Varner
# Purpose: Implementation of gradient descent with limited
# memory damped lbfgs

"""
    FixedDampedBFGSNLSMaxValGD{T} <: AbstractOptimizerData{T}

Mutable structure that parameterizes and stores values during the optimization
    routine for damped BFGS with non-monotone line search
    implemented with the maximum value of previous objective functions and 
    a fixed step size initialization.

# Fields

- `name::String`, name of the solver for reference.
- `c::T`, initial factor used in the approximation of the Hessian.
- `β::T`, shift applied to the approximation of the hessian to ensure it is
    bounded away from zero.
- `B::Matrix{T}`, BFGS approximation to the hessian.
- `δB::Matrix{T}`, buffer matrix for the update to the BFGS approximation.
- `r::Vector{T}`, buffer vector for the update vector fo the BFGS approximation.
- `s::Vector{T}`, buffer vector for the difference between consecutive iterates
    that is used in the BFGS update.
- `y::Vector{T}`, buffer vector for the difference between consecutive gradient
    values that is used in the BFGS update.
- `α::T`, the initial step size for line search. 
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `step::Vector{T}`, buffer array for the step direction used.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference objective value for the line search criterion.
- `objective_hist::CircularVector{T, Vector{T}}`, buffer array of size     
    `window_size` that stores `window_size` previous objective values.
- `max_value::T`, maximum value of `objective_hist`. This is the reference 
    objective value used in the line search procedure.
- `max_index::Int64`, index of the maximum value that corresponds to the 
    reference objective value.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
- `iter_hist::Vector{Vector{T}}`, store the iterate sequence as the algorithm 
    progresses. The initial iterate is stored in the first position.
- `grad_val_hist::Vector{T}`, stores the norm gradient values at each iterate. 
    The norm of the gradient evaluated at the initial iterate is stored in the 
    first position.
- `stop_iteration::Int64`, the iteration number the algorithm stopped on. The 
    iterate that induced stopping is saved at `iter_hist[stop_iteration + 1]`.

# Constructors

    FixedDampedBFGSNLSMaxValGD(::Type{T}; x0::Vector{T}, c::T, β::T, α::T,
        δ::T, ρ::T, line_search_max_iteration::Int64, window_size::Int64,
        threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial starting point for the optimization algorithm.
- `c::T`, initial factor used in the approximation of the Hessian.
- `β::T`, shift applied to the approximation of the hessian to ensure it is
    bounded away from zero.
- `α::T`, the initial step size for line search.   
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correpsond to stricter descetn conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference value for the line search criterion.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct FixedDampedBFGSNLSMaxValGD{T} <: AbstractOptimizerData{T}
    name::String
    # BFGS parameters
    c::T
    β::T
    B::Matrix{T}
    δB::Matrix{T}
    r::Vector{T}
    s::Vector{T}
    y::Vector{T}
    # parameters for line search
    α::T
    δ::T
    ρ::T
    line_search_max_iteration::Int64
    step::Vector{T}
    # parameters for non-monotone line search
    window_size::Int64
    objective_hist::CircularVector{T, Vector{T}}
    max_value::T
    max_index::Int64
    # default parameters
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function FixedDampedBFGSNLSMaxValGD(::Type{T};
    x0::Vector{T},
    c::T,
    β::T,
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iteration::Int64,
    window_size::Int64,
    threshold::T,
    max_iterations::Int64) where {T}

    # length for initialization purposes
    d = length(x0)

    # initialization iterate history
    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 
        1:max_iterations + 1]
    iter_hist[1] = x0

    # initialization of gradient and dummy value for stop_iteration
    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    # name of optimizer for reference
    name::String = ""
    if window_size == 1
        name = "Gradient Descent with line search using damped BFGS updates"
    else
        name = "Gradient Descent with (non-monotone) line search using"*
        " damped BFGS updates"
    end

    return FixedDampedBFGSNLSMaxValGD{T}(
        name,
        c,
        β,
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        α,
        δ,
        ρ,
        line_search_max_iteration,
        zeros(T, d),
        window_size, 
        CircularVector(zeros(T, window_size)),
        T(0),
        -1,
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    )
end

"""
    fixed_damped_bfgs_nls_maxval_gd(optData::FixedDampedBFGSNLSMaxValGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of a fixed step size, damped BFGS Quasi-Newton method, which
is globalized through a (non-)monotone Armijo line search scheme. The method
and its parameters are specified by `optData`, and the optimization routine
is applied to a problem specified by `progData`.

# Reference(s)

[Nocedal and Wright, "Numerical Optimization". Springer. 2nd Edition. 
    Chapter 6 and 18.](@cite nocedal2006Numerical)

# Method

The method we describe is a version of Algorithm 3.2 in the 
    [reference above](@cite nocedal2006Numerical).

Let ``k + 1 \\in \\mathbb{N}``, and ``\\theta_{k} \\in \\mathbb{R}^n``.
Let ``F(\\theta)`` be a function, and let ``\\dot F(\\theta)`` and
``\\ddot F(\\theta)`` be the gradient and hessian of ``F(\\theta)``,
respectively. Let ``\\alpha \\in \\mathbb{R}_{>0}``, ``\\delta \\in (0, 1)``,
``\\rho \\in (0, 1)`` be algorithmic parameters. 
Let ``B_0 \\in \\mathbb{R}^{n \\times n}`` be an initial Quasi-Newton matrix.
Then, iterate ``\\theta_{k+1}`` is produced by the following procedure.

```math
    \\theta_{k+1} = \\theta_{k} - \\delta^j \\alpha d_k, ~d_k \\in\\mathbb{R}^n
```

where ``j`` is found by a 
[backtracking procedure](@ref OptimizationMethods.backtracking!) using
``\\max(F(\\theta_{k}),...,F(\\theta_{\\max(0, k-M)}))`` for 
``M + 1 \\in \\mathbb{N}`` as a reference value.

The vector ``d_k \\in \\mathbb{R}^n`` is defined as the Quasi-Newton step using
the [damped BFGS method](@ref OptimizationMethods.update_bfgs!). In particular,
let ``B_k \\in \\mathbb{R}^{n \\times n}`` be the damped BFGS matrix, then

```math
    d_k = B_k^{-1} \\nabla F(\\theta_k).
```

# Arguments

- `optData::FixedDampedBFGSNLSMaxValGD{T}`, the specification for 
    the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function fixed_damped_bfgs_nls_maxval_gd(
    optData::FixedDampedBFGSNLSMaxValGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initializations
    precomp, store = initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)

    # Iteration 0
    iter = 0

    # Update iteration 
    x = copy(optData.iter_hist[iter + 1]) 
    grad!(progData, precomp, store, x)

    # Store Values
    optData.grad_val_hist[iter + 1] = norm(store.grad)

    # Initialize approximation
    fill!(optData.B, 0)
    OptimizationMethods.add_identity!(optData.B,
        optData.c * optData.grad_val_hist[iter + 1])

    # Update the objective cache
    optData.max_value = F(x)
    optData.objective_hist[iter + 1] = optData.max_value
    optData.max_index = iter + 1

    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # update the iteration number
        iter += 1

        # store values for update
        optData.s .= -x
        optData.y .= -store.grad

        # backtrack
        optData.step .= optData.B \ store.grad
        backtrack_success = OptimizationMethods.backtracking!(
            x,
            optData.iter_hist[iter],
            F,
            store.grad,
            optData.step,
            optData.max_value,
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration
        )

        # if backtracking is not successful, return the previous point
        if !backtrack_success
            optData.stop_iteration = (iter - 1)
            return optData.iter_hist[iter]
        end
        
        # compute the next gradient and hessian values
        OptimizationMethods.grad!(progData, precomp, store, x)

        # update approximation
        optData.s .+= x
        optData.y .+= store.grad
        update_success = OptimizationMethods.update_bfgs!(optData.B, 
            optData.r, optData.δB,
            optData.s, optData.y; damped_update = true)

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # update the objective cache for non-monotone line search
        F_x = F(x)
        optData.objective_hist[iter + 1] = F_x
        if (iter % optData.window_size) + 1 == optData.max_index
            optData.max_value, optData.max_index = 
                findmax(optData.objective_hist)
        end

    end

    optData.stop_iteration = iter

    return x
end