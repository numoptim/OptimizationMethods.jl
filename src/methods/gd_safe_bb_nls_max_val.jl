# Date: 2025/04/07
# Author: Christian Varner
# Purpose: Implement a (non-)monotone line search method with
# a safe barzilai-borwein step size

"""
    SafeBarzilaiBorweinNLSMaxValGD{T} <: AbstractOptimizerData{T}

# Fields

- `name::String`, name of the solver for reference. 
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference objective value for the line search criterion.
- `objective_hist::CircularVector{T, Vector{T}}`, 
    buffer array of size `window_size` that stores `window_size` previous
    objective values.
- `max_value::T`, maximum value of `objective_hist`. This is the reference 
    objective value used in the line search procedure.
- `max_index::Int64`, index of the maximum value that corresponds to the 
    reference objective value.
- `init_stepsize::T`, initial step size to start the method. 
- `long_stepsize::Bool`, flag for step size; if true, use the long version of 
    Barzilai-Borwein. If false, use the short version. 
- `iter_diff::Vector{T}`, a buffer for storing differences between subsequent
    iterate values that are used for computing the step size
- `grad_diff::Vector{T}`, a buffer for storing differences between gradient 
    values at adjacent iterates, which is used to compute the step size
- `α_lower::T`, value that is used to safeguard too small and too large of an
    initial step size produced by Barzilai-Borwein
- `α_default::T`, if the step size produced by Barzilai-Borwein is outside
    the safeguarded region, then the initial step size for line search is
    set to this value.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    corresponds to the gradient norm at iteration `k`.
- `stop_iteration::Int64`, the iteration number that the solver stopped on.
    The terminal iterate is saved at `iter_hist[stop_iteration+1]`.

# Constructors

    SafeBarzilaiBorweinNLSMaxValGD(::Type{T}; x0::Vector{T}, δ::T, ρ::T,
        window_size::Int64, line_search_max_iteration::Int64, init_stepsize::T,
        long_stepsize::Bool, α_lower::T, α_default::T, threshold::T, 
        max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference value for the line search criterion.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `init_stepsize::T`, initial step size used for the first iteration. 
- `long_stepsize::Bool`, flag for step size; if true, use the long version of
    Barzilai-Borwein, if false, use the short version.
- `α_lower::T`, value that is used to safeguard too small and too large of an
    initial step size produced by Barzilai-Borwein
- `α_default::T`, if the step size produced by Barzilai-Borwein is outside
    the safeguarded region, then the initial step size for line search is
    set to this value. 
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct SafeBarzilaiBorweinNLSMaxValGD{T} <: AbstractOptimizerData{T}
    name::String
    δ::T
    ρ::T
    line_search_max_iteration::Int64
    window_size::Int64
    objective_hist::CircularVector{T, Vector{T}}
    max_value::T
    max_index::Int64
    init_stepsize::T
    long_stepsize::Bool
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    α_lower::T
    α_default::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function SafeBarzilaiBorweinNLSMaxValGD(::Type{T};
    x0::Vector{T},
    δ::T,
    ρ::T,
    window_size::Int64,
    line_search_max_iteration::Int64,
    init_stepsize::T,
    long_stepsize::Bool,
    α_lower::T,
    α_default::T,
    threshold::T, 
    max_iterations::Int64) where {T}

    # initial step size must be positive
    @assert init_stepsize > 0 "Initial step size must be a positive value."

    # window size must be positive
    @assert window_size >= 1 "$(window_size) needs to be a natural number."

    # create the name for the method
    name = "Safe Barzilai Borwein Gradient Descent with (Non)-monotone"*
        " line search"

    d = length(x0)

    iter_diff = zeros(T, d)
    grad_diff = zeros(T, d)
    objective_hist = CircularVector(zeros(T, window_size))
    max_value = T(0.0)
    max_index = -1

    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:max_iterations + 1]
    iter_hist[1] = x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    return SafeBarzilaiBorweinNLSMaxValGD{T}(name, δ, ρ, line_search_max_iteration, 
    window_size, objective_hist, max_value, max_index, init_stepsize, long_stepsize,
    iter_diff, grad_diff, α_lower, α_default, threshold, max_iterations, iter_hist,
    grad_val_hist, stop_iteration)
end

"""
    safe_barzilai_borwein_nls_maxval_gd(optData::SafeBarzilaiBorweinNLSMaxValGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with a non-monotone line search strategy
using the maximum value of a fixed number of previous objective values
on an optimization problem specified by `progData`. The method initializes
the line search routine with a safeguarded version of the Barzilai-Borwein
step size.

# Reference(s)

[Grippo, L and Lampariello, F and Lucidi, S. "A Nonmonotone Line Search
    Technique for Newton's Method". SIAM. 1986.](@cite grippo1986Nonmonotone)

[Raydan, Marcos. "The Barzilai and Borwein Gradient Method for the Large Scale
    Unconstrained Minimization Problem". SIAM Journal of Optimization. 
    1997.](@cite raydan1997Barzilai)

# Method

Let ``\\theta_{k-1}`` be the current iterate, and let 
``\\alpha_{k-1} \\in \\mathbb{R}_{>0}``, ``\\delta \\in (0, 1)``, and
``\\rho \\in (0, 1)``. The ``k^{th}`` iterate is generated as 
``\\theta_k = \\theta_{k-1} - \\delta^t\\alpha_{k-1} \\dot F(\\theta_{k-1})`` 
where ``t + 1 \\in \\mathbb{N}`` is the smallest such number satisfying

```math
    F(\\theta_k) \\leq \\max_{\\max(0, k-M) \\leq j < k} F(\\theta_{k-1}) - 
    \\rho\\delta^t\\alpha_{k-1}||\\dot F(\\theta_{k-1})||_2^2,
```

where ``||\\cdot||_2`` is the L2-norm, and ``M \\in \\mathbb{N}_{>0}``. The
initial step size ``\\alpha_{k-1}`` for line search is computed depending
on the iteration number and `optData.long_stepsize`.

## Long Step Size Version (if `optData.long_stepsize==true`)

If ``k=0``, then ``\\alpha_0`` is set to `optData.init_stepsize`. For ``k>0``,
let

```math 
\\beta_k = \\frac{ \\Vert x_k - x_{k-1} \\Vert_2^2}{(x_k - x_{k-1})^\\intercal 
    (\\dot f(x_k) - \\dot f(x_{k-1}))},
```

then ``\\alpha_k = \\beta_k`` when ``\\beta_k \\in`` `[optData.α_lower, 
1/optData.α_lower]`, otherwise ``\\alpha_k =`` `optData.α_default`.

## Short Step Size Version (if `optData.long_stepsize==false`)

If ``k=0``, then ``\\alpha_0`` is set to `optData.init_stepsize`. For ``k>0``,

```math
\\alpha_k = \\frac{(x_k - x_{k-1})^\\intercal (\\dot f(x_k) - 
    \\dot f(x_{k-1}))}{\\Vert \\dot f(x_k) - \\dot f(x_{k-1})\\Vert_2^2},
```

then ``\\alpha_k = \\beta_k`` when ``\\beta_k \\in`` `[optData.α_lower, 
1/optData.α_lower]`, otherwise ``\\alpha_k =`` `optData.α_default`.

# Arguments

- `optData::SafeBarzilaiBorweinNLSMaxValGD{T}`, the specification for the
    optimization method.
- `progData<:AbstractNLPModel{T, S}`, the specification for the optimization
    problem.
"""
function safe_barzilai_borwein_nls_maxval_gd(optData::SafeBarzilaiBorweinNLSMaxValGD{T},
    progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

    # initialize the problem
    precomp, store = initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)

    # Iteration 0
    iter = 0

    # Update iteration 
    x = copy(optData.iter_hist[iter + 1]) 
    grad!(progData, precomp, store, x)

    # Store Values
    optData.grad_val_hist[iter + 1] = norm(store.grad)
    step_size = optData.init_stepsize
    step_size_function = optData.long_stepsize ? 
        bb_long_step_size : bb_short_step_size

    # Initialize the objective storage
    optData.max_value = F(x)
    optData.objective_hist[iter + 1] = optData.max_value
    optData.max_index = iter + 1

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)
        
        iter += 1

        # changes to compute next step size
        optData.iter_diff .= -x
        optData.grad_diff .= -store.grad

        # backtracking
        success = OptimizationMethods.backtracking!(x, optData.iter_hist[iter],
            F, store.grad, optData.grad_val_hist[iter]^2, optData.max_value, 
            step_size, optData.δ, optData.ρ; 
            max_iteration = optData.line_search_max_iteration)

        # check if backtracking was successful
        if !success
            optData.stop_iteration = iter - 1
            return optData.iter_hist[iter]
        end

        # compute the next gradient value
        grad!(progData, precomp, store, x)

        # compute the next step size
        optData.iter_diff .+= x
        optData.grad_diff .+= store.grad
        step_size = step_size_function(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > 1/optData.α_lower
            step_size = optData.α_default
        end

        # compute the next reference value
        F_x = F(x)
        optData.objective_hist[iter + 1] = F_x
        if (iter % optData.window_size) + 1 == optData.max_index
            optData.max_value, optData.max_index = 
                findmax(optData.objective_hist)
        end

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end