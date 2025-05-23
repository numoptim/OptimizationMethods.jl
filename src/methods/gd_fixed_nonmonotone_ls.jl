# Date: 01/27/2025
# Author: Christian Varner
# Purpose: Implement non-monotone line search scheme
# using the maximum value of the past M objective values

################################################################################
# Notes for later development:
#   1) Methods that use backtracking.jl essentially only deviate in
#   in the step_direction and reference value, therefore it might be nice
#   for later development to make a larger struct that encompasses
#   all these algorithms.
#
#   2) Currently this struct is only for fixed step sizes. In the future
#   it would be nice to separate out the functionality of step sizes from
#   the optimization loop so that a separate struct doesn't have to be made
#   for non fixed step size nls
#
#   3) The fields that handle NLS shoudl probably ideally be a struct
#   so that we can interchange different reference value computations.
#   E.g., instead of the maximum, take the average value.
################################################################################

"""
    FixedStepNonmonLSMaxValGD{T} <: AbstractOptimizerData{T}

Mutable `struct` that represents gradient descent using non-monotone line search
where the initial step size for line search is fixed. This `struct` also keeps
track of values during the optimization procedure implemented in 
`fixed_step_nls_maxval_gd`.

# Fields

- `name::String`, name of the solver for reference.
- `α::T`, the initial step size for line search.     
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference objective value for the line search criterion.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `objective_hist::CircularVector{T, Vector{T}}`, buffer array of size 
    `window_size` that stores `window_size` previous objective values.
- `max_value::T`, maximum value of `objective_hist`. This is the reference 
    objective value used in the line search procedure.
- `max_index::Int64`, index of the maximum value that corresponds to the 
    reference objective value.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    correpsonds to the gradient norm at iteration `k`.
- `stop_iteration::Int64`, the iteration number that the solver stopped on.
    The terminal iterate is saved at `iter_hist[stop_iteration+1]`.

# Constructors

    FixedStepNonmonLSMaxValG(::Type{T}; x0::Vector{T}, α::T, δ::T, ρ::T,
        window_size::Int64, line_search_max_iteration::Int64,
        threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `α::T`, the initial step size for line search.     
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correpsond to stricter descetn conditions, and
    smaller values correspond to looser descent conditions.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference value for the line search criterion.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct FixedStepNonmonLSMaxValG{T} <: AbstractOptimizerData{T}
    name::String
    α::T                                        
    δ::T
    ρ::T
    window_size::Int64                         
    line_search_max_iteration::Int64
    objective_hist::CircularVector{T, Vector{T}}                   
    max_value::T                                
    max_index::Int64                            
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64

    # inner constructor
    FixedStepNonmonLSMaxValG{T}(name, α, δ, ρ, window_size, line_search_max_iteration,
        threshold, max_iterations, iter_hist, grad_val_hist, 
        stop_iteration) where {T} = 
        begin
            new(name, α, δ, ρ, window_size, line_search_max_iteration, 
                CircularVector(zeros(T, window_size)), 
                T(0.0), -1, threshold,
                max_iterations, iter_hist, grad_val_hist, stop_iteration)
        end
end
function FixedStepNonmonLSMaxValG(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    ρ::T,
    window_size::Int64,
    line_search_max_iteration::Int64,
    threshold::T,
    max_iterations::Int64,
) where {T}

    # error checking
    @assert window_size > 0 "The number of objective values considered"*
    " in the non-monotone line search has to be greater than zero."

    d = length(x0)

    # initialization iterate history
    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 
        1:max_iterations + 1]
    iter_hist[1] = x0

    # initialization of gradient and dummy value for stop_iteration
    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    # name of the optimizer for reference
    name = "Gradient Descent with non-monotone line search using the max value"*
    " of the previous $(window_size) values" 

    return FixedStepNonmonLSMaxValG{T}(name, α, δ, ρ, window_size,
        line_search_max_iteration, threshold, max_iterations, 
        iter_hist, grad_val_hist, stop_iteration)
end

"""
    fixed_step_nls_maxval_gd(optData::FixedStepNonmonLSMaxValG{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with non-monotone line search using
the maximum value of a fixed number of previous objective values on
an optimization problem specified by `progData`. This implementation 
initializes the line search procedure with `optData.α` every iteration.

# Reference(s)

[Grippo, L., et al. "A Nonmonotone Line Search Technique for Newton's Method."
    SIAM Journal on Numerical Analysis, vol. 23, no. 4, 1886, pp. 707-16.
    JSTOR, http://www.jstor.org/stable/2157617.](@cite grippo1986Nonmonotone)

# Method

Let ``\\theta_{k-1}`` be the current iterate, and let 
``\\alpha \\in \\mathbb{R}_{>0}``, ``\\delta \\in (0, 1)``, and
``\\rho \\in (0, 1)``. The ``k^{th}`` iterate is generated as
``\\theta_k = \\theta_{k-1} - \\delta^t\\alpha \\dot F(\\theta_{k-1})`` 
where ``t + 1 \\in \\mathbb{N}`` is the smallest such number satisfying

```math
    F(\\theta_k) \\leq \\max_{\\max(0, k-M) \\leq j < k} F(\\theta_{k-1}) - 
    \\rho\\delta^t\\alpha||\\dot F(\\theta_{k-1})||_2^2,
```

where ``||\\cdot||_2`` is the L2-norm, and ``M \\in \\mathbb{N}_{>0}``.

# Arguments

- `optData::FixedStepNonmonLSMaxValG{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function fixed_step_nls_maxval_gd(
    optData::FixedStepNonmonLSMaxValG{T},
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

    # Update the objective cache
    optData.max_value = F(x)
    optData.objective_hist[1] = optData.max_value
    optData.max_index = 1

    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        iter += 1

        # backtracking
        success = OptimizationMethods.backtracking!(x, optData.iter_hist[iter], 
            F, store.grad, optData.grad_val_hist[iter] ^ 2, optData.max_value,
            optData.α, optData.δ, optData.ρ; 
            max_iteration = optData.line_search_max_iteration)

        # if backtracking is not successful return the previous point
        if !success
            optData.stop_iteration = (iter - 1)
            return optData.iter_hist[iter]
        end

        # compute the next gradient value
        OptimizationMethods.grad!(progData, precomp, store, x)

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # update the objective cache 
        F_x = F(x)
        optData.objective_hist[iter + 1] = F_x
        if ((iter) % optData.window_size) + 1 == optData.max_index
            optData.max_value, optData.max_index = 
            findmax(optData.objective_hist)
        end
    end

    optData.stop_iteration = iter

    return x
end