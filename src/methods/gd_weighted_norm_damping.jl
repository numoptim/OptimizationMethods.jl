# Date: 11/14/2024
# Author: Christian Varner
# Purpose: Implement WNGrad

"""
    WeightedNormDampingGD{T} <: AbstractOptimizerData{T}

A mutable struct that represents gradient descent using the weighted-norm damping step size
It stores the specification for the method and records values during iteration.

# Fields

- `name::String`, name of the optimizer for recording purposes
- `init_norm_damping_factor::T`, initial damping factor. Inverse of the initial step size.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm at most
    `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not including the 
    initial iterate.
- `iter_hist::Vector{Vector{T}}`, store the iterate sequence as the algorithm progresses.
    The initial iterate is stored in the first position.
- `grad_val_hist::Vector{T}`, stores the norm gradient values at each iterate. The norm
    of the initial iterate is stored in the first position.
- `stop_iteration::Int64`, the iteration number the algorithm stopped on. The iterate
    that induced stopping is saved at `iter_hist[stop_iteration + 1]`.

# Constructors

    WeightedNormDampingGD(::Type{T}; x0::Vector{T}, init_norm_damping_factor::T, 
        threshold::T, max_iterations::Int64) where {T}

Constructs an instance of type `WeightedNormDampingGD{T}`.

## Arguments

- `T::DataType`, type for data and computation
- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `init_norm_damping_factor::T`, initial damping factor. Inverse of the initial step size.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm at most
    `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not including the 
    initial iterate.
"""
mutable struct WeightedNormDampingGD{T} <: AbstractOptimizerData{T}
    name::String
    init_norm_damping_factor::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64

    WeightedNormDampingGD{T}(name, init_norm_damping_factor, threshold, max_iterations,
    iter_hist, grad_val_hist, stop_iteration) where {T} =
    begin
        @assert init_norm_damping_factor > 0 "init_norm_damping_factor is non-zero or negative"
        @assert threshold > 0 "threshold is zero or negative"
        @assert max_iterations > 0 "max_iterations is zero or negative"
        return new(name, init_norm_damping_factor, threshold, max_iterations, iter_hist, 
            grad_val_hist, stop_iteration)
    end
end
function WeightedNormDampingGD(::Type{T};
    x0::Vector{T},
    init_norm_damping_factor::T,
    threshold::T,
    max_iterations::Int64) where {T}

    # name of optimizer
    name::String = "Gradient Descent with Weighted-Norm Damping Step Size"
    
    # initialize iter_hist and grad_val_hist
    d::Int64 = length(x0)
    iter_hist::Vector{Vector{T}} = 
        Vector{Vector{T}}([Vector{T}(under, d) for i in 1:(max_iterations + 1)])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1) 
    stop_iteration::Int64 = -1 ## dummy value

    # return objective
    return WeightedNormDampingGD{T}(name, init_norm_damping_factor, threshold, 
        max_iterations, iter_hist, grad_val_hist, stop_iteration)
end

"""
    weighted_norm_damping_gd(optData::WeightedNormDampingGD{T}, 
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Method that implements gradient descent with weighted norm damping step size using the
specifications in `optData` on the problem specified by `progData`.

# Method
Let ``\\theta_k`` be the ``k^{th}`` iterate, and ``\\alpha_k`` be the ``k^{th}`` step size.
The optimization method generate iterates following

```math
\\theta_{k + 1} = \\theta_{k} - \\alpha_k \\dot F(\\theta_k),
```

where ``\\dot F`` is the gradient of the objective function ``F``.

The step size depends on the iteration number ``k``. For ``k = 0``, the step size
is ``\\alpha_0 = 1/optData.init_norm_damping_factor``. For ``k > 0``, the step size
is iteratively updated as

```math
\\alpha_k = (1/\\alpha_{k-1} + ||\\dot F(\\theta_k)||_2^2 * \\alpha_{k-1})^{-1}.
```

For more information on the method, see the reference below.

# Reference

Wu, Xiaoxia et. al. "WNGrad: Learn the Learning Rate in Gradient Descent". arxiv, 
https://arxiv.org/abs/1803.02865

# Arguments

- `optData::WeightedNormDampingGD{T}`, specification for the optimization algorithm.
- `progData::P where P <: AbstractNLPModel{T, S}`, specification for the problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 
"""
function weighted_norm_damping_gd(
    optData::WeightedNormDampingGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem
    precomp, store = initialize(progData)

    # initialization of variables for optimization
    iter::Int64 = 0

    x::S = copy(optData.iter_hist[1])
    step_size::T = 1 / optData.init_norm_damping_factor

    grad!(progData, precomp, store, x)

    optData.grad_val_hist[1] = norm(store.grad)

    while (iter < optData.max_iterations) && (optData.grad_val_hist[iter + 1] > optData.threshold)
        iter += 1

        # take step
        x .-= step_size .* store.grad

        # update history
        grad!(progData, precomp, store, x)
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # compute the step size for the next iteration
        step_size = 1/((1 / step_size) + (optData.grad_val_hist[iter + 1]^2) * step_size)
    end

    optData.stop_iteration = iter

    return x
end