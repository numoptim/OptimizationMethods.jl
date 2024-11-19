# Date: 11/18/2024
# Author: Christian Varner
# Purpose: Implement Nesterov's Accelerated Gradient Descent

"""
    NesterovAcceleratedGD{T} <: AbstractOptimizerData

A structure that represents Nesterov Accelerated Gradient Descent. Stores
variables related to the method, and tracks quantities as the algorithm
progresses.

# Fields

- `name:String`, name of the solver for reference.
- `step_size::T`, step size used in the method. 
- `z::Vector{T}`, buffer array used for acceleration
- `y::Vector{T}`, buffer array used for acceleration
- `A::T`, term used for acceleration
- `A_prev::T`, term used for acceleration
- `B::T`, term used for acceleration
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

# Constructors
    
        NesterovAcceleratedGD(::Type{T}; x0::Vector{T}, step_size::T,
            threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `step_size::T`, step size used in the method. 
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct NesterovAcceleratedGD{T} <: AbstractOptimizerData{T}
    name::String
    step_size::T
    z::Vector{T}
    y::Vector{T}
    A::T
    A_prev::T
    B::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64

    # inner constructor
    NesterovAcceleratedGD{T}(name, step_size, z, y, A, A_prev, B, threshold, 
    max_iterations, iter_hist, grad_val_hist, stop_iteration) where {T} = 
    begin

        # test invariants
        @assert step_size > 0 "step size is non-positive."
        @assert threshold > 0 "`threshold` is non-positive."
        @assert max_iterations >= 0 "`max_iterations` is negative"
        @assert length(grad_val_hist) == max_iterations + 1 "`grad_val_hist` 
        should be of $(max_iterations+1) not $(length(grad_val_hist))"
        @assert length(iter_hist) == max_iterations + 1 "`iter_hist` should
        be of length $(max_iterations+1) not $(length(iter_hist))"
        @assert length(z) == length(y) "Length of z and y need to be equal"

        # return new object
        return new(name, step_size, z, y, A, A_prev, B, threshold, 
        max_iterations, iter_hist, grad_val_hist, stop_iteration)
    end
end
function NesterovAcceleratedGD(
    ::Type{T};
    x0::Vector{T},
    step_size::T,
    threshold::T,
    max_iterations::Int64,
) where {T}

    # name
    name::String = "Nesterov Accelerated Gradient Descent"

    d = length(x0)

    # initialization of buffers for the algorithm
    z = Vector{T}(undef, d)
    y = Vector{T}(under, d)

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # initialize the structure
    return NesterovAcceleratedGD{T}(name, step_size, z, y, T(0), T(0), T(0), 
    threshold, max_iterations, iter_hist, grad_val_hist, stop_iteration)
end

"""
    nesterov_accelerated_gd(optData::NesterovAcceleratedGD{T}, progData::P
        where P <: AbstractNLPModel{T, S}) where {T, S}

Implements Nesterov's Accelerated Gradient Descent as specified by `optData` on
the problem specified by `progData`.

# Reference(s)

The specific algorithm implementation follows the psuedo-code in

Li et. al. "Convex and Non-convex Optimization Under Generalized Smoothness".
arxiv, https://arxiv.org/abs/2306.01264.

# Method

TODO: finish documentation

# Arguments

- `optData::NesterovAcceleratedGD{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function nesterov_accelerated_gd(
    optData::NesterovAcceleratedGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data and save initial values
    precomp, store = OptimizationMethods.initialize(progData)

    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, x)

    grad_norm = norm(store.grad)
    optData.grad_val_hist[1] = grad_norm

    # precompute terms for the method
    step_size = optData.step_size        
    optData.B = 1                        # B_{iter+1}
    optData.A = optData.B + 1/step_size  # A_{iter+1}
    optData.A_prev = 1/step_size         # A_{iter}
    optData.y .= copy(x)                 # y_{iter}
    optData.z .= copy(x)                 # z_{iter}

    while (iter < optData.max_iterations) && (grad_norm > optData.threshold)
        iter += 1

        # take step with y
        x .= optData.y - step_size * store.grad

        # update variables used in nesterov accelerated method
        optData.z .-= step_size * (optData.A - optData.A_prev) * store.grad
        optData.B += .5 * (1 + sqrt(4 * optData.B + 1))
        optData.A_prev = optData.A
        optData.A = optData.B + 1 / step_size
        optData.y .= x + (1 - (optData.A_prev / optData.A)) .* (optData.z - x)

        # update histories
        OptimizationMethods.grad!(progData, precomp, store, x)
        grad_norm = norm(store.grad)

        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = grad_norm

        # compute the gradient at y
        OptimizationMethods.grad!(progData, precomp, store, optData.y)
    end

    optData.stop_iteration = iter

    return x

end