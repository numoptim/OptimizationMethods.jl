# Date: 11/18/2024
# Author: Christian Varner
# Purpose: Implement Nesterov's Accelerated Gradient Descent

"""
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

    NesterovAcceleratedGD{T}(name, step_size, z, y, A, A_prev, B, threshold, 
    max_iterations, iter_hist, grad_val_hist, stop_iteration) where {T} = 
    begin
        @assert step_size > 0 "step size is non-positive."
        @assert threshold > 0
        @assert max_iterations >= 0
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
    z = zeros(T, d)
    y = zeros(T, d)

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
"""
function nesterov_accelerated_gd(
    optData::AbstractOptimizerData,
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
    optData.B = 1
    optData.A = optData.B + 1/step_size
    optData.A_prev = 1/step_size
    optData.y .= copy(x)
    optData.z .= copy(x)

    while (iter < optData.max_iterations) && (grad_norm > optData.threshold)
        iter += 1

        # take step with y
        x .= optData.y - step_size * store.grad

        # update variables used in nesterov accelerated method
        optData.z .-= step_size * (optData.A - optData.A_prev) * store.grad
        optData.B += .5*(1+sqrt(4*optData.B + 1))
        optData.A_prev = optData.A
        optData.A += optData.B + 1/step_size
        optData.y .= x + (1 - (optData.A_prev/optData.A))*(optData.z - x)

        # update histories
        OptimizationMethods.grad!(progData, precomp, store, x)
        grad_norm = norm(store.grad)

        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = grad_norm

        # compute the gradient at y
        OptimizationMethods.grad!(progData, precomp, store, optData.y)
    end

    optData.stop_iteration = -1

    return x

end