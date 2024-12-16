# Date: 12/16/2024
# Author: Christian Varner
# Purpose: Implementation of diminishing step size
# gradient descent

"""
    TODO - documentation
"""
mutable struct DiminishingStepGD{T} <: AbstractOptimizerData{T}
    name::String
    step_size_function::Function
    step_size_scaling::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function DiminishingStepGD(
    ::Type{T};
    x0::Vector{T},
    step_size_function::Function,
    step_size_scaling::T,
    threshold::T,
    max_iterations::Int,
) where {T}
    
    # error checking
    
    # initialization histories
    d = length(x0)
    iter_hist = Vector{T}([Vector(undef, d) for i in 1:(max_iterations + 1)])
    iter_hist[1] = x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    # instantiate and return the struct
    return DiminishingStepGD("Gradient Descent with Diminishing Step Size",
                                step_size_function, step_size_scaling,
                                threshold, max_iterations, iter_hist, 
                                grad_val_hist, stop_iteration)
end

"""
    TODO - optimization method
"""
function diminishing_step_gd(
    optData::DiminishingStepGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}
    # initialization of storage and pre-computed values
    precomp, store = OptimizationMethods.initialize(progData)
    
    # update histories
    iter = 0
    x = copy(optData.iter_hist[iter + 1])
    grad!(progData, precomp, store, x)
    optData.grad_val_hist[iter + 1] = norm(store.grad)
    
    # do gradient descent with diminishing step size
    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        iter += 1

        # take a step
        step_size = optData.step_size_scaling * optData.step_size_function(iter)
        x .-= step_size .* store.grad
        grad!(progData, precomp, store, x)

        # update histories
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end
    
    optData.stop_iteration = iter

    return x
end
