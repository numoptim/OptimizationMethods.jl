# Date: 01/27/2025
# Author: Christian Varner
# Purpose: Implement non-monotone line search scheme
# using the maximum value of the past M objective values

################################################################################
# Notes for later development:
#   Methods that use backtracking.jl essentially only deviate in
#   in the step_direction and reference value, therefore it might be nice
#   for later development to make a larger struct that encompasses
#   all these algorithms.
################################################################################

"""
# Fields
# Constructors
## Arguments
## Keyword Arguments
"""
mutable struct NonmonotoneLSMaxValGD{T} <: AbstractOptimizerData{T}
    name::String
    α::T
    δ::T
    ρ::T
    window_size::Int64                          # for reference value
    line_search_max_iteration::Int64
    objective_hist::Vector{T}                   # cache of previous values
    max_value::T                                # current maximum value
    max_index::Int64                            # index of max_value
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonmonotoneLSMaxValGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    ρ::T,
    window_size::Int64,
    line_search_max_iteration::Int64,
    threshold::T,
    max_iterations::Int64,
)
    d = length(x0)

    # initialization iterate history
    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 
        1:max_iteration + 1]
    iter_hist[1] = x0

    # initialization of gradient and dummy value for stop_iteration
    grad_val_hist = Vector{T}(undef, max_iteration + 1)
    stop_iteration = -1

    # name of the optimizer for reference
    name = "Gradient Descent with non-monotone line search using the max value"*
    " of the previous $(window_size) values" 

    return NonmonotoneLSMaxValGD(name, α, δ, ρ, window_size,
        line_search_max_iteration, zeros(T, window_size), T(0.0),
        -1, threshold, max_iterations, iter_hist, grad_val_hist,
        stop_iteration)
end

"""
# Reference(s)
# Method
# Arguments
"""
function nonmonotone_ls_maxval_gd(
    optData::NonmonotoneLSMaxValGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
)
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
    optData.objective_hist[optData.window_size] = optData.max_value
    optData.max_index = optData.window_size

    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        iter += 1

        # backtracking
        OptimizationMethods.backtracking!(x, optData.iter_hist[iter], 
            optData.max_value, F, store.grad, optData.grad_val_hist[iter] ^ 2, 
            optData.α, optData.δ, optData.ρ; optData.line_search_max_iteration)

        # compute the next gradient value
        OptimizationMethods.grad!(progData, precomp, store, x)

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # TODO - can a max-heap be used here instead?
        # TODO - should this be separated out into a different function?

        # update the objective cache
        F_x = F(x)
        popfirst!(optData.objective_hist)
        push!(optData.objective_hist, F_x)

        # update the maximum value
        if optData.max_index == 1
            optData.max_value, optData.max_index = 
                findmax(optData.objective_hist)
        else
            if optData.max_value < F_x
                optData.max_value = F_x
                optData.max_index = optData.window_size
            else
                optData.max_index -= 1
            end
        end
    end

    optData.stop_iteration = iter

    return x
end