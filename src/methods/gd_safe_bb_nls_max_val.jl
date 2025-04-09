# Date: 2025/04/07
# Author: Christian Varner
# Purpose: Implement a (non-)monotone line search method with
# a safe barzilai-borwein step size

"""
    SafeBarzilaiBorweinNLSMaxValGD{T} <: AbstractOptimizerData{T}
"""
mutable struct SafeBarzilaiBorweinNLSMaxValGD{T} <: AbstractOptimizerData{T}
    name::String
    δ::T
    ρ::T
    line_search_max_iteration::Int64
    window_size::Int64
    objective_hist::Vector{T}
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

    @assert init_stepsize > 0 "Initial step size must be a postive value."

    name = "Safe Barzilai Borwein Gradient Descent with (Non)-monotone"*
        " line search"

    d = length(x0)

    iter_diff = zeros(T, d)
    grad_diff = zeros(T, d)
    objective_hist = zeros(T, window_size)
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
    backtracking_safe_bb_gd(optData::SafeBarzilaiBorweinNLSMaxValGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

# Reference(s)

[Raydan, Marcos. "The Barzilai and Borwein Gradient Method for the Large Scale
    Unconstrained Minimization Problem". SIAM Journal of Optimization. 
    1997.](@cite raydan1997Barzilai)

# Method

# Arguments

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
    optData.objective_hist[optData.window_size] = optData.max_value
    optData.max_index = optData.window_size

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
        shift_left!(optData.objective_hist, optData.window_size)
        optData.objective_hist[optData.window_size] = F_x
        optData.max_value, optData.max_index = 
            update_maximum(optData.objective_hist, optData.max_index-1,
                optData.window_size)

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end