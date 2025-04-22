# Date: 2025/04/22
# Author: Christian Varner
# Purpose: Implementation of gradient descent with limited
# memory damped lbfgs

"""
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
        "damped BFGS updates"
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
    OptimizationMethods.add_identity(optData.B,
        optData.c * norm(store.grad))

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
        optData.step .= store.grad

        # compute step
        chol_success = OptimizationMethods.cholesky_and_solve(optData.step, 
            optData.B)

        if isnan(optData.step[1])
            optData.stop_iteration = (iter - 1)
            return optData.iter_hist[iter]
        end

        # backtrack
        success = OptimizationMethods.backtracking!(
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
        if !success
            optData.stop_iteration = (iter - 1)
            return optData.iter_hist[iter]
        end
        
        # compute the next gradient and hessian values
        OptimizationMethods.grad!(progData, precomp, store, x)

        # update approximation
        optData.s .+= x
        optData.y .+= store.grad
        OptimizationMethods.update_bfgs!(optData.B, optData.r, optData.δB,
            optData.s, optData.y; damped_update = true)
        OptimizationMethods.add_identity(optData.B, optData.β)

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