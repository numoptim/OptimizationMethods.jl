# Date: 2025/05/06
# Author: Christian Varner
# Purpose: Implementation of the fixed step size modified newton
# method globalized through the watchdog technique

"""
"""
mutable struct WatchdogFixedMNewtonGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    ∇∇F_θk::Matrix{T}
    norm_∇F_ψ::T
    # modified newton helpers
    β::T
    λ::T
    hessian_modification_max_iteration::Int64
    d0k::Vector{T}
    # line search parameters
    α::T
    δ::T
    ρ::T
    line_search_max_iterations::Int64
    max_distance_squared::T
    # watchdog stopping parameters
    η::T
    inner_loop_max_iterations::Int64
    # nonmonotone line search reference values
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    # default parameters
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function WatchdogFixedMNewtonGD(::Type{T};
    x0::Vector{T},
    β::T,
    λ::T,
    hessian_modification_max_iteration::Int64,
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iterations::Int64,
    η,
    inner_loop_max_iterations::Int64,
    window_size::Int64,
    threshold::T,
    max_iterations::Int64
) where {T}
    
    name::String = "Gradient Descent with fixed step size and modified"*
    " Newton directions, globalized through Watchdog."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return WatchdogFixedMNewtonGD{T}(
        name,
        T(0),
        zeros(T, d),
        zeros(T, d, d),
        T(0),
        β,
        λ,
        hessian_modification_max_iteration,
        zeros(T, d),
        α,
        δ,
        ρ,
        line_search_max_iterations,
        T(0),
        η,
        inner_loop_max_iterations,
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
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::WatchdogFixedMNewtonGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    max_iterations = 100) where {T}

    # initialization for inner loop
    j::Int64 = 0
    dist::T = T(0)
    optData.max_distance_squared = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # hessian modification
        res = add_identity_until_pd!(store.hess;
            λ = optData.λ,
            β = optData.β,
            max_iterations = optData.hessian_modification_max_iteration)

        # if modification failed take negative gradient step; otherwise
        # take a modified newton step
        if !res[2]
            ψjk .-= optData.α .* store.grad
        else
            optData.λ = res[1] / 2
            lower_triangle_solve!(store.grad, store.hess')
            upper_triangle_solve!(store.grad, store.hess)
            ψjk .-= optData.α .* store.grad
        end

        # save the first step in case we need to backtrack
        if j == 1
            optData.d0k .= store.grad
        end

        # update the maximum distance
        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        OptimizationMethods.hess!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # check other stopping condition
        if optData.norm_∇F_ψ <= optData.η * (1 + abs(optData.F_θk))
            if OptimizationMethods.obj!(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end

    end # end while loop

    return j

end

"""
"""
function watchdog_fixed_mnewton_gd(
    optData::WatchdogFixedMNewtonGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.hess!(progData, precomp, store, x)
    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.grad_val_hist[1] = norm(store.grad)

    # Initialize the objective history
    M = length(optData.objective_hist)
    optData.objective_hist[1] = F(optData.iter_hist[1]) 
    optData.reference_value, optData.reference_value_index = 
        optData.objective_hist[1], 1

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration counter
        iter += 1

        # inner loop
        optData.F_θk = optData.objective_hist[iter]
        optData.∇F_θk .= store.grad
        optData.∇∇F_θk .= store.hess
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            max_iterations = optData.inner_loop_max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.ρ * optData.max_distance_squared

            # revert to previous iterate
            x .= optData.iter_hist[iter]

            # backtrack on the previous iterate
            backtrack_success = OptimizationMethods.backtracking!(
                x,
                optData.iter_hist[iter],
                F,
                optData.∇F_θk,
                optData.d0k,
                optData.reference_value,
                optData.α,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)

            # if backtrack not successful terminate algorithm
            if !backtrack_success
                optData.stop_iteration = (iter - 1)
                return optData.iter_hist[iter]
            end

            Fx = F(x)
        end

        # update the objective_hist
        optData.objective_hist[iter + 1] = Fx
        if (iter % M) + 1 == optData.reference_value_index
            optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
        end

        # update iter and grad value history
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        OptimizationMethods.hess!(progData, precomp, store, ψjk)
        optData.grad_val_hist = norm(store.grad)
        optData.iter_hist[iter + 1] .= x
    end

    optData.stop_iteration = iter

    return x
end