# Date: 2025/05/02
# Author: Christian Varner
# Purpose: Implementation of the watchdog technique
# with (non-)monotone line search fall back

"""
"""
mutable struct WatchdogFixedGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    # line search helpers
    α::T
    ρ::T
    line_search_max_iterations::Int64
    max_distance_squared::T
    # nonmonotone line search reference value
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    acceptance_cnt::Int64
    # default parameters
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function WatchdogFixedGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    ρ::T,
    window_size::Int64,
    line_search_max_iterations::Int64,
    η::T,
    threshold::T,
    max_iterations::Int64
) where {T}

    # error checking
    @assert 0 < δ0 "Initial scaling factor $(δ0) needs to be positive."

    @assert δ0 <= δ_upper "Initial scaling factor $(δ0) needs to be smaller"*
    " than its upper bound $(δ_upper)."

    @assert α > 0 "The fixed step size $(α) needs to be positive."

    # name for recording purposes
    name::String = "Gradient Descent with Triggering Events and Nonsequential"*
        "Armijo with fixed step size and gradient directions."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return WatchdogFixedGD{T}(
    ) 
end

################################################################################
# Utility
################################################################################

"""
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::WatchdogFixedGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64) where {T, S}

    # initialization for inner loop
    j::Int64 = 0
    dist::T = T(0)
    optData.max_distance_squared = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]

    # stopping conditions
    while j < max_iteration 

        # Increment the inner loop counter
        j += 1

        # take step
        ψjk .-= optData.α .* store.grad

        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # other stopping condition
        if optData.norm_∇F_ψ <= optData.η * (1 + optData.F_θk)
            if OptimizationMethods.obj!(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end
    end

    return j
end

"""
"""
function nonsequential_armijo_fixed_gd(
    optData::NonsequentialArmijoFixedGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, optData.iter_hist[1])
    optData.grad_val_hist[1] = norm(store.grad)

    # update constants needed for triggering events
    optData.τ_lower = optData.grad_val_hist[1] / sqrt(2)  
    optData.τ_upper = sqrt(10) * optData.grad_val_hist[1]   

    # Initialize the objective history
    M = length(optData.objective_hist)
    optData.acceptance_cnt += 1
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
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            radius = optData.inner_loop_radius;
            max_iterations = optData.max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.max_distance_squared

            # backtrack on the previous iterate
            x .= optData.iter_hist[iter]
            backtrack_success = OptimizationMethods.backtracking!(
                x,
                optData.iter_hist[iter],
                F,
                optData.∇F_θk,
                optData.grad_val_hist[iter] ^ 2,
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
        optData.iter_hist[iter + 1] .= x
        
        OptimizationMethods.grad!(progData, precomp, store, x)
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end