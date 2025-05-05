# Date: 2025/05/05
# Author: Christian Varner
# Implementation of gradient descent with Barzilai-Borwein steps
# globalized through a watchdog technique

"""
"""
mutable struct WatchdogBarzilaiBorweinGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    # step size helpers
    init_stepsize::T
    bb_step_size::Function
    α_lower::T
    α_default::T
    iter_diff_checkpoint::Vector{T}
    grad_diff_checkpoint::Vector{T}
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    # line search parameters
    ρ::T
    line_search_max_iterations::Int64
    max_distance_squared::T
    # watchdog stopping parameters
    η::T
    inner_loop_max_iterations::Int64
    # nonmonotone line search reference value
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

"""
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::WatchdogBarzilaiBorweinGD{T}, 
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

    # compute the initial step size
    step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
    if step_size < optData.α_lower || step_size > (1/optData.α_lower)
        step_size = optData.α_default
    end

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # update iter diff and grad diff
        optData.iter_diff .= -ψjk
        optData.grad_diff .= -store.grad

        # take step
        ψjk .-= step_size .* store.grad

        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # update values in iter_diff and grad_diff
        optData.iter_diff .+= ψjk
        optData.grad_idff .+= store.grad

        # safe guard against too large or too small step sizes
        step_size = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end

        # check other stopping condition
        if optData.norm_∇F_ψ <= optData.η * (1 + abs(optData.F_θk))
            if OptimizationMethods.obj!(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end
    end

    return j

end

"""
"""
function watchdog_barzilai_borwein_gd(
    optData::WatchdogBarzilaiBorweinGD{T},
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
        optData.iter_diff_checkpoint .= optData.iter_diff
        optData.grad_diff_checkpoint .= optData.grad_diff
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            max_iterations = optData.inner_loop_max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.max_distance_squared

            # revert to previous iterate
            x .= optData.iter_hist[iter]

            # compute the initial step size
            step_size = (iter == 1) ? optData.init_stepsize : 
                optData.bb_step_size(optData.iter_diff_checkpoint, 
                    optData.grad_diff_checkpoint)
            if step_size < optData.α_lower || step_size > (1/optData.α_lower)
                step_size = optData.α_default
            end

            # update iter_diff and grad_diff
            optData.iter_diff .= -x
            optData.grad_diff .= -optData.∇F_θk

            # backtrack on the previous iterate
            backtrack_success = OptimizationMethods.backtracking!(
                x,
                optData.iter_hist[iter],
                F,
                optData.∇F_θk,
                optData.grad_val_hist[iter] ^ 2,
                optData.reference_value,
                step_size,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)

            # if backtrack not successful terminate algorithm
            if !backtrack_success
                optData.stop_iteration = (iter - 1)
                return optData.iter_hist[iter]
            end

            # update iter_diff and grad_diff
            OptimizationMethods.grad!(progData, precomp, store, x)
            optData.grad_val_hist[iter + 1] = norm(store.grad)

            optData.iter_diff .+= x
            optData.grad_diff .+= store.grad

            Fx = F(x)
        else
            OptimizationMethods.grad!(progData, precomp, store, x)
            optData.grad_val_hist[iter + 1] = norm(store.grad)
        end

        # update the objective_hist
        optData.objective_hist[iter + 1] = Fx
        if (iter % M) + 1 == optData.reference_value_index
            optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
        end

        # update iter and grad value history
        optData.iter_hist[iter + 1] .= x
    end

    optData.stop_iteration = iter

    return x
end