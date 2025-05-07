# Date: 2025/05/07
# Author: Christian Varner
# Purpose: Implementation of damped BFGS

"""
"""
mutable struct WatchdogFixedDampedBFGSGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    B_θk::Matrix{T}
    norm_∇F_ψ::T
    # Quantities involved in damped BFGS update
    c::T
    Bjk::Matrix{T}
    δBjk::Matrix{T}
    rjk::Vector{T}
    sjk::Vector{T}
    yjk::Vector{T}
    d0k::Vector{T}
    djk::Vector{T}
    # line search parameters
    α::T
    δ::T
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
function WatchdogFixedDampedBFGSGD(
    ::Type{T};
    c::T,
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iterations::Int64,
    η::T,
    inner_loop_max_iterations::Int64,
    window_size::Int64,
    threshold::T,
    max_iterations::Int64
) where {T}

    name::String = "Gradient Descent with Fixed Step Size and Damped BFGS Steps"*
        " Globalized by Watchdog."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # initialize objective cache
    objective_hist = CircularVector(zeros(T, window_size))

    return WatchdogFixedDampedBFGSGD{T}(
        name,
        T(0),                                                   # F_θk
        zeros(T, d),                                            # ∇F_θk
        zeros(T, d, d),                                         # B_θk
        T(0),                                                   # norm_∇F_ψ
        c,                                                      
        zeros(T, d, d),                                         # Bjk
        zeros(T, d, d),                                         # δBjk
        zeros(T, d),                                            # rjk
        zeros(T, d),                                            # sjk
        zeros(T, d),                                            # yjk
        zeros(T, d),                                            # d0k
        zeros(T, d),                                            # djk
        α,                                                      
        δ,
        ρ,
        line_search_max_iterations,
        T(0),                                                   # max_distance_squared
        η,
        inner_loop_max_iterations,
        objective_hist,
        T(0),                                                   # reference_value
        -1,                                                     # reference_value_index
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
    optData::WatchdogFixedDampedBFGSGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    max_iterations = 100
) where {T}

    # initialization for inner loop
    j::Int64 = 0
    dist::T = T(0)
    optData.max_distance_squared = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # store values for update
        optData.sjk .= -ψjk
        optData.yjk .= -store.grad

        # compute step
        optData.djk .= optData.Bjk \ store.grad

        # take step
        ψjk .-= optData.Bjk \ store.grad

        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # update approximation
        optData.sjk .+= ψjk
        optData.yjk .+= store.grad
        update_success = OptimizationMethods.update_bfgs!(
            optData.Bjk, optData.rjk, optData.δBjk,
            optData.sjk, optData.yjk; damped_update = true)

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
function watchdog_fixed_damped_bfgs_gd(
    optData::WatchdogFixedDampedBFGSGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, s)
    optData.grad_val_hist[1] = norm(store.grad) 

    # Initialize hessian approximation
    fill!(optData.Bjk, 0)
    OptimizationMethods.add_identity(optData.Bjk,
        optData.c * optData.grad_val_hist[iter + 1])

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
        optData.B_θk .= optData.Bjk
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            max_iterations = optData.inner_loop_max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.ρ * optData.max_distance_squared

            # revert to previous iterate and approximation (for update)
            optData.Bjk .= optData.B_θk
            x .= optData.iter_hist[iter]

            # update iter_diff and grad_diff
            optData.sjk .= -x
            optData.yjk .= -optData.∇F_θk

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

            # update iter_diff and grad_diff
            OptimizationMethods.grad!(progData, precomp, store, x)
            optData.grad_val_hist[iter + 1] = norm(store.grad)

            # update BFGS approximation
            optData.sjk .+= x
            optData.yjk .+= store.grad
            update_success = OptimizationMethods.update_bfgs!(
                optData.Bjk, optData.rjk, optData.δBjk,
                optData.sjk, optData.yjk; damped_update = true)

            Fx = F(x)
        else
            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
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