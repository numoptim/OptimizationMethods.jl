# Date: 2025/05/07
# Author: Christian Varner
# Purpose: Implementation of damped BFGS

"""
"""
mutable struct WatchdogFixedDampedBFGSGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    # Quantities involved in damped BFGS update
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
        T(0),
        zeros(T, d),
        T(0),
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        α,
        δ,
        ρ,
        line_search_max_iterations,
        T(0),
        η,
        inner_loop_max_iterations,
        objective_hist,
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
end