# Date: 2025/04/07
# Author: Christian Varner
# Purpose: Implement a (non-)monotone line search method with
# a safe barzilai-borwein step size

"""
    BacktrackingSafeBBGD{T} <: AbstractOptimizerData{T}
"""
mutable struct BacktrackingSafeBBGD{T} <: AbstractOptimizerData{T}
    name::String
    δ::T
    ρ::T
    line_search_max_iteration::Int64
    objective_hist::Vector{T}
    max_value::T
    max_index::Int64
    init_stepsize::T
    long_stepsize::Bool
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    α_lower::T
    α_upper::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function BacktrackingSafeBBGD(::Type{T};
    x0::Vector{T},
    δ::T,
    ρ::T,
    M::Int64,
    line_search_max_iteration::Int64,
    init_stepsize::T,
    long_stepsize::Bool,
    α_lower::T,
    α_upper::T,
    threshold::T, 
    max_iterations::Int64) where {T}

    @assert init_stepsize > 0 "Initial step size must be a postive value."

    name = "Safe Barzilai Borwein Gradient Descent with (Non)-monotone"*
        " line search"

    d = length(x0)

    iter_diff = zeros(T, d)
    grad_diff = zeros(T, d)
    objective_hist = zeros(T, M)
    max_value = T(0.0)
    max_index = -1

    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:max_iterations + 1]
    iter_hist[1] = x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    return BacktrackingSafeBBGD{T}(name, δ, ρ, line_search_max_iteration, 
    objective_hist, max_value, max_index, init_stepsize, long_stepsize,
    iter_diff, grad_diff, α_lower, α_upper, threshold, max_iterations, iter_hist,
    grad_val_hist, stop_iteration)
end

"""
"""
function backtracking_safe_bb_gd(optData::BacktrackingSafeBBGD{T},
    progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

end