# Date: 01/07/2025
# Author: Christian Varner
# Purpose: Implementation of the adaptive trust region method by
# Gratton

"""
"""
mutable struct AdaptiveTrustRegionGD{T} <: AbstractOptimizerData{T}
    name::String
    τ::T
    μ::T
    ζ::T
    w::Vector{T}
    Δ::Vector{T}
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64

    AdaptiveTrustRegionGD{T}(name, τ, μ, ζ, threshold, max_iterations,
        iter_hist, grad_val_hist, stop_iteration) = 
    begin
        d = length(iter_hist[1])
        return new(name, τ, μ, ζ, zeros(T, d), zeros(T, d), threshold, max_iterations,
            iter_hist, grad_val_hist, stop_iteration)
    end
end
function AdaptiveTrustRegionGD(
    ::Type{T};
    x0::Vector{T},
    τ::T,
    μ::T,
    ζ::T,
    threshold::T,
    max_iterations::Int64,
) where {T}

end

"""
"""
function adaptive_trust_region_gd(
    optData::AdaptiveTrustRegionGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
)

    # initialize storage and pre-computed values
    precomp, store = OptimizationMethods.initialize(progData)

    # iteration 0
    iter = 0

    x = copy(optData.iter_hist[iter + 1])

    grad!(progData, precomp, store, x)
    optData.grad_val_hist[iter + 1] = norm(store.grad)
    optData.w .= (optData.ζ + store.grad .^ 2) .^ optData.μ

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)
        
        iter += 1

        # compute trust region radius
        optData.Δ .= abs.(store.grad) ./ optData.w

        # take step
        x .-= optData.τ .* sign(store.grad) .* optData.Δ
        grad!(progData, precomp, store, x)

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # prep for next iteration
        optData.w .= ((optData.w .^ (1/μ)) + (store.grad .^ 2)) .^ μ
    end

    optData.stop_iteration = iter

    return x
end