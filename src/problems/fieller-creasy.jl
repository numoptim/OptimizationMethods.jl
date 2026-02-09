# Date: 2025/02/09
# Authors: Christian Varner
# Purpose Implementation of the fieller-creasy problem

"""
"""
mutable struct FiellerCreasy{T, S} <: AbstractDefaultQL{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    y1::Vector{T}
    y2::Vector{T}
end
function FiellerCreasy(
    ::Type{T};
    μ::Vector{Float64},
    σ::Float64 = 1.,
    nobs::Int64 = 100,
    θ::Float64 = 2.,
) where {T}

    # initialize the meta data and counters
    meta = NLPModelMeta(
        1,
        name = "Quasi-likelihood with logistic link function and centered exp",
        x0 = zeros(T, 1)
    )
    counters = Counters()

    # generate observations
    y1 = randn(T, nobs) .+ μ
    y2 = randn(T, nobs) .+ μ/θ

    return FiellerCreasy(meta, counters, y1, y2)

end

"""
"""
struct PrecomputeFiellerCreasy{T} <: AbstractPrecompute{T}
end

"""
"""
struct AllocateFiellerCreasy{T} <: AbstractFiellerCreasy{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateFiellerCreasy(progData::FiellerCreasy{T, S}) where {T, S}
    grad = zeros(T, 1)
    hess = zeros(T, 1, 1)
    return AllocateFiellerCreasy{T}(grad, hess)
end

"""
"""
function initialize(progData::FiellerCreasy{T, S}) where {T, S}
    precompute = PrecomputeFiellerCreasy{T}()
    store = AllocateFiellerCreasy(progData)

    return precompute, store
end