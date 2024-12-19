# Date: 12/19/2024
# Author: Christian Varner
# Purpose: Implement a quasi-likelihood examples
# with a linear mean and variance function that is 1 + mean + sin(2pi*mean)

"""
    TODO - documentation
"""
mutable struct QLLinearSine{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    design::Matrix{T}
    response::Vector{T}
end
function QLLinearSine(
    ::Type{T};
    nobs::Int64 = 1000,
    nvar::Int64 = 50
) where {T}

    @assert nobs > 0 "Number of observations is $(nobs) which is not"*
    "greater than 0."
    
    @assert nvar > 0 "Number of variables is $(nvar) which is not"*
    "greater than 0."

    # initialize the meta data and counters
    meta = NLPModelMeta(
        nvar,
        name = "Quasi-likelihood with linear mean and sine variance",
        x0 = zeros(T, nvar)
    )
    counters = Counters()

    # simulate the design matrix
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ T(sqrt(nvar - 1)))

    # get reponses
    β_true = randn(T, nvar)
    μ_obs = design * β_true
    ϵ = T.((rand(Distributions.Arcsine()) .- .5)./(1/8))
    V(μ) = 1 + μ + sin(2 * pi * μ)
    response = μ_obs + V.(μ_obs) * ϵ

    return QLLinearSine(
        meta,
        counters,
        design,
        response
    )
end

###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################