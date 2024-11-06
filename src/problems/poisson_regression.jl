# Date: 10/05/2024
# Author: Christian Varner
# Purpose: Implementation of Poisson regression with a canonical link function.

# NLPModel struct
"""
"""
mutable struct PoissonRegression{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    design::Matrix{T}
    response::Vector{T}
end
function PoissonRegression(
    ::Type{T};
    nobs::Int64 = 1000,
    nvar::Int64 = 50
) where {T}
    # initialize the meta
    meta = NLPModelMeta(
        nvar = nvar,
        name = "Poisson Regression w/ Canonical Link",
        x0 = zeros(T, nvar)
    )

    # initialize the counters
    counters = Counters()

    # initialize the design matrix
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ T(sqrt(nvar - 1)))

    # create the responses
    ## initialization of coefficient vector -- note this might note
    ## be the optimal solution to the optimization problem
    β = randn(T, nvar)

    ## generate data
    poisson_dist = Distributions.Poisson{T}
    λ = exp.(design * β)
    response = Vector{T}([rand(poisson_dist(λ[i])) for i in 1:nobs])

    # return the poisson regression struct
    return PoissonRegression(
        meta, 
        counters,
        design,
        response
    )
end
function PoissonRegression(
    design::Matrix{T},
    response::Vector{T};
    x0::Vector{T} = zeros(T, size(design)[2])
) where {T}
    # initialize meta
    meta = NLPModelMeta(
        name = nvar,
        name = "Poisson Regression w/ Canonical Link",
        x0 = x0
    )

    # initialize counters
    counters = Counters()

    # return the struct
    return PoissonRegression(
        meta,
        counters,
        design,
        response
    )
end

# Precomputed struct -- it might be useful at some point to allows things to control computed memory
struct PrecomputePoissReg{T} <: AbstractPrecompute{T}
    coef_coef_t::Matrix{T}
end
function PrecomputePoissReg(
    progData::PoissonRegression{T, S}
) where {T, S}

    # design matrix
    coef = progData.design
    nobs, nvar = size(coef)
    
    # for hessian calculation
    coef_coef_t = zeros(T, nobs, nvar, nvar)
    for i in 1:n
        coef_coef_t = coef[i] .* coef[i]'
    end
end

# Allocated memory struct
struct AllocatePoissReg{T} <: AbstractProblemAllocate{T}
    linear_effect::Vector{T}
    predicted_rates::Vector{T}
    residuals::Vector{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocatePoissReg(
    progData::PoissonRegression{T, S}
) where {T, S}
    nobs = size(progData.coef)[1]
    nvar = size(progData.coef)[2]

    return AllocatePoissReg(
        zeros(T, nobs),
        zeros(T, nobs),
        zeros(T, nobs),
        zeros(T, nvar),
        zeros(T, nvar, nvar)
    )
end

# Functionality

## operations without precomputed and allocated memory

## operations with precomputed and without allocated memory

## operations with precompute and allocated memory
