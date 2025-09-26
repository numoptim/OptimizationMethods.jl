# Date: 2025/09/18
# Author: Christian Varner
# Purpose: Implement a struct were the mean function is the logistic
# while the variance function is allowed be to be input by the user.

# TODO: testing

mutable struct QLLogistic{T, S} <: AbstractDefaultQL{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    design::Matrix{T}
    response::Vector{T}
    mean::Function
    mean_first_derivative::Function
    mean_second_derivative::Function
    variance::Function
    variance_first_derivative::Function
    weighted_residual::Function

    QLLogistic{T, S}(meta::NLPModelMeta{T, S}, 
                    counters::Counters,
                    design::Matrix{T}, 
                    response::Vector{T},
                    V::Function,
                    dV::Function) where {T, S} =
    begin
        weighted_residual(μ, y) = (y - μ) / V(μ)      
        new(
            meta,
            counters,
            design,
            response,
            OptimizationMethods.logistic,
            OptimizationMethods.dlogistic,
            OptimizationMethods.ddlogistic,
            V,
            dV,
            weighted_residual,
        )
    end
end
function QLLogistic(
    ::Type{T},
    V::Function,
    dV::Function;
    nobs::Int64 = 1000,
    nvar::Int64 = 50,
) where {T}

    # initialize the meta data and counters
    meta = NLPModelMeta(
        nvar,
        name = "Quasi-likelihood with logistic link function and user-defined variance",
        x0 = zeros(T, nvar)
    )
    counters = Counters()

    # simulate the design matrix
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ T(sqrt(nvar - 1)))

    # get reponses
    β_true_mean = randn(T, nvar)
    β_true = β_true_mean + randn(T, nvar)
    η = design * β_true
    μ_obs = OptimizationMethods.logistic.(η)
    ϵ = T.((rand(Distributions.Arcsine(), nobs) .- .5)./sqrt(1/8)) # standardize

    # generate responses
    response = μ_obs + T.(V.(μ_obs) .^ (.5)) .* ϵ

    return QLLogistic{T, Vector{T}}(
        meta,
        counters,
        design,
        response,
        V,
        dV
    )
end
function QLLogistic(
    design::Matrix{T},
    response::Vector{T},
    V::Function,
    dV::Function;
    x0::Vector{T} = zeros(T, size(design, 2)),
) where {T}

    @assert size(design, 1) == size(response, 1) "Number rows in design matrix"*
    " must be equal to the number of observations."

    @assert size(design, 2) == size(x0, 1) "Number of columns in design matrix"*
    " must be equal to parameter dimension."
    
    # initialize meta
    meta = NLPModelMeta(
            size(design, 2),
            name = "Quasi-likelihood with logistic link function and user-defined variance",
            x0 = x0
           )

    # initialize counters
    counters = Counters()

    return QLLogistic{T, Vector{T}}(
        meta,
        counters,
        design,
        response,
        V,
        dV
    )
end

"""
    PrecomputeQLLogistic{T} <: AbstractDefaultQLPrecompute{T}

Structure that holds precomputed values for the quasi-likelihood problem.
    These values are precomputed to save on time, and they remain unchanged
    throughout the algorithm.

# Fields

- `obs_obs_t`, 3d array where `obs_obs_t[i, :, :]` contains the outer produce
    between the ith covariate vector and itself.

# Constructor

    PrecomputeQLLogistic(progData::QLLogistic{T, S}
        ) where {T, S}

Initializes the field values for the precompute data structure and returns 
    a `struct`.
"""
struct PrecomputeQLLogistic{T} <: AbstractDefaultQLPrecompute{T}
    obs_obs_t::Array{T, 3}
end
function PrecomputeQLLogistic(progData::QLLogistic{T, S}
    ) where {T, S}
    
    # get the size of the matrix
    nobs, nvar = size(progData.design)

    # create the space
    obs_obs_t = zeros(T, nobs, nvar, nvar)
    
    for i in 1:nobs
        obs_obs_t[i, :, :] .= view(progData.design, i, :) *
            view(progData.design, i, :)'
    end

    return PrecomputeQLLogistic{T}(obs_obs_t)
end

"""
    AllocateQLLogistic{T} <: AbstractDefaultQLAllocate{T}

Mutable struct that contains buffer arrays for various computations used for
    this objective function and for optimization algorithms.

# Fields

- `linear_effect::Vector{T}`, buffer array for `progData.response * x`.   
- `μ::Vector{T}`, buffer array for response prediction.
- `∇μ_η::Vector{T}`, buffer array for first derivatives for the mean function
    evaluated at each point in `linear_effect`.
- `∇∇μ_η::Vector{T}`, buffer arry for second derivatives for the mean function
    evaluated at each point in `linear_effect`.
- `variance::Vector{T}`, buffer array for the variance function evaluated at
    each point `μ`
- `∇variance::Vector{T}`, buffer array for the first derivatives for the 
    variance function evaluated at each point in `μ`
- `weighted_residual::Vector{T}`, buffer array for the weighted residuals.
- `grad::Vector{T}`, buffer array for the gradient vector.
- `hess::Matrix{T}`, buffer matrix for the hessian.

# Constructors

    AllocateQLLogistic(progData::QLLogistic{T,S}
        ) where {T,S}

Allocates memory for each of the field values and returns the struct.
"""
struct AllocateQLLogistic{T} <: AbstractDefaultQLAllocate{T}
    linear_effect::Vector{T}   
    μ::Vector{T}
    ∇μ_η::Vector{T}
    ∇∇μ_η::Vector{T}
    variance::Vector{T}
    ∇variance::Vector{T}
    weighted_residual::Vector{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateQLLogistic(progData::QLLogistic{T, S}
    ) where {T, S}

    # get dimensions
    nobs = size(progData.design, 1)
    nvar = size(progData.design, 2)

    # initialize memory
    linear_effect = zeros(T, nobs)
    μ = zeros(T, nobs)
    ∇μ_η = zeros(T, nobs)
    ∇∇μ_η = zeros(T, nobs)
    variance = zeros(T, nobs)
    ∇variance = zeros(T, nobs)
    weighted_residual = zeros(T, nobs)
    grad = zeros(T, nvar)
    hess = zeros(T, nvar, nvar)

    return AllocateQLLogistic{T}(
        linear_effect,
        μ, ∇μ_η, ∇∇μ_η,
        variance, ∇variance,
        weighted_residual,
        grad,
        hess
    )
end

"""
    initialize(progData::QLLogisticCenteredExp{T,S}) where {T,S}

Creates a `PrecomputeQLLogisticCenteredExp` and `AllocateQLLogisticCenteredExp` 
    struct, returning them in that order.
"""
function initialize(progData::QLLogistic{T, S}) where {T, S}
    precomp = PrecomputeQLLogistic(progData)
    store = AllocateQLLogistic(progData)

    return precomp, store
end