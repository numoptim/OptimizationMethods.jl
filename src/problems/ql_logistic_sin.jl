# Date: 12/19/2024
# Author: Christian Varner
# Purpose: Implement a quasi-likelihood examples
# with a linear mean and variance function that is 1 + mean + sin(2pi*mean)

"""
    TODO - documentation
"""
mutable struct QLLogisticSin{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    design::Matrix{T}
    response::Vector{T}
end
function QLLogisticSin(
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
        name = "Quasi-likelihood with logistic link function and sine variance",
        x0 = zeros(T, nvar)
    )
    counters = Counters()

    # simulate the design matrix
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ T(sqrt(nvar - 1)))

    # get reponses
    β_true = randn(T, nvar)
    η = design * β_true
    μ_obs = OptimizationMethods.logistic.(η)
    ϵ = T.((rand(Distributions.Arcsine()) .- .5)./(1/8))

    # generate responses
    response = μ_obs + OptimizationMethods.linear_plus_sin.(μ_obs) * ϵ

    return QLLogisticSin(
        meta,
        counters,
        design,
        response
    )
end
function QLLogisticSin(
    design::Matrix{T},
    response::Vector{T},
    x0::Vector{T} = zeros(T, size(design)[2])
) where {T}

    @assert size(design, 1) == size(response, 1) "Number rows in design matrix is not"*
    "equal to the number of observations."

    @assert size(design, 2) == size(x0, 1) "Number of columns in design matrix is not"*
    "equal to the number of elements in x0."
    
    # initialize meta
    meta = NLPModelMeta(
            size(design, 2),
            name = "Quasi-likelihood with logistic link function and sine variance",
            x0 = x0
           )

    # initialize counters
    counters = Counters()

    # return the struct
    return QLLogisticSin(
        meta,
        counters,
        design,
        response
    )
end

# precompute struct
"""
"""
struct PrecomputeQLLogisticSin{T} <: AbstractPrecompute{T}
end
function PrecomputeQLLogisticSin(progData::QLLogisticSin{T, S}) where {T, S}
end

# allocate struct
"""
"""
mutable struct AllocateQLLogisticSin{T} <: AbstractProblemAllocate{T}
    linear_effect::Vector{T}   
    μ::Vector{T}
    ∇μ_∇η::Vector{T}
    variance::Vector{T}
    residual::Vector{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocateQLLogisticSin(progData::QLLogisticSin{T, S}) where {T, S}

    # get dimensions
    nobs = size(progData.design, 1)
    nvar = size(progData.design, 2)

    # initialize memory
    linear_effect = zeros(T, nobs)
    μ = zeros(T, nobs)
    ∇μ_∇η = zeros(T, nobs)
    variance = zeros(T, nobs)
    residual = zeros(T, nobs)
    grad = zeros(T, nvar)
    hess = zeros(T, nvar, nvar)

    return AllocateQLLogisticSin(
        linear_effect,
        μ, ∇μ_∇η,
        variance, 
        residual,
        grad, 
        hess
    )
end

"""
"""
function initialize(progData::QLLogisticSin{T, S}) where {T, S}
    precomp = PrecomputeQLLogisticSin(progData)
    store = AllocateQLLogisticSin(progData)

    return precomp, store
end

###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

args = [:(progData::QLLogisticSin{T, S}), 
        :(x::Vector{T})
       ]

@eval begin

    @doc"""
    """
    function residual(μi, yi)
        return (yi - μi)/OptimizationMethods.linear_plus_sin(μi)
    end

    @doc"""
    """
    function NLPModels.obj($(args...)) where {T, S}
        increment!(progData, :neval_obj)
        η = progData.design * x
        μ_hat = OptimizationMethods.logistic.(η)
        obj = 0
        for i in 1:length(progData.response)
            ## create numerical integration problem
            prob = IntegralProblem(
                residual, 
                (0, μ_hat[i]), 
                progData.response[i]
                )

            ## solve the numerical integration problem
            ## TODO: this is just with some default parameters
            ## TODO: what happens when the code is false
            obj -= solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
        end

        return obj
    end

    @doc"""
    """
    function NLPModels.grad($(args...)) where {T, S}
        increment!(progData, :neval_grad)

        # compute values required for gradient
        η = progData.design * x
        μ_hat = OptimizationMethods.logistic.(η)
        var = OptimizationMethods.linear_plus_sin.(μ_hat)
        d = OptimizationMethods.dlogistic.(η)
        residual = (progData.response .- μ_hat)./var

        # compute and return gradient
        g = zeros(length(x))
        for i in 1:length(progData.response)
            g .-= residual[i] * d[i] .* view(progData.design, i, :)
        end
        return g
    end

    @doc"""
    """
    function NLPModels.objgrad($(args...)) where {T, S}
        o = obj(progData, x)
        g = grad(progData, x)
        return o, g 
    end
    
    @doc"""
    """
    function hess($(args...)) where {T, S}
        increment!(progData, :neval_hess)
    end

end

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args = [
    :(progData::QLLogisticSin{T, S}),
    :(precomp::PrecomputeQLLogisticSin{T}),
    :(x::Vector{T})
]

@eval begin

end

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################

args = [
    :(progData::QLLogisticSin{T, S}),
    :(precomp::PrecomputeQLLogisticSin{T}),
    :(store::AllocateQLLogisticSin{T}),
    :(x::Vector{T})
]

@eval begin

    @doc"""
    """
    function residual(μi, yi)
        return (yi - μi)/OptimizationMethods.linear_plus_sin(μi)
    end

    @doc"""
    """
    function NLPModels.obj($(args...); recompute = true) where {T, S}
        increment!(progData, :neval_obj)
        
        # recompute possible values
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= OptimizationMethods.logistic.(η)
        end
        
        # recompute possible objective functions
        obj = 0
        for i in 1:length(progData.response)
            ## create numerical integration problem
            prob = IntegralProblem(
                residual, 
                (0, store.μ[i]), 
                progData.response[i]
                )

            ## solve the numerical integration problem
            ## TODO: this is just with some default parameters
            ## TODO: what happens when the code is false
            obj -= solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
        end

        return obj
    end

    @doc"""
    """
    function NLPModels.grad!($(args...); recompute = true) where {T, S}
        increment!(progData, :neval_grad)
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= OptimizationMethods.logistic.(store.linear_effect)
            store.∇μ_∇η .= OptimizationMethods.dlogistic.(store.linear_effect)
            store.variance .= OptimizationMethods.logistic_plus_sin.(store.μ)
            store.residual .= (progData.response .- store.μ)./store.variance
        end

        fill!(store.grad, 0)
        for i in 1:length(progData.response)
            store.grad .-= store.residual[i] * store.∇μ_∇η[i] .* 
                view(progData.design, i, :)
        end
    end

    @doc"""
    """
    function NLPModels.objgrad($(args...); recompute = true) where {T, S}
        NLPModels.grad!(progData, precomp, store, x; recompute = true)
        o = NLPModels.obj(progData, precomp, store, x; recompute = false)
        return o
    end

end