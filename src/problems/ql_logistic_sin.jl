# Date: 12/19/2024
# Author: Christian Varner
# Purpose: Implement a quasi-likelihood examples
# with a linear mean and variance function that is 1 + mean + sin(2pi*mean)

"""
    QLLogisticSin{T, S} <: AbstractNLPModel{T, S}

Implements a Quasi-likelihood objective with a logistic link function and
    `linear_plus_sin` variance funtion. If the design matrix and the
    responses are not supplied, they are randomly generated.

# Objective Function

Let ``A`` be the design matrix, and ``b`` be the responses. Each row of ``A``
    and corresponding entry in ``b`` are the predictor and observations from one
    unit. The statistical model for this objective function assums ``b``
    are between ``0`` and ``1``.

Let ``A_i`` be row ``i`` of ``A`` and ``b_i`` entry ``i`` of ``b``. Let
```math
    \\mu_i(x) = \\mathrm{logistic}(A_i^\\intercal x)
```
and
```math
    v_i(\\mu) = 1 + \\mu(x) + \\sin(2 * \\pi * \\mu(x)). 
```

Let ``n`` be the number of rows in ``A``, then the quasi-likelihood objective is
```math
    F(x) = -\\sum_{i=1}^n \\int_0^{\\mu_i(x)} (b_i - \\mu)/v_i(\\mu) d\\mu.
```

!!! note
    ``F(x)`` does not have an easily expressible closed form, so a numerical
    integration scheme is used to evaluate the objective. The gradient
    and hessian have closed form solutions however.

# Fields

- `meta::NLPModelMeta{T, S}`, NLPModel struct for storing meta information for 
    the problem
- `counters::Counters`, NLPModel Counter struct that provides evaluations 
    tracking.
- `design::Matrix{T}`, covariate matrix for the problem/experiment (``A``).
- `response::Vector{T}`, observations for the problem/experiment (``b``).
- `mean::Function`
- `mean_first_derivative::Function`
- `mean_second_derivative::Function`
- `variance::Function`
- `variance_first_derivative::Function`
- `residual::Function`

# Constructors

## Inner Constructors

    QLLogisticSin{T, S}(meta::NLPModelMeta{T, S}, counters::Counters,
        design::Matrix{T}, response::Vector{T})

## Outer Constructors

    function QLLogisticSin(::Type{T}; nobs::Int64 = 1000,
        nvar::Int64 = 50) where {T}
 
    function QLLogisticSin(design::Matrix{T}, response::Vector{T}, 
        x0::Vector{T} = zeros(T, size(design)[2])) where {T}

"""
mutable struct QLLogisticSin{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    design::Matrix{T}
    response::Vector{T}
    mean::Function
    mean_first_derivative::Function
    mean_second_derivative::Function
    variance::Function
    variance_first_derivative::Function
    residual::Function

    QLLogisticSin{T, S}(meta, counters, design, response) where {T, S} = 
    begin
        new(meta, counters, design, response, 
            OptimizationMethods.logistic,           # force the correct mean function
            OptimizationMethods.dlogistic,          # force the correct derivative function
            OptimizationMethods.ddlogistic,         # force the correct derivative function
            OptimizationMethods.linear_plus_sin,    # force the correct variance function
            OptimizationMethods.dlinear_plus_sin,   # force the correct derivative function
            F(μ, y) = (y - μ)/OptimizationMethods.linear_plus_sin(μ)
        )
    end
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

    return QLLogisticSin{T, Vector{T}}(
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
    return QLLogisticSin{T, Vector{T}}(
        meta,
        counters,
        design,
        response,
    )
end

# precompute struct
"""
"""
struct PrecomputeQLLogisticSin{T} <: AbstractPrecompute{T}
    obs_obs_t::Array{T, 3}
end
function PrecomputeQLLogisticSin(progData::QLLogisticSin{T, S}) where {T, S}

    # get the size of the matrix
    nobs, nvar = size(progData.design)

    # create the space
    obs_obs_t = zeros(T, nobs, nvar, nvar)
    
    for i in 1:nobs
        obs_obs_t[i, :, :] .= view(progData.design, i, :) *
            view(progData, i, :)'
    end

    return PrecomputeQLLogisticSin{T}(obs_obs_t)
end

# allocate struct
"""
"""
mutable struct AllocateQLLogisticSin{T} <: AbstractProblemAllocate{T}
    linear_effect::Vector{T}   
    μ::Vector{T}
    ∇μ_η::Vector{T}
    ∇∇μ_η::Vector{T}
    variance::Vector{T}
    ∇variance::Vector{T}
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
    function NLPModels.obj($(args...)) where {T, S}
        increment!(progData, :neval_obj)
        η = progData.design * x
        μ_hat = progData.mean.(η)
        obj = 0
        for i in 1:length(progData.response)
            ## create numerical integration problem
            prob = IntegralProblem(
                progData.residual, 
                (0, μ_hat[i]), 
                progData.response[i]
                )

            ## solve the numerical integration problem
            ## TODO: this is just with some default parameters
            ## TODO: what happens when the code is false
            obj -= solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
        end

        return T(obj)
    end

    @doc"""
    """
    function NLPModels.grad($(args...)) where {T, S}
        increment!(progData, :neval_grad)

        # compute values required for gradient
        η = progData.design * x
        μ_hat = progData.mean.(η)
        var = progData.variance.(μ_hat)
        d = progData.dlogistic.(η)
        residual = (progData.response .- μ_hat)./var

        # compute and return gradient
        g = zeros(T, length(x))
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

        # compoute required quantities
        η = progData.design * x
        μ_hat = progData.mean.(η)
        ∇μ_η = progData.mean_first_derivative.(η)
        ∇∇μ_η = progData.mean_second_derivative.(η)
        var = progData.variance.(μ_hat)
        ∇var = progData.variance_first_derivative.(μ_hat)
        r = (progData.response .- μ_hat)./var

        # compute hessian
        nobs, nvar = size(progData.design)
        H = zeros(T, nvar, nvar)
        for i in 1:nobs
            t1 = var[i]^(-2) * ∇var[i] * (∇μ_η[i]^2) * r[i]
            t2 = var[i]^(-1) * (∇μ_η[i]^2)
            t3 = var[i]^(-1) * r[i] * ∇∇μ_η[i] 
            oi = view(progData.design, i, :) * view(progData.design, i, :)

            H -= (t3 - t1 - t2) * oi 
        end

        return H
    end

end

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::QLLogisticSin{T, S}),
    :(precomp::PrecomputeQLLogisticSin{T}),
    :(x::Vector{T})
]

@eval begin

    @doc"""
    """
    function NLPModels.obj($(args_pre...)) where {T, S}
        return NLPModels.obj(progData, x)
    end

    @doc"""
    """
    function NLPModels.grad($(args_pre...)) where {T, S}
        return NLPModels.grad(progData, x)
    end

    @doc"""
    """
    function NLPModels.objgrad($(args_pre...)) where {T, S}
        return NLPModels.objgrad(progData, x)
    end

    @doc"""
    """
    function NLPModels.hess($(args_pre...)) where {T, S}
        increment!(progData, :neval_hess)

        # compoute required quantities
        η = progData.design * x
        μ_hat = progData.mean.(η)
        ∇μ_η = progData.mean_first_derivative.(η)
        ∇∇μ_η = progData.mean_second_derivative.(η)
        var = progData.variance.(μ_hat)
        ∇var = progData.variance_first_derivative.(μ_hat)
        r = (progData.response .- μ_hat)./var

        # compute hessian
        nobs, nvar = size(progData.design)
        H = zeros(T, nvar, nvar)
        for i in 1:nobs
            t1 = var[i]^(-2) * ∇var[i] * (∇μ_η[i]^2) * r[i]
            t2 = var[i]^(-1) * (∇μ_η[i]^2)
            t3 = var[i]^(-1) * r[i] * ∇∇μ_η[i] 

            H -= (t3 - t1 - t2) * view(precomp.obs_obs_t, i, :, :)
        end

        return H
    end
end

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################

args_store = [
    :(progData::QLLogisticSin{T, S}),
    :(precomp::PrecomputeQLLogisticSin{T}),
    :(store::AllocateQLLogisticSin{T}),
    :(x::Vector{T})
]

@eval begin

    @doc"""
    """
    function NLPModels.obj($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_obj)
        
        # recompute possible values
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= progData.mean.(store.linear_effect)
        end
        
        # recompute possible objective functions
        obj = 0
        for i in 1:length(progData.response)
            ## create numerical integration problem
            prob = IntegralProblem(
                progData.residual, 
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
    function NLPModels.grad!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_grad)
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= progData.mean.(store.linear_effect)
            store.∇μ_η .= progData.mean_first_derivative.(store.linear_effect)
            store.variance .= progData.variance.(store.μ)
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
    function NLPModels.objgrad($(args_store...); recompute = true) where {T, S}
        NLPModels.grad!(progData, precomp, store, x; recompute = true)
        o = NLPModels.obj(progData, precomp, store, x; recompute = false)
        return o
    end

    @doc"""
    """
    function hess!($(args_store...); recompute = true) where {T, S}
        increment!(progData, :neval_hess)

        # compoute required quantities
        if recompute
            store.linear_effect .= progData.design * x
            store.μ .= progData.mean.(store.linear_effect)
            store.∇μ_η .= progData.mean_first_derivative.(store.linear_effect)
            store.∇∇μ_η .= progData.mean_second_derivative.(store.linear_effect)
            store.variance .= progData.variance.(store.μ)
            store.∇variance .= progData.variance_first_derivative.(store.μ)
            store.residual .= (progData.response .- store.μ)./store.variance
        end

        # compute hessian
        nobs = size(progData.design, 1)
        fill!(store.hess, 0)
        for i in 1:nobs
            t1 = store.variance[i]^(-2) * store.∇variance[i] * 
                (store.∇μ_η[i]^2) * store.residual[i]
            t2 = store.variance[i]^(-1) * (store.∇μ_η[i]^2)
            t3 = store.variance[i]^(-1) * store.residual[i] * store.∇∇μ_η[i] 

            store.hess -= (t3 - t1 - t2) * view(precomp.obs_obs_t, i, :, :)
        end
    end

end