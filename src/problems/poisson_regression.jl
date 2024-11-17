# Date: 10/05/2024
# Author: Christian Varner
# Purpose: Implementation of Poisson regression with a canonical link function.

# NLPModel struct
"""
    PoissonRegression{T, S} <: AbstractNLPModel{T, S}

Implements Poisson regression with the canonical link function. If the design
    matrix (i.e., the covariates) and responses are not supplied, they are 
    randomly generated. 

# Objective Function

Let ``A`` be the design matrix, and ``b`` be the responses. Each row of ``A``
    and corresponding entry in ``b`` are the predictor and observation from one 
    unit. The entries in ``b`` must be integer valued and non-negative.

Let ``A_i`` be row ``i`` of ``A`` and ``b_i`` entry ``i`` of ``b``. Let
```math
\\mu_i(x) = \\exp(A_i^\\intercal x).
```

Let ``n`` be the number of rows of ``A`` (i.e., number of observations), then
    the negative log-likelihood of the model is 
```math
\\sum_{i=1}^n \\mu_i(x) - b_i (A_i^\\intercal x) + C(b),
```
where ``C(b)`` is a constant depending on the data. We implement the objective  
    function to be the negative log-likelihood up to the constant term ``C(b)``. 
    That is,
```math
F(x) = \\sum_{i=1}^n \\mu_i(x) - b_i (A_i^\\intercal x).
```

!!! Remark
    Because the additive term ``C(b)`` is not included in the objective function,
    the objective function can take on negative values.

# Fields

- `meta::NLPModelMeta{T, S}`, NLPModel struct for storing meta information for 
    the problem
- `counters::Counters`, NLPModel Counter struct that provides evaluations 
    tracking.
- `design::Matrix{T}`, covariate matrix for the problem/experiment (``A``).
- `response::Vector{T}`, observations for the problem/experiment (``b``).

# Constructors

    PoissonRegression(::Type{T}; nobs::Int64 = 1000, nvar::Int64 = 50) where {T}

Construct the `struct` for Poisson Regression when simulated data is needed. 
    The design matrix (``A``) and response vector ``b`` are randomly generated 
    as follows. 
    For the design matrix, the first column is all ones, and the rest are
    generated according to a normal distribution where each row has been 
    scaled to have unit variance (excluding the first column). 
    For the response vector, let ``\\beta`` be the "true" relationship between 
    the covariates and response vector for the poisson regression model, 
    then the ``i``th entry of the response vector is generated from a Poisson 
    Distribution with rate parameter ``\\exp(A_i^\\intercal \\beta)``.

    PoissonRegression(design::Matrix{T}, response::Vector{T}; 
        x0::Vector{T} = zeros(T, size(design)[2])) where {T}

Constructs the `struct` for Poisson Regression when the design matrix and response 
    vector are known. The initial guess, `x0` is a keyword argument that is set 
    to all zeros by default. 

!!! Remark
    When using this constructor, the number of rows of `design` must be equal to 
    the size of response. When providing `x0`, the number of entries must be the 
    same as the number of columns in `design`.
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
) where T
    
    # error checking
    @assert nobs > 0 "`nobs` is non-zero or negative"
    @assert nvar > 0 "`nvar` is non-zero or negative"

    # initialize the meta
    meta = NLPModelMeta(
        nvar,
        name = "Poisson Regression w/ Canonical Link",
        x0 = zeros(T, nvar)
    )

    # initialize the counters
    counters = Counters()

    # initialize the design matrix, normalize rows to have unit variance 
    # excludes first entry
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ T(sqrt(nvar - 1)))

    # create the responses
    ## initialization of coefficient vector -- note this might note
    ## be the optimal solution to the optimization problem
    β = randn(T, nvar)

    ## generate rates 
    λ = exp.(design * β)

    ## Generate response 
    response = T.(rand.(Distributions.Poisson{T}.(λ)))

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

    @assert size(design, 1) == size(response, 1) "Number rows in the design matrix
    and number of entries in the reponse are not equal. These must be equal."
    @assert size(design, 2) == size(x0, 1) "`x0` has a different size then the number
    of columns in the design matrix. These must be equal."

    # initialize meta
    meta = NLPModelMeta(
        size(design, 2),
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

# Precomputed struct 
# IMPROVEMENT: Allow precomputed values to be determined by memory availability
# /user preferences
"""
    PrecomputePoissReg{T} <: AbstractPrecompute{T}

Store precomputed values that are used in computation of the hessian for poisson 
    regression.

# Fields

- `obs_obs_t::Matrix{T}`, tensor where `obs_obs_t[i, :, :]` is equal to 
    ``A_i A_i^\\intercal``. 

# Constructor
    
    PrecomputePoissReg(progData::PoissonRegression{T, S})

Requests memory for `obs_obs_t` which is an `nobs` by `nvar` by `nvar` tensor, 
    and computes the outer produce for each row of `A`. Returns the structure.
"""
struct PrecomputePoissReg{T} <: AbstractPrecompute{T}
    obs_obs_t::Array{T, 3}
end
function PrecomputePoissReg(
    progData::PoissonRegression{T, S}
) where {T, S}

    # design matrix
    nobs, nvar = size(progData.design)
    
    # for hessian calculation
    obs_obs_t = zeros(T, nobs, nvar, nvar)
    for i in 1:nobs
        obs_obs_t[i, :, :] .= view(progData.design, i, :) * 
            view(progData.design, i, :)'
    end

    return PrecomputePoissReg{T}(obs_obs_t)
end

# Allocated memory struct
"""
    AllocatePoissReg{T} <: AbstractProblemAllocate{T}

Structure that creates buffer arrays to store computed values for the Poisson 
    regression problem. 

# Fields

- `linear_effect::Vector{T}`, memory for the term `A * x`
- `predicted_rates::Vector{T}`, memory for the term `exp.(A * x)`
- `residuals::Vector{T}`, memory for the term `exp.(A * x) - b`
- `grad::Vector{T}`, memory for the gradient of the objective
- `hess::Matrix{T}`, memory for the hessian of the objective

# Constructor

    AllocatePoissReg(progData::PoissonRegression{T, S}) where {T, S}

Requests the memory for each of the fields and initializes each buffer to 
    contain zeros. Returns the structure.
"""
mutable struct AllocatePoissReg{T} <: AbstractProblemAllocate{T}
    linear_effect::Vector{T}
    predicted_rates::Vector{T}
    residuals::Vector{T}
    grad::Vector{T}
    hess::Matrix{T}
end
function AllocatePoissReg(
    progData::PoissonRegression{T, S}
) where {T, S}
    nobs = size(progData.design)[1]
    nvar = size(progData.design)[2]

    return AllocatePoissReg(
        zeros(T, nobs),
        zeros(T, nobs),
        zeros(T, nobs),
        zeros(T, nvar),
        zeros(T, nvar, nvar)
    )
end

"""
    initialize(progData::PoissonRegression{T, S}) where {T, S}

Constructs `PrecomputePoissReg` and `AllocatePoissReg` with the problem data 
    specified by `progData` and returns them (in the same order). 
"""
function initialize(progData::PoissonRegression{T, S}) where {T, S}
    precomp = PrecomputePoissReg(progData)
    store = AllocatePoissReg(progData)

    return precomp, store
end

# Functionality

## operations without precomputed and allocated memory

args = [
    :(progData::PoissonRegression{T, S}),
    :(x::Vector{T})
]

@eval begin 

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.

    !!! Remark:
        The objective function is computed up to a constant.
        That is we compute `F(x)` and the negative log-likelihood
        is `F(x) + C(b)` where `C(b)` depends on the responses `b`.
    """
    function NLPModels.obj($(args...)) where {T,S}
        increment!(progData, :neval_obj)
        linear_predictor = progData.design * x
        predicted_rates = exp.(linear_predictor)
        return sum(predicted_rates) - dot(progData.response, linear_predictor)
    end

   @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. 
    """
    function NLPModels.grad($(args...)) where {T,S}
        increment!(progData, :neval_grad)
        linear_predictor = progData.design * x
        predicted_rates = exp.(linear_predictor)
        residual = predicted_rates - progData.response
        
        n, p = size(progData.design) 
        g = zeros(T, p)
        for i in 1:n
            g .+= residual[i] .* view(progData.design, i, :)
        end
        return g 
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`.
    """
    function NLPModels.objgrad($(args...)) where {T, S}
        o = obj(progData, x)
        g = grad(progData, x)
        return o, g
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. 
    """
    function hess($(args...)) where {T,S}
        increment!(progData, :neval_hess)
        linear_predictor = progData.design * x
        predicted_rates = exp.(linear_predictor)
        n, p = size(progData.design)
        H = zeros(T, p, p)
        for i in 1:n
            H .+= predicted_rates[i] .* 
            (view(progData.design, i, :) * view(progData.design, i, :)')
        end
        return H
    end
end

## operations with precomputed and without allocated memory

args = [
    :(progData::PoissonRegression{T, S}),
    :(precomp::PrecomputePoissReg{T}),
    :(x::Vector{T})
]

@eval begin 

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`.

    !!! Remark:
        The objective function is computed up to a constant.
        That is we compute `F(x)` and the negative log-likelihood
        is `F(x) + C(b)` where `C(b)` depends on the responses `b`.
    """
    function NLPModels.obj($(args...)) where {T,S}
        return NLPModels.obj(progData, x)
    end

   @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. 
    """
    function NLPModels.grad($(args...)) where {T,S}
        return NLPModels.grad(progData, x)
    end

    @doc """
        objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`.
    """
    function NLPModels.objgrad($(args...)) where {T, S}
        o = obj(progData, x)
        g = grad(progData, x)
        return o, g
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x` utilizing the 
        precomputed values from `precomp`.
    """
    function hess($(args...)) where {T,S}
        increment!(progData, :neval_hess)
        linear_predictor = progData.design * x
        predicted_rates = exp.(linear_predictor)
        n, p = size(progData.design)
        H = zeros(T, p, p)
        for i in 1:n
            H .+= predicted_rates[i] .* view(precomp.obs_obs_t, i, :, :) 
        end
        return H
    end
end

## operations with precompute and allocated memory

args = [
    :(progData::PoissonRegression{T, S}),
    :(precomp::PrecomputePoissReg{T}),
    :(store::AllocatePoissReg{T}),
    :(x::Vector{T})
]

@eval begin 

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))    
        ) where {T,S}

    Computes the objective function at the value `x`. If `recompute = true`,
        the values in `store` relating to the objective function computation are 
        recalculated. Otherwise, the values already in `store` are used to 
        compute the objective.

    !!! Remark:
        The objective function is computed up to a constant.
        That is we compute `F(x)` and the negative log-likelihood
        is `F(x) + C(y)` where `C(y)` depends on the responses.
    """
    function NLPModels.obj($(args...); recompute::Bool = true) where {T,S}
        increment!(progData, :neval_obj)
        if recompute
            store.linear_effect .= progData.design * x
            store.predicted_rates .= exp.(store.linear_effect)
        end
        return sum(store.predicted_rates) - dot(progData.response, store.linear_effect)
    end

   @doc """
        grad!(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient of the objective function at `x`. Stores the result in 
        `store.grad`. If `recompute = true`, the values in `store` that are needed
        for the computation of the gradient are recomputed and used. Otherwise,
        the values already in `store` are used to compute the gradient.
    """
    function NLPModels.grad!($(args...); recompute::Bool = true) where {T,S}
        increment!(progData, :neval_grad)
        if recompute
            store.linear_effect .= progData.design * x
            store.predicted_rates .= exp.(store.linear_effect)
            store.residuals .= store.predicted_rates - progData.response
        end
        
        n, p = size(progData.design)
        fill!(store.grad, 0)
        for i in 1:n
            store.grad .+= store.residuals[i] .* view(progData.design, i, :)
        end
    end

    @doc """
        objgrad!(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at `x`, and gradient of the objective 
        function at `x`. Gradient value is stored in `store.grad` and returns 
        the objective value. If `recompute = true`, the values relating to the 
        gradient are recomputed and saved in `store`. These are also used for 
        the objective. If `recompute = false`, then the values in `store` are 
        used to compute the gradient and objective.
    """
    function NLPModels.objgrad!($(args...); recompute::Bool = true) where {T, S}
        NLPModels.grad!(progData, precomp, store, x; recompute = recompute)
        return NLPModels.obj(progData, precomp, store, x; recompute = false)
    end

    @doc """
        hess!(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
        
    Computes the Hessian of the objective function at `x`. The hessian is stored 
        in `store.hess`. If `recompute = true`, then values needed to compute 
        the hessian in `store` are recomputed and used. Otherwise, the values 
        in `store` are used to compute the hessian.
    """
    function hess!($(args...); recompute::Bool = true) where {T,S}
        increment!(progData, :neval_hess)
        
        if recompute
            store.linear_effect .= progData.design * x
            store.predicted_rates .= exp.(store.linear_effect)
        end

        n = size(progData.design, 1)
        for i in 1:n
            store.hess .+= store.predicted_rates[i] .* view(precomp.obs_obs_t, i, :, :) 
        end
    end
end
