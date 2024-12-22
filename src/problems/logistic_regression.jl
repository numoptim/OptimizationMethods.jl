# OptimizationMethods.jl

"""
    LogisticRegression{T,S} <: AbstractNLPModel{T,S}

Implements logistic regression problem with canonical link function. If
    the covariate (i.e., design) matrix and response vector are not supplied,
    then these are simulated. 

# Objective Function 

Let ``A`` be the covariate matrix and ``b`` denote the response vector. The
    rows of ``A`` and corresponding entry of ``b`` correspond to 
    the predictors and response for the same experimental unit. 
    Let ``A_i`` be the vector that is row `i` of ``A``,
    and let ``b_i`` be the `i` entry of ``b``. Note, ``b_i`` is either `0` or 
    `1`.

Let ``\\mu(x)`` denote a vector-valued function whose `i`th entry is 
```math
\\mu_i(x) = \\frac{1}{1 + \\exp(-A_i^\\intercal x)}.
```

If ``A`` has ``n`` rows (i.e., ``n`` is the number of observations), 
then the objective function is negative log-likelihood function given by

```math
F(x) = -\\sum_{i=1}^n b_i \\log( \\mu_i(x) ) + (1 - b_i) \\log(1 - \\mu_i(x)).
```

# Fields

- `meta::NLPModelMeta{T,S}`, data structure for nonlinear programming models
- `counters::Counters`, counters for a nonlinear programming model
- `design::Matrix{T}`, the design matrix, ``A``, of the logistic regression
    problem
- `response::Vector{Bool}`, the response vector, ``b``, of the logistic
    regression problem

# Constructors

    LogisticRegression(::Type{T}; nobs::Int64 = 1000, nvar::Int64 = 50) where T

Constructs a simulated problem where the number of observations is `nobs` and
    the dimension of the parameter is `nvar`. The generated design matrix's
    first column is all `1`s. The remaining columns are independent random
    normal entries such that each row (excluding the first entry) has unit
    variance. The design matrix is stored as type `Matrix{T}`.

    LogisticRegression(design::Matrix{T}, response::Vector{Bool};
        x0::Vector{T} = ones(T, size(design, 2)) ./ sqrt(size(design, 2))
        ) where T

Constructs a `LogisticRegression` problem with design matrix `design` and 
    response vector `response`. The default initial iterate, `x0` is 
    a scaling of the vector of ones. `x0` is optional. 
"""
mutable struct LogisticRegression{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    design::Matrix{T}
    response::Vector{Bool}
end
function LogisticRegression(
    ::Type{T};
    nobs::Int64 = 1000,
    nvar::Int64 = 50
) where T

    meta = NLPModelMeta(
        nvar,
        name = "Logistic Regression",
        x0 = ones(T, nvar) ./ T(sqrt(nvar)),
    )

    # Constructs a design matrix of dimension nobs by nvar
    # Design matrix's first column is 1s
    # The remaining entries come from a normal distribution such that 
    # the variance of any row (excluding the first column) is 1. 
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ T(sqrt(nvar - 1)))

    # Simulated solution (note, this may not be the solution to the optimization
    # problem, nor the maximum likelihood estimator)
    β = randn(T, nvar) / T(sqrt(nvar))

    # Linear effect 
    η = design * β

    # Probabilities 
    p = 1 ./ (1 .+ exp.(-η))

    # Simulated response vector 
    response = Bool[rand() <= p[i] for i = 1:nobs]

    return LogisticRegression(
        meta,
        Counters(),
        design,
        response
    )
end
function LogisticRegression(
    design::Matrix{T},
    response::Vector{Bool};
    x0::Vector{T} = ones(T, size(design, 2)) ./ T(sqrt(size(design, 2)))
) where T

    nobs, nvar = size(design)

    # Check dimension of parameter and number of columns of design matrix are 
    # the same 
    @assert nvar == length(x0) "Design matrix has $nvar columns while the 
    initial value is of length $(length(x0)). Must be the same dimension."

    # Check that the number of rows of design matrix and number of responses 
    # are the same
    @assert nobs == length(response) "Design matrix has $nobs observations 
    while the response vector has $(length(response)) observations. Must be 
    the same dimension."
    
    meta = NLPModelMeta(
        nvar,
        name = "Logistic Regression",
        x0 = x0,
    )

    return LogisticRegression(
        meta,
        Counters(),
        design,
        response
    )
end

"""
    PrecomputeLogReg{T} <: AbstractPrecompute{T}

Immutable structure for initializing and storing repeatedly used calculations.
    Nothing is stored for logistic regression.
"""
struct PrecomputeLogReg{T} <: AbstractPrecompute{T}
end

"""
    AllocateLogReg{T} <: AbstractProblemAllocate{T}

Immutable structure for preallocating important quantities for the logistic
    regression problem. 

# Fields 

- `linear_effect::Vector{T}`, stores the estimated linear effect
- `probabilities::Vector{T}`, stores the estimated probabilities of observing `1` 
- `residuals::Vector{T}`, stores the difference between the responses and the 
    estimated probabilities.
- `grad::Vector{T}`, stores the gradient vector 
- `hess::Matrix{T}`, stores the Hessian matrix

# Constructor 

    AllocateLogReg(prog::LogisticRegression{T,S}) where {T,S}

Preallocates the data structures for optimizing the logistic regression function.
"""
struct AllocateLogReg{T} <: AbstractProblemAllocate{T}
    linear_effect::Vector{T}
    probabilities::Vector{T}
    residuals::Vector{T}
    grad::Vector{T}
    hess::Matrix{T}
end

function AllocateLogReg(progData::LogisticRegression{T,S}) where {T,S}
    nobs = size(progData.design, 1)
    nvar = progData.meta.nvar

    return AllocateLogReg(
        zeros(T, nobs),
        zeros(T, nobs),
        zeros(T, nobs),
        zeros(T, nvar),
        zeros(T, nvar, nvar)
    )
end

"""
    initialize(progData::LogisticRegression{T,S}) where {T,S}

Generates the precomputed and storage structs given a Logistic Regression
    problem.
"""
function initialize(progData::LogisticRegression{T,S}) where {T,S}
    precompute = PrecomputeLogReg{T}()
    store = AllocateLogReg(progData)

    return precompute, store
end

###############################################################################
# Operations that are not in-place. Does not make use of precomputed values.
###############################################################################

args = [
    :(progData::LogisticRegression{T,S})
    :(x::Vector{T})
]

@eval begin

    @doc """
        obj(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args...)) where {T,S}
        increment!(progData, :neval_obj)
        η = progData.design * x
        η .= logistic.(η)
        l = T(0)
        for i = 1:size(progData.design, 1)
            l += progData.response[i] ? log(η[i]) : log(1 - η[i])
        end

        return -l
    end

    @doc """
        grad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`.
    """
    function NLPModels.grad($(args...)) where {T,S}
        increment!(progData, :neval_grad)
        η = progData.design * x 
        η .= logistic.(η)
        return -progData.design' * (progData.response - η)
    end

    @doc """
         objgrad(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        values returned are the objective function value followed by the 
        gradient function value. 
    """
    function NLPModels.objgrad($(args...)) where {T,S}
        increment!(progData, :neval_obj)
        increment!(progData, :neval_grad)
        η = progData.design * x
        η .= logistic.(η)
        l, g = T(0), zeros(T, length(x))
        for i = 1:size(progData.design, 1)
            l += progData.response[i] ? log(η[i]) : log(1 - η[i])
            g += (progData.response[i] - η[i]) * progData.design[i,:]
        end

        return -l, -g
    end

    @doc """
        hess(
            $(join(string.(args),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`.
    """
    function hess($(args...)) where {T,S}
        increment!(progData, :neval_hess)
        η = progData.design * x
        η .= logistic.(η)
        η .= η .* (1 .- η)

        return progData.design'*(η .* progData.design)
    end
end

###############################################################################
# Operations that are not in-place. Makes use of precomputed values. 
###############################################################################

args_pre = [
    :(progData::LogisticRegression{T,S}),
    :(preComp::PrecomputeLogReg{T}),
    :(x::Vector{T})
]

funcs = Dict(
    :(NLPModels.obj) => "Computes the objective function value at `x`.",
    :(NLPModels.grad) => "Computes the gradient function value at `x`.",
    :(NLPModels.objgrad) => "Computes the objective and gradient at `x`. 
        Returns the objective value, then the gradient value.",
    :(hess) => "Computes the Hessian value at `x`."
)

for (func, desc) in funcs
    
    #Precompute does not have any additional information.
    #This loop will call the version of the function that uses only 
    #progData and x.
    @eval begin
        @doc """
            $(string($func))(
                $(join(string.(args_pre),",\n\t    "))
            ) where {T,S}
        
        $($desc)
        """
        function $func($(args_pre...)) where {T,S}
            return $func(progData, x)
        end

    end
end 

###############################################################################
# Operations that are in-place. Makes use of precomputed values. 
###############################################################################
  
args_store = [
    :(progData::LogisticRegression{T,S}),
    :(preComp::PrecomputeLogReg{T}),
    :(store::AllocateLogReg{T}),
    :(x::Vector{T})
]


@eval begin 

    @doc """
        obj(
            $(join(string.(args_store),"\n\t    "))
        ) where {T,S}

    Computes the objective function at the value `x`.
    """
    function NLPModels.obj($(args_store...)) where {T,S}
        increment!(progData, :neval_obj)
        store.linear_effect .= progData.design * x
        store.probabilities .= logistic.(store.linear_effect)
        l = T(0)
        for i = 1:size(progData.design, 1)
            l += progData.response[i] ? log(store.probabilities[i]) : 
                log(1 - store.probabilities[i])
        end

        return -l
    end

    @doc """
        grad!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}

    Computes the gradient function value at `x`. The gradient is computed in
        place and saved in `store.grad`.
    """
    function NLPModels.grad!($(args_store...)) where {T,S}
        increment!(progData, :neval_grad)
        store.linear_effect .= progData.design * x 
        store.probabilities .= logistic.(store.linear_effect)
        store.residuals .= progData.response - store.probabilities

        # Update gradient 
        store.grad .= -(progData.design' * store.residuals)

        return nothing 
    end

    @doc """
         objgrad!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}

    Computes the objective function and gradient function value at `x`. The
        objective function value is returned. The gradient function value is 
        stored in `store.grad`. 
    """
    function NLPModels.objgrad!($(args_store...)) where {T,S}
        increment!(progData, :neval_obj)
        increment!(progData, :neval_grad)

        store.linear_effect .= progData.design * x 
        store.probabilities .= logistic.(store.linear_effect)
        store.residuals .= progData.response - store.probabilities

        l = T(0)
        store.grad .= zeros(T, length(x))

        for i = 1:size(progData.design, 1)
            l += progData.response[i] ? log(store.probabilities[i]) : 
                log(1 - store.probabilities[i])
            store.grad .-= store.residuals[i] * progData.design[i,:]
        end

        return -l
    end


    @doc """
        hess!(
            $(join(string.(args_store),",\n\t    "))
        ) where {T,S}
    
    Computes the Hessian function value at `x`. The Hessian function value is
        stored in `store.hess`.
    """
    function hess!($(args_store...)) where {T,S}
        increment!(progData, :neval_hess)
        store.linear_effect .= progData.design * x
        store.probabilities .= logistic.(store.linear_effect)
        
        store.hess .= zeros(T, length(x), length(x))
        for i = 1:size(progData.design, 1)
            store.hess .+= store.probabilities[i] * (1 - store.probabilities[i]) *
                progData.design[i,:] * progData.design[i,:]'
        end

        return nothing
    end


end
