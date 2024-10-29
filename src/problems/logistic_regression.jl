# OptimizationMethods.jl

"""
    LogisticRegression{T,S} <: AbstractNLPModel{T,S}

Implements logistic regression problem with canonical link function. If
    the covariate (i.e., design) matrix and response vector are not supplied,
    then these are simulated. 

# Objective Function 

Let ``A`` be the covariate matrix and ``b`` denote the response vector. The
    rows of ``A`` and corresponding entry of ``b`` correspond to the same 
    observation. 
    Let the ``A_i`` be the vector that is row `i` of ``A``,
    and let ``b_i`` be the `i` entry of ``b``. Note, ``b_i`` is either `0` or 
    `1`.

Let ``\\mu(x)`` denote a vector-valued function whose `i`th entry is 
```math
\\mu_i(x) = \\frac{1}{1 + \\exp(-A_i^\\intercal x)}.
```

If ``A`` has ``n`` rows (i.e., ``n`` is the number of observations), 
then the objective function is negative log-likelihood function given by

```math
F(x) = -\\sum_{i=1}^n b_i \\log( \\mu_i(x) ) + (1 - b_i) \\log(1- \\mu_i(x)).
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
    meta::NLPModelMeta{T,S}
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
        name = "LogisticRegression",
        x0 = ones(T, nvar) ./ sqrt(nvar),
    )

    # Constructs a design matrix of dimension nobs by nvar
    # Design matrix's first column is 1s
    # The remaining entries come from a normal distribution such that 
    # the variance of any row (excluding the first column) is 1. 
    design = hcat(ones(T, nobs), randn(T, nobs, nvar-1) ./ sqrt(nvar - 1))

    # Simulated solution (note, this may not be the solution to the optimization
    # problem, nor the maximum likelihood estimator)
    β = randn(nvar) / sqrt(nvar)

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
    x0::Vector{T} = ones(T, size(design, 2)) ./ sqrt(size(design, 2))
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

# TODO: Pre-allocation struct 
"""
    AllocateLogReg{T} <: AbstractProblemAllocate{T}

Immutable structure for preallocating important quantities for the logistic
    regression problem. 
"""
struct AllocateLogReg{T} <: AbstractProblemAllocate{T}
    weighted_design::Matrix{T}
    weighted_response::Vector{T}
    probabilities::Vector{T}
end

# TODO: Function initialize

################################################################################
# Utilities
################################################################################
"""
    logit(η::T} where T

Implements
    ```math
        \\mathrm{logit}(\\eta) = \\frac{1}{1 + \\exp(-\\eta)},
    ```
    where `T` is a scalar value.
"""
function logit(η::T) where T
    return 1/(1 + exp(-η))
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
    
    Computes the objective function at the value `x`
    """
    function NLPModels.obj($(args...)) where {T,S}
        η = progData.design * x
        η .= logit.(η)
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

    Computes the gradient function value at `x`
    """
    function NLPModels.grad($(args...)) where {T,S}
        η = progData.design * x 
        η .= logit.(η)
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
        η = progData.design * x
        η .= logit.(η)
        l, g = T(0), zeros(T, length(x))
        for i = 1:size(progData.design, 1)
            l += progData.response[i] ? log(η[i]) : log(1 - η[i])
            g += (progData.response[i] - η[i]) * progData.design[i,:]
        end

        return -l, -g
    end

    #TODO: hess
    @doc """
        
    """


end