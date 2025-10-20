# Date: 2025/10/08
# Author: Christian Varner
# Purpose: Functionality for sampling data for quasi-likelihood problems
# constraining the responses to be in (0, 1).

"""
    inverse_logistic(μ)

Compute the inverse of the logistic function.

# Arguments
- `μ::Float64`, A value in the interval (0, 1).

# Returns
- `η::Float64`, The inverse logistic of `μ`.
"""
function inverse_logistic(μ::Float64)
    return log(μ/(1-μ))
end

"""
    get_design(a::Float64, nobs::Int64, nvar::Int64)

Generate a design matrix and true coefficients for the quasi-likelihood
    logistic regression problem. Data is generated such that the
    linear predictor `η = x * β` lies in the interval
    `[inverse_logistic(a), inverse_logistic(1-a)]` for a given 
    `a` in `(0, 0.5)`.

# Arguments
- `a::Float64`, A value in the interval (0, 0.5).
- `nobs::Int64`, The number of observations to generate. 
- `nvar::Int64`, The number of variables to generate.

# Returns
- `x::Matrix{Float64}`, The design matrix of size `(nobs, nvar)`.
- `β::Vector{Float64}`, The true coefficients of size `(nvar,)`.  
"""
function get_design(a::Float64, nobs::Int64, nvar::Int64)
    ub = inverse_logistic(1-a)/nvar
    lb = inverse_logistic(a)/nvar 

    x = rand(nobs, nvar)
    β = (ub - lb) .* rand(nvar) .+ lb

    return x, β
end

"""
    get_noise(a::Float64, nobs::Int64, vmax::Float64)

Generate noise for the quasi-likelihood regression problem. Added to
    response. Noise is generated such that `V(μ)ϵ` lies in the interval
    `[-a, a]` for a given `a` in `(0, 0.5)`, where `V(μ)` is the variance
    function of the regression model.

# Arguments
- `a::Float64`, A value in the interval (0, 0.5).
- `nobs::Int64`, The number of errors to generate.
- `vmax::Float64`, upper bound on the variance function
    of the regression model in `[-a, a]`.

# Returns
- `ϵ::Vector{Float64}`, The generated noise of size `(nobs,)
"""
function get_noise(a::Float64, nobs::Int64, vmax::Float64)
    lb = -a/sqrt(vmax)
    ub = a/sqrt(vmax)
    ϵ = (ub - lb) .* rand(nobs) .+ lb
    return ϵ
end