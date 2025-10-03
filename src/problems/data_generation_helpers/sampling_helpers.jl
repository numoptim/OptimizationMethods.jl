"""
    inverse_logistic(μ)

Compute the inverse of the logistic function.
"""
function inverse_logistic(μ::Float64)
    return log(μ/(1-μ))
end

"""
    get_design(a::Float64, nobs::Int64, nvar::Int64)

Generate a design matrix and true coefficients for the quasi-likelihood
    logistic regression problem.
"""
function get_design(a::Float64, nobs::Int64, nvar::Int64)
    ub = sqrt(inverse_logistic(1-a)/nvar)

    x = (ub) .* rand(nobs, nvar)
    β = (ub) .* rand(nvar)

    return x, β
end

"""
    get_noise(a::Float64, nobs::Int64, vmax::Float64)

Generate noise for the quasi-likelihood regression problem. Added to
    response.
"""
function get_noise(a::Float64, nobs::Int64, vmax::Float64)
    lb = -a/sqrt(vmax)
    ub = a/sqrt(vmax)
    ϵ = (ub - lb) .* rand(nobs) .+ lb
    return ϵ
end