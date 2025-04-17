# Date: 12/11/2024
# Author: Christian Varner
# Purpose: File to put variance functions for quasi-likelihood regression

"""
    monomial_plus_constant(μ::T, p::T, c::T) where {T}

Implements the variance function
```math
    V(\\mu) = (\\mu^{2})^p + c.
```
See [Quasi-likelihood Estimation](@ref) for details.

# Arguments

- `μ::T`, scalar. In the regression context, this is the estimated mean of a 
    datapoint.
- `p::T`, scalar. Power applied to `μ^2`.
- `c::T`, scalar. Translate the effect on variance by `c`.

!!! note
    For a negative `c` this might not be a valid variance function.
    For `p` smaller than ``.5``, the variance function is not continuously 
    differentiable at ``0``.
"""
function monomial_plus_constant(μ::T, p::T, c::T) where {T}
    return (μ^(2))^p + c
end

"""
    linear_plus_sin(μ::T) where {T}

Implements the variance function 
```math
    V(\\mu) = 1 + \\mu + \\mathrm{sin}(2\\pi\\mu).
```
See [Quasi-likelihood Estimation](@ref) for details.

# Arguments

- `μ::T`, scalar. In the regression context, this is the estimated mean of a 
    datapoint.
"""
function linear_plus_sin(μ::T) where {T}
    return 1 + μ + T(sin(2*pi*μ))
end

"""
    centered_exp(μ::T, p::T, c::T) where {T}

Implements the variance function
```math
    V(\\mu) = \\exp\\left( -|\\mu - c|^{2p} \\right).
```
See [Quasi-likelihood Estimation](@ref) for details.

# Arguments

- `μ::T`, scalar. In the regression context, this is the estimated mean of a 
    datapoint.
- `p::T`, scalar. Power applied to `(μ-c)^2`.
- `c::T`, scalar. Center where noise level is highest.

!!! note
    For `p` smaller than ``.5``, the variance function is not continuously 
    differentiable at ``c``.
"""
function centered_exp(μ::T, p::T, c::T) where {T}
    return exp(-abs(μ-c)^(2*p))
end

"""
    centered_shifted_log(μ::T, p::T, c::T, d::T) where {T}

Implements the variance function
```math
    V(\\mu) = \\log(|\\mu-c|^{2p} + 1) + d.
```
See [Quasi-likelihood Estimation](@ref) for details.

# Arguments

- `μ::T`, scalar. In the regression context, this is the estimated mean of a 
    datapoint.
- `p::T`, scalar. Power applied to `|\\mu-c|^2`.
- `c::T`, scalar. Center where noise level is lowest.
- `d::T`, scalar. Irreducible variance. Corresponds to the minimum variance.
    Must be positive.

!!! note
    For `p` smaller than or equal to ``.5``, the variance function is not 
    continuously differentiable at ``c``. When `d <= 0`, the variance function
    is not well-defined, that is it can take on negative and zero values.
"""
function centered_shifted_log(μ::T, p::T, c::T, d::T) where {T}
    return log(abs(μ-c)^(2*p) + 1) + d
end 