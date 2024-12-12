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

- `μ::T`, scalar. In regression context, this is the linear effect.
- `p::T`, scalar. Power applied to `μ^2`.
- `c::T`, scalar. Translate the effect on variance by `c`.

!!! note
    For `c` negative this might not be a valid variance function.
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
    V(\\mu) = 1 + \\mu + \\text{sin}(2\\pi\\mu).
```
See [Quasi-likelihood Estimation](@ref) for details.

# Arguments

- `μ::T`, scalar. In regression context, this is the linear effect.
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

- `μ::T`, scalar. In regression context, this is the linear effect.
- `p::T`, scalar. Power applied to `(μ-c)^2`.
- `c::T`, scalar. Center where noise level is highest.

!!! note
    For `c` negative this might not be a valid variance function.
    For `p` smaller than ``.5``, the variance function is not continuously 
    differentiable at ``0``.
"""
function centered_exp(μ::T, p::T, c::T) where {T}
    return exp(-abs(μ-c)^(2*p))
end

"""
    centered_shifted_log(μ::T, p::T, c::T) where {T}

Implements the variance function
```math
    V(\\mu) = \\log(|\\mu-c|^{2*p} + 1).
```
See [Quasi-likelihood Estimation](@ref) for details.

# Arguments

- `μ::T`, scalar. In regression context, this is the linear effect.
- `p::T`, scalar. Power applied to `μ^2`.
- `c::T`, scalar. Cneter where noise level is lowest.

!!! note
    For `c` negative this might not be a valid variance function.
    For `p` smaller than ``.5``, the variance function is not continuously 
    differentiable at ``0``.
"""
function centered_shifted_log(μ::T, p::T, c::T) where {T}
    return log(abs(μ-c)^(2*p) + 1)
end 