# Date: 12/26/2024
# Author: Christian Varner
# Purpose: Implement first derivatives for variance functions found
# in variance_functions.jl

"""
    dlinear_plus_sin(μ::T) where {T}

Compute the following function
```math
    \\frac{d}{d\\mu} (1 + \\mu + \\sin(2 \\pi \\mu)) = 1 + 2 \\pi \\cos(2\\pi \\mu).
```

# Arguments

- `μ::T`, point at which to compute the derivative. In the context of 
    regression, this is the mean.
"""
function dlinear_plus_sin(μ::T) where {T}
    return T(1 + cos(2 * pi * μ) * 2 * pi)
end

"""
    dcentered_exp(μ::T, p::T, c::T) where {T}

Compute the following function
```math
    \\frac{d}{d\\mu} \\exp\\left( -|\\mu - c|^{2p} \\right),
```
where ``c \\in \\mathbb{R}`` and ``p \\in \\mathbb{R}``. See [Quasi-likelihood Estimation](@ref) for details.

!!! warning
    For `p` smaller than ``.5``, the derivative is not well-defined at ``c``.

# Arguments

- `μ::T`, scalar. In the regression context, this is the estimated mean of a 
    datapoint.
- `p::T`, scalar. Power applied to `(μ-c)^2`.
- `c::T`, scalar. Center where noise level is highest.
"""
function dcentered_exp(μ::T, p::T, c::T) where {T}
    return -centered_exp(μ, p, c) * sign(μ-c) * 2 * p * abs(μ-c)^(2*p-1)
end