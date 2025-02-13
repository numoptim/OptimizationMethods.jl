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
    dcentered_shifted_log(μ::T, p::T, c::T) where {T}

Compute and returns the following function
```math
    \\frac{d}{d\\mu} \\log(|\\mu-c|^{2p} + 1) =
    sign(\\mu - c) \\frac{2p|\\mu - c|^{2p-1}}{|\\mu-c|^{2p} + 1},
```
where ``sign(x)`` returns the sign of ``x``. This equality is only 
correct everywhere when ``p > .5``.

!!! warning
    The function does not check the correctness of `p` and is not guaranteed
    to return the correct derivative at ``c`` when `p <= .5`.

# Arguments

- `μ::T`, point at which to compute the derivative. In the context of 
    regression, this is the mean.
- `p::T`, scalar. Power applied to ``|\\mu-c|^2``.
- `c::T`, scalar. Center where noise level is lowest.
"""
function dcentered_shifted_log(μ::T, p::T, c::T) where {T}
    return sign(μ - c) * (2 * p * abs(μ - c)^(2 * p -1))/(abs(μ - c)^(2 * p) + 1)
end 