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