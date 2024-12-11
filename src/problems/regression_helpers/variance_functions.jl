# Date: 12/11/2024
# Author: Christian Varner
# Purpose: File to put variance functions for quasi-likelihood regression

"""
    monomial_plus_constant(μ::T, p::T, c::T) where {T}

Implement the variance function that is equal to
```math
    V(\\mu) = (\\mu^{2})^p + c.
```
"""
function monomial_plus_constant(μ::T, p::T, c::T) where {T}
    return (μ^(2))^p + c
end