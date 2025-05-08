# Date: 12/26/2024
# Author: Christian Varner
# Purpose: Functions that compute the first and second derivatives
# of the link functions found in link_functions.jl

"""
    dlogistic(η::T) where {T}

First derivative of the `logistic` function. Implements
```math
    \\frac{d}{d\\eta}  \\mathrm{logistic}(\\eta) = \\frac{\\exp(-\\eta)}{(1+\\exp(-\\eta))^2}
```
where `T` is a scalar type.

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function dlogistic(η::T) where {T}
    if -η > 709
        @warn "The input to this function is large, therefore a NaN will be produced."
    end 
    return exp(-η)/((1 + exp(-η))^2)
end

"""
    ddlogistic(η::T) where {T}

Double derivative of the `logistic` function. Implements
```math
    \\frac{d}{d^2\\eta} \\mathrm{logistic}(\\eta) = 
    \\frac{2\\exp(-2\\eta)}{(1+\\exp(-\\eta))^3} -
    \\frac{\\exp(-\\eta)}{(1+\\exp(-\\eta))^2}
```
where `T` is a scalar type.

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function ddlogistic(η::T) where {T}
    return -exp(-η)/((1+exp(-η))^2) + 2*exp(-2*η)/((1+exp(-η))^3)
end