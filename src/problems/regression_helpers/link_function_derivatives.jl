"""
    dlogistic(η::T) where {T}

First derivative of the `logistic` function. Implements
```math
    \\nabla \\mathrm{logistic}(\\eta) = \\frac{\\exp(-\\eta)}{(1+\\exp(-\\eta))^2}
```
where `T` is a scalar type.

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function dlogistic(η::T) where {T}
    return exp(-η)/((1 + exp(-η))^2)
end

"""
    ddlogistic(η::T) where {T}


Double derivative of the `logistic` function. Implements
```math
```
where `T` is a scalar type.

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function ddlogistic(η::T) where {T}
    # TODO
end