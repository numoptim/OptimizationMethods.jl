# Date: 12/11/2024
# Author: Christian Varner
# Purpose: To reduce code duplicity in the implementation
# of quasi-likelihood methods, we want a code base for link
# functions.

"""
    logistic(η::T} where {T}

Implements
```math
 \\mathrm{logistic}(\\eta) = \\frac{1}{1 + \\exp(-\\eta)},
```
where `T` is a scalar value.

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function logistic(η::T) where {T}
    return 1/(1 + exp(-η))
end

"""
    inverse_complimentary_log_log(η::T) where {T}

Implements the link function
```math
    g(\\eta) = 1 - \\exp(-\\exp(\\eta)).
```

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function inverse_complimentary_log_log(η::T) where {T}
    return 1-exp(-exp(η))
end

"""
    inverse_probit(\\eta::T) where {T}

Implements the link function
```math
    g(\\eta) = \\frac{1}{2\\pi} \\exp(-.5\\eta^2).
```

# Arguments

- `η::T`, scalar. In the regression context this is the linear effect.
"""
function inverse_probit(η::T) where {T}
    return (1/sqrt(2 * pi)) * exp(-((η)^2)/2)
end