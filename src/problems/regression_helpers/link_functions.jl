# Date: 12/11/2024
# Author: Christian Varner
# Purpose: To reduce code duplicity in the implementation
# of quasi-likelihood methods, we want a code base for link
# functions.

"""
    logistic(η::T} where T

Implements

```math
 \\mathrm{logistic}(\\eta) = \\frac{1}{1 + \\exp(-\\eta)},
```
where `T` is a scalar value.
"""
function logistic(η::T) where T
    return 1/(1 + exp(-η))
end