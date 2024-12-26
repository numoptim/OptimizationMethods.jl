# Date: 12/26/2024
# Author: Christian Varner
# Purpose: Implement first derivatives for variance functions found
# in variance_functions.jl

"""
    TODO
"""
function dlinear_plus_sin(μ::T) where {T}
    return 1 + cos(2 * pi * μ) * 2 * pi
end