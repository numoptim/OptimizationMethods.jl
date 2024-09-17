# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement weighted norm dampening for gradient descent

"""
TODO - documentation
"""
function weighted_norm_dampening(
    func :: AbstractNLPModel{T, S},
    x0 :: S,
    max_iter :: Int64;
    bk :: T = 1.0
) where S <: Vector{T} where T <: Real

    # initialization of point
    xk = zeros(T, size(x0))
    xk .= x0

    # initialization of buffer
    gk = zeros(T, size(x0))
    grad!(gk, func, xk)

    # main loop
    while k <= max_iter
        xk .-= (1/bk) * gk
        grad!(gk, func, xk)
        bk = bk + (norm(gk)^2)/bk # TODO: possible allocations
        k += 1
    end

    # return value
    return xk
end