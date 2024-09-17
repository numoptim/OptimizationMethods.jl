# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement constant step size gradient descent

"""
TODO - Documentation
"""
function constant_step_size_gd(
    func :: AbstractNLPModel{T, S},
    x0 :: S,
    max_iter :: Int64;
    alfa0 :: T = 1e-4
) where S <: Vector{T} where T <: Real

    # iterate initialization
    xk = zeros(T, size(x0))
    xk .= x0

    # gradient buffer
    gk = zeros(T, size(x0))

    # do constant step size gradient descent
    k = 1
    while k <= max_iter
        grad!(gk, func, xk)
        xk .-= alfa0 * gk
        k += 1
    end

    return xk
end