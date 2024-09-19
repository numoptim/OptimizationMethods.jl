# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implement constant step size gradient descent

"""
    constant_step_size_gd(func, x0, max_iter; alfa0)

Implementation of constant step size method using negative gradient directions.

# Arguments
- `func :: AbstractNLPModel{T, S}`, function to optimize. Must have grad! implemented.
- `x0 :: S`, initial starting value.
- `max_iter :: Int64`, max iteration limit.
- `alfa0 :: T = 1e-4` (Optional), constant step size.
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
        xk .-= alfa0 .* gk
        k += 1
    end

    return xk
end