# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Implementing diminishing step size gradinet descent

"""
TODO - Documentation
"""
function diminishing_step_size_gd(
    func :: AbstractNLPModel{T, S},
    x0 :: S,
    max_iter :: Int64;
    step_size :: Union{Function, Nothing} = nothing
) where S <: Vector{T} where T <: Real

    function default_step_size(k :: Int64)
        return 1/k
    end

    if isnothing(step_size)
        step_size = default_step_size
    end

    # iterate initialization
    xk = zeros(T, size(x0))
    xk .= x0

    # gradient buffer
    gk = zeros(T, size(x0))

    # do diminishing step size
    k = 1
    while k <= max_iter
        grad!(gk, func, sk)
        xk .-= step_size(k) * gk # TODO - I am not sure what parameters should be passed to this
        k += 1
    end

    return xk
end