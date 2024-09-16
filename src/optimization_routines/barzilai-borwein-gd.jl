# Date: 09/16/2024
# Author: Christian Varner
# Purpose: Implement barzilai-borwein.

export barzilai_borwein_gd

function barzilai_borwein_gd(
    func::AbstractNLPModel,     # objective function
    x0::AbstractVector,         # initial point
    max_iter::Int64;            # max iteration
    alfa0::Float64 = 1e-4,      # initial step size
    long::Bool = true           # whether to use long or short step sizes
)

    # step size helper functions
    function _long_step_size(Δx::AbstractVector, Δg::AbstractVector) # long variant
        return (Δx' * Δx) / (Δx' * Δg)        # ||x_k - x_{k-1}||_2^2 / (x_k - x_{k-1}) * (gk - gk-1)
    end

    function _short_step_size(Δx::AbstractVector, Δg::AbstractVector) # short variant
        return (Δx' * Δg) / (Δg' * Δg)
    end

    # get function
    step_size = long ? _long_step_size : _short_step_size

    # initializations -- iterate
    buffer = zeros(size(x))
    xprev = zeros(size(x))
    xk = zeros(size(x))
    xprev .= xk .= x0

    # initialization -- gradient
    gprev = zeros(size(x))
    gk = zeros(size(x))

    # first iteration
    gprev .= grad(func, xk) 
    xk .-= alfa0 * gprev

    # main iteration
    for k in 2:max_iter
        gk .= grad(func, xk)
        buffer .= xk - step_size(Δx = xk - xprev, Δg = gk - gprev) * gk
        xprev .= xk
        xk .= buffer
        gprev .= gk
    end

    # return
    return x
end