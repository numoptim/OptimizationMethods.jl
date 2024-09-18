# Date: 09/16/2024
# Author: Christian Varner
# Purpose: Implement barzilai-borwein.

"""
    barzilai_borwein_gd(func, x0, max_iter; alfa0, long)

Implementation of barzilai-borwein step size method using negative gradient
directions. To see more about the method, take a look at

Barzilai and Borwein. "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis.

# Arguments
- `func::AbstractNLPModel{T, S}`, function to optimize. Must have grad! implemented.
- `x0::S`, initial starting value.
- `max_iter::Int64`, max iteration limit.
- `alfa0::T = 1e-4` (Optional), initial step size.
- `long::Bool = true` (Optional), flag to indicate the use of the long version or the short version
"""
function barzilai_borwein_gd(
    func::AbstractNLPModel{T, S},     # objective function
    x0::S,                            # initial point
    max_iter::Int64;                  # max iteration
    alfa0::T = 1e-4,            # initial step size
    long::Bool = true                 # whether to use long or short step sizes
) where S <: Vector{T} where T <: Real

    # step size helper functions
    function _long_step_size(Δx::S, Δg::S) # long variant
        return (Δx' * Δx) / (Δx' * Δg)        # ||x_k - x_{k-1}||_2^2 / (x_k - x_{k-1}) * (gk - gk-1)
    end

    function _short_step_size(Δx::S, Δg::S) # short variant
        return (Δx' * Δg) / (Δg' * Δg)
    end

    # get function
    step_size = long ? _long_step_size : _short_step_size

    # initializations -- iterate
    xprev :: S = zeros(T, size(x0))
    xk :: S = zeros(T, size(x0))
    xprev .= xk .= x0

    # initialization -- gradient
    gprev :: S  = zeros(T, size(x0))
    gk :: S = zeros(T, size(x0))

    # initializations -- step size
    alfak :: T = zero(T)

    # first iteration
    grad!(gprev, func, xk) 
    xk .-= alfa0 * gprev

    # main iteration
    k = 2
    while k <= max_iter
        # one iteration of barzilai-borwein
        grad!(gk, func, xk)

        # use buffer to compute step size efficiently
        xprev .*= -1
        xprev .+= xk

        gprev .*= -1
        gprev .+= gk

        # compute step size
        alfak = step_size(xprev, gprev) 

        # do not update with a nan alfak
        if isnan(alfak)
            return xk
        end
        xprev .= xk
        xk .-= alfak .* gk 
        gprev .= gk
        
        # update iteration number
        k += 1
    end

    # return
    return xk
end