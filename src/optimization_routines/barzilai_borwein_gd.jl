# Date: 09/16/2024
# Author: Christian Varner
# Purpose: Implement barzilai-borwein.

"""
    barzilai_borwein_gd(progData, x, max_iter; alfa0, long)

Implementation of barzilai-borwein step size method using negative gradient
directions. To see more about the method, take a look at:

Barzilai and Borwein. "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis.

The method will take advantage of precomputed values and allocated space initialized 
by calling `initialize(progData)` (see documentation for problems).

## Arguments
- `progData::AbstractNLPModel{T, S}`, function to optimize
- `x::S`, initial starting value
- `max_iter::Int64`, max iteration limit
- `alfa0::T = 1e-4` (Optional), initial step size
- `long::Bool = true` (Optional), flag to indicate the use of the long version or the short version
"""
function barzilai_borwein_gd(
    progData::AbstractNLPModel{T, S},  # objective function
    x::S,                              # initial point
    max_iter::Int64;                   # max iteration
    alfa0::T = T(1e-4),                   # initial step size
    long::Bool = true                  # whether to use long or short step sizes
) where S <: Vector{T} where T <: Real

    # step size helper functions -- long variant of step size
    function _long_step_size(Δx::S, Δg::S)
        return (Δx' * Δx) / (Δx' * Δg)
    end

    # step size helper function -- short variant of step size
    function _short_step_size(Δx::S, Δg::S)
        return (Δx' * Δg) / (Δg' * Δg)
    end

    # get function
    step_size = long ? _long_step_size : _short_step_size

    # initialize progData
    precomp, store = OptimizationMethods.initialize(progData)

    # initializations -- iterate
    xprev :: S = zeros(T, size(x))
    xk :: S = zeros(T, size(x))
    xprev .= xk .= x

    # initialization -- gradient
    gprev :: S  = zeros(T, size(x))

    # initializations -- step size
    alfak :: T = zero(T)

    # first iteration
    OptimizationMethods.grad!(progData, precomp, store, xk) # TODO - implement function wrapper to get around this
    xk .-= alfa0 .* store.grad
    gprev .= store.grad

    # main iteration
    k = 2
    while k <= max_iter
        # one iteration of barzilai-borwein
        OptimizationMethods.grad!(progData, precomp, store, xk) 

        # use buffer to compute step size efficiently
        xprev .*= -1
        xprev .+= xk

        gprev .*= -1
        gprev .+= store.grad

        # compute step size
        alfak = step_size(xprev, gprev) 

        # do not update with a nan alfak
        if isnan(alfak) || isinf(alfak)
            return xk
        end
        xprev .= xk
        xk .-= alfak .* store.grad 
        gprev .= store.grad
        
        # update iteration number
        k += 1
    end

    # return
    return xk
end