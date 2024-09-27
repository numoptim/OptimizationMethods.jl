# Date: 09/16/2024
# Author: Christian Varner
# Purpose: Implement barzilai-borwein.

"""
    (x, stats) = barzilai_borwein_gd(progData, x, max_iter; alfa0, long)

Implementation of barzilai-borwein step size method using negative gradient
directions. To see more about the method, take a look at:

Barzilai and Borwein. "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis.

The method will take advantage of precomputed values and allocated space initialized 
by calling `initialize(progData)` (see documentation for problems).

## Arguments

- `progData::AbstractNLPModel{T, S}`, function to optimize
- `x::S`, initial starting value
- `max_iter::Int64`, max iteration limit
- `gradient_condition`, if positive, the algorithm stops once the gradient is less than or equal to `gradient_condition`. If negative the condition is not checked.
- `alfa0::T = 1e-4` (Optional), initial step size
- `long::Bool = true` (Optional), flag to indicate the use of the long version or the short version

"""
function barzilai_borwein_gd(
    progData::AbstractNLPModel{T, S},           # objective function
    x::S,                                       # initial point
    max_iter::Int64;                            # max iteration
    gradient_condition :: T = T(1e-10),         # gradient tolerance
    alfa0::T = T(1e-4),                         # initial step size
    long::Bool = true                           # whether to use long or short step sizes
) where S <: Vector{T} where T <: Real

    # step size helper functions -- long variant of step size
    function _long_step_size(Δx::S, Δg::S)
        return (Δx' * Δx) / (Δx' * Δg)
    end

    # step size helper function -- short variant of step size
    function _short_step_size(Δx::S, Δg::S)
        return (Δx' * Δg) / (Δg' * Δg)
    end

    # Initialization of simple stats
    stats = OptimizationMethods.SimpleStats(T)
    start_time = time()

    # initializations -- iterates
    xprev :: S = zeros(T, size(x))
    xk :: S = zeros(T, size(x))
    xprev .= xk .= x

    # if max_iter is negative then just return
    if max_iter <= 0
        stats.time = time() - start_time 
        stats.status_message = "max_iter is smaller than or equal to 0."
        return xk, stats
    end

    # get step size function
    step_size = long ? _long_step_size : _short_step_size

    # initializations -- gradient condition, gradient buffer, and step size
    grad_above_tol(g :: S) = (gradient_condition < 0) || (norm(g) > gradient_condition) ? true : false
    gprev :: S  = zeros(T, size(x))
    alfak :: T = zero(T)

    # initialize optimization problem with program data
    precomp, store = OptimizationMethods.initialize(progData) 

    # check if stopping conditions are already satisfied
    OptimizationMethods.grad!(progData, precomp, store, xk)
    if !grad_above_tol(store.grad)
        # update the simple stats
        stats.time = time() - start_time 
        stats.total_iters = 0
        stats.grad_norm = norm(store.grad)
        stats.nobj = progData.counters.neval_obj
        stats.ngrad = progData.counters.neval_grad
        stats.nhess = progData.counters.neval_hess

        # update the status code and message
        stats.status = (!grad_above_tol(store.grad), gradient_condition)
        stats.status_message = "Gradient at initial point was already below tolerance."

        return xk, stats
    end

    # first iteration with beginning step size
    xk .-= alfa0 .* store.grad
    gprev .= store.grad
    OptimizationMethods.grad!(progData, precomp, store, xk) 

    # main iteration
    k = 2
    while k <= max_iter && grad_above_tol(store.grad)
        # compute Δx
        xprev .*= -1
        xprev .+= xk

        # compute Δg
        gprev .*= -1
        gprev .+= store.grad

        # compute step size
        alfak = step_size(xprev, gprev) 

        # do not update with a nan or inf alfak
        if isnan(alfak) || isinf(alfak)

            # update the simple stats
            stats.time = time() - start_time 
            stats.total_iters = k - 1
            stats.grad_norm = norm(store.grad)
            stats.nobj = progData.counters.neval_obj
            stats.ngrad = progData.counters.neval_grad
            stats.nhess = progData.counters.neval_hess

            # update the status code and message
            stats.status = (!grad_above_tol(store.grad), gradient_condition)
            stats.status_message = "Termination due to step size being nan or inf."
            
            return xk, stats
        end

        # update
        xprev .= xk
        xk .-= alfak .* store.grad 
        gprev .= store.grad

        # one iteration of barzilai-borwein
        OptimizationMethods.grad!(progData, precomp, store, xk) 
        
        # update iteration number
        k += 1
    end

    # update the simple stats
    stats.time = time() - start_time 
    stats.total_iters = k - 1
    stats.grad_norm = norm(store.grad)
    stats.nobj = progData.counters.neval_obj
    stats.ngrad = progData.counters.neval_grad
    stats.nhess = progData.counters.neval_hess

    # update the status code
    stats.status = (!grad_above_tol(store.grad), gradient_condition)

    # update status message
    (k > max_iter) && (stats.status_message = "Max iteration was reached.")
    (!grad_above_tol(store.grad)) && (stats.status_message = "Gradient tolerance was reached.")

    # return data
    return xk, stats
end