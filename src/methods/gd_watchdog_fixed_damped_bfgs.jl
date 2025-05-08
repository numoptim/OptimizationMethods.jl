# Date: 2025/05/07
# Author: Christian Varner
# Purpose: Implementation of damped BFGS

"""
    WatchdogFixedDampedBFGSGD{T} <: AbstractOptimizerData{T}

Mutable structure that parameterizes gradient descent with fixed
    step size and damped BFGS directions. The structure also stores and tracks
    values during the progress of applying the method to an optimization
    problem.

# Fields

- `name::String`, name of the optimizer for reference.
- `F_θk::T`, objective function value at the beginning of the inner loop
    for one of the inner loop stopping condition.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `B_θk::Matrix{T}`, buffer matrix for the BFGS approximation prior to the
    start of the inner loop. This is saved in case bactracking is required,
    making the next approximation dependent on this value.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `c::T`, initial factor used in the approximation of the Hessian.
- `Bjk::Matrix{T}`, buffer matrix for the damped BFGS approximation in the
    inner loop.
- `δBjk::Matrix{T}`, buffer matrix for the update term added to the BFGS 
    approximation.
- `rjk::Vector{T}`, buffer vector for the update term in the damped BFGS
    approximation.
- `sjk::Vector{T}`, buffer vector for a term used in the damped BFGS approximation.
    Should correspond to the difference of consecutive iterates in the 
    inner loop.
- `yjk::Vector{T}`, buffer vector for a term used in the damped BFGS approximation.
    Should correspond to the difference of gradient values between 
    consecutive iterates in the inner loop.
- `djk::Vector{T}`, buffer vector used to store the step used in the inner
    loop.
- `α::T`, fixed step size used in the inner loop.
- `δ::T`, step size reduction parameter used in the line search routine. 
- `ρ::T`, parameter used in backtracking and the watchdog condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `line_search_max_iterations::Int64`, maximum number of line search
    iterations
- `max_distance_squared::T`, maximum distance between the starting inner loop
    iterates and the rest of the inner loop iterates. Used in the watchdog condition.
- `η::T`, term used in the stopping conditions for the inner loop.
- `inner_loop_max_iterations::Int64`, maximum number of iterations in the
    inner loop.
- `objective_hist::CircularVector{T, Vector{T}}`, vector of previous accepted 
    objective values for non-monotone cache update.
- `reference_value::T`, the maximum objective value in `objective_hist`.
- `reference_value_index::Int64`, the index of the maximum value in `objective_hist`.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    is at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
- `iter_hist::Vector{Vector{T}}`, store the iterate sequence as the algorithm 
    progresses. The initial iterate is stored in the first position.
- `grad_val_hist::Vector{T}`, stores the norm gradient values at each iterate. 
    The norm of the gradient evaluated at the initial iterate is stored in the 
    first position.
- `stop_iteration::Int64`, the iteration number the algorithm stopped on. The 
    iterate that induced stopping is saved at `iter_hist[stop_iteration + 1]`.

# Constructors

    WatchdogFixedDampedBFGSGD(::Type{T}; c::T, α::T, δ::T, ρ::T,
        line_search_max_iterations::Int64, η::T, 
        inner_loop_max_iterations::Int64, window_size::Int64,
        threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `c::T`, initial factor used in the approximation of the Hessian.
- `α::T`, fixed step size used in the inner loop.
- `δ::T`, step size reduction parameter used in the line search routine. 
- `ρ::T`, parameter used in backtracking and the watchdog condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `line_search_max_iterations::Int64`, maximum number of line search
    iterations
- `η::T`, term used in the stopping conditions for the inner loop.
- `inner_loop_max_iterations::Int64`, maximum number of iterations in the
    inner loop.
- `window_size::Int64`, size of the objective cache.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    is at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.

"""
mutable struct WatchdogFixedDampedBFGSGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    B_θk::Matrix{T}
    norm_∇F_ψ::T
    # Quantities involved in damped BFGS update
    c::T
    Bjk::Matrix{T}
    δBjk::Matrix{T}
    rjk::Vector{T}
    sjk::Vector{T}
    yjk::Vector{T}
    d0k::Vector{T}
    djk::Vector{T}
    # line search parameters
    α::T
    δ::T
    ρ::T
    line_search_max_iterations::Int64
    max_distance_squared::T
    # watchdog stopping parameters
    η::T
    inner_loop_max_iterations::Int64
    # nonmonotone line search reference value
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    # default parameters
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function WatchdogFixedDampedBFGSGD(
    ::Type{T};
    c::T,
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iterations::Int64,
    η::T,
    inner_loop_max_iterations::Int64,
    window_size::Int64,
    threshold::T,
    max_iterations::Int64
) where {T}

    name::String = "Gradient Descent with Fixed Step Size and Damped BFGS Steps"*
        " Globalized by Watchdog."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # initialize objective cache
    objective_hist = CircularVector(zeros(T, window_size))

    return WatchdogFixedDampedBFGSGD{T}(
        name,
        T(0),                                                   # F_θk
        zeros(T, d),                                            # ∇F_θk
        zeros(T, d, d),                                         # B_θk
        T(0),                                                   # norm_∇F_ψ
        c,                                                      
        zeros(T, d, d),                                         # Bjk
        zeros(T, d, d),                                         # δBjk
        zeros(T, d),                                            # rjk
        zeros(T, d),                                            # sjk
        zeros(T, d),                                            # yjk
        zeros(T, d),                                            # d0k
        zeros(T, d),                                            # djk
        α,                                                      
        δ,
        ρ,
        line_search_max_iterations,
        T(0),                                                   # max_distance_squared
        η,
        inner_loop_max_iterations,
        objective_hist,
        T(0),                                                   # reference_value
        -1,                                                     # reference_value_index
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    )
end

"""
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::WatchdogFixedDampedBFGSGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    max_iterations = 100
) where {T}

    # initialization for inner loop
    j::Int64 = 0
    dist::T = T(0)
    optData.max_distance_squared = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # store values for update
        optData.sjk .= -ψjk
        optData.yjk .= -store.grad

        # compute step
        optData.djk .= optData.Bjk \ store.grad

        # take step
        ψjk .-= optData.Bjk \ store.grad

        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # update approximation
        optData.sjk .+= ψjk
        optData.yjk .+= store.grad
        update_success = OptimizationMethods.update_bfgs!(
            optData.Bjk, optData.rjk, optData.δBjk,
            optData.sjk, optData.yjk; damped_update = true)

        # check other stopping condition
        if optData.norm_∇F_ψ <= optData.η * (1 + abs(optData.F_θk))
            if OptimizationMethods.obj!(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end
    end

    return j
end

"""
"""
function watchdog_fixed_damped_bfgs_gd(
    optData::WatchdogFixedDampedBFGSGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, s)
    optData.grad_val_hist[1] = norm(store.grad) 

    # Initialize hessian approximation
    fill!(optData.Bjk, 0)
    OptimizationMethods.add_identity(optData.Bjk,
        optData.c * optData.grad_val_hist[iter + 1])

    # Initialize the objective history
    M = length(optData.objective_hist)
    optData.objective_hist[1] = F(optData.iter_hist[1]) 
    optData.reference_value, optData.reference_value_index = 
        optData.objective_hist[1], 1

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration counter
        iter += 1

        # inner loop
        optData.F_θk = optData.objective_hist[iter]
        optData.∇F_θk .= store.grad
        optData.B_θk .= optData.Bjk
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            max_iterations = optData.inner_loop_max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.ρ * optData.max_distance_squared

            # revert to previous iterate and approximation (for update)
            optData.Bjk .= optData.B_θk
            x .= optData.iter_hist[iter]

            # update iter_diff and grad_diff
            optData.sjk .= -x
            optData.yjk .= -optData.∇F_θk

            # backtrack on the previous iterate
            backtrack_success = OptimizationMethods.backtracking!(
                x,
                optData.iter_hist[iter],
                F,
                optData.∇F_θk,
                optData.d0k,
                optData.reference_value,
                optData.α,
                optData.δ,
                optData.ρ;
                max_iteration = optData.line_search_max_iterations)

            # if backtrack not successful terminate algorithm
            if !backtrack_success
                optData.stop_iteration = (iter - 1)
                return optData.iter_hist[iter]
            end

            # update iter_diff and grad_diff
            OptimizationMethods.grad!(progData, precomp, store, x)
            optData.grad_val_hist[iter + 1] = norm(store.grad)

            # update BFGS approximation
            optData.sjk .+= x
            optData.yjk .+= store.grad
            update_success = OptimizationMethods.update_bfgs!(
                optData.Bjk, optData.rjk, optData.δBjk,
                optData.sjk, optData.yjk; damped_update = true)

            Fx = F(x)
        else
            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
        end

        # update the objective_hist
        optData.objective_hist[iter + 1] = Fx
        if (iter % M) + 1 == optData.reference_value_index
            optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
        end

        # update iter and grad value history
        optData.iter_hist[iter + 1] .= x
    end

    optData.stop_iteration = iter

    return x
end