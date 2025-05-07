# Date: 2025/05/06
# Author: Christian Varner
# Purpose: Implementation of the fixed step size modified newton
# method globalized through the watchdog technique

"""
    WatchdogFixedMNewtonGD{T} <: AbstractOptimizerData{T}

Mutable structure that parameterizes gradient descent with fixed step sizes
    with modified newton directions. The structure also stores and tracks
    values during the progress of applying the method to an optimization
    problem.

# Fields
    
- `name::String`, name of the optimizer for reference.
- `F_θk::T`, objective function value at the beginning of the inner loop
    for one of the inner loop stopping condition.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `β::T`, argument for the function used to modify the hessian.
- `λ::T`, argument for the function used to modify the hessian.
- `hessian_modification_max_iteration::Int64`, max number of attempts
    at modifying the hessian per-step.
- `d0k::Vector{T}`, buffer array for the first step of the inner loop.
    Saved in case the inner loop fails and backtracking using this step
    is required.
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

    WatchdogFixedMNewtonGD(::Type{T}; x0::Vector{T}, β::T, λ::T,
        hessian_modification_max_iteration::Int64, α::T, δ::T, ρ::T,
        line_search_max_iterations::Int64, η::T,
        inner_loop_max_iterations::Int64, window_size::Int64,
        threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, type for data and computation.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `β::T`, argument for the function used to modify the hessian.
- `λ::T`, argument for the function used to modify the hessian.
- `hessian_modification_max_iteration::Int64`, max number of attempts
    at modifying the hessian per-step.
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
mutable struct WatchdogFixedMNewtonGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    # modified newton helpers
    β::T
    λ::T
    hessian_modification_max_iteration::Int64
    d0k::Vector{T}
    # line search parameters
    α::T
    δ::T
    ρ::T
    line_search_max_iterations::Int64
    max_distance_squared::T
    # watchdog stopping parameters
    η::T
    inner_loop_max_iterations::Int64
    # nonmonotone line search reference values
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
function WatchdogFixedMNewtonGD(::Type{T};
    x0::Vector{T},
    β::T,
    λ::T,
    hessian_modification_max_iteration::Int64,
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iterations::Int64,
    η,
    inner_loop_max_iterations::Int64,
    window_size::Int64,
    threshold::T,
    max_iterations::Int64
) where {T}
    
    name::String = "Gradient Descent with fixed step size and modified"*
    " Newton directions, globalized through Watchdog."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return WatchdogFixedMNewtonGD{T}(
        name,
        T(0),
        zeros(T, d),
        T(0),
        β,
        λ,
        hessian_modification_max_iteration,
        zeros(T, d),
        α,
        δ,
        ρ,
        line_search_max_iterations,
        T(0),
        η,
        inner_loop_max_iterations,
        CircularVector(zeros(T, window_size)),
        T(0),
        -1,
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
    optData::WatchdogFixedMNewtonGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    max_iterations = 100) where {T}

    # initialization for inner loop
    j::Int64 = 0
    dist::T = T(0)
    optData.max_distance_squared = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # hessian modification
        res = add_identity_until_pd!(store.hess;
            λ = optData.λ,
            β = optData.β,
            max_iterations = optData.hessian_modification_max_iteration)

        # if modification failed take negative gradient step; otherwise
        # take a modified newton step
        if !res[2]
            ψjk .-= optData.α .* store.grad
        else
            optData.λ = res[1] / 2
            lower_triangle_solve!(store.grad, store.hess')
            upper_triangle_solve!(store.grad, store.hess)
            ψjk .-= optData.α .* store.grad
        end

        # save the first step in case we need to backtrack
        if j == 1
            optData.d0k .= store.grad
        end

        # update the maximum distance
        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        OptimizationMethods.hess!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # check other stopping condition
        if optData.norm_∇F_ψ <= optData.η * (1 + abs(optData.F_θk))
            if OptimizationMethods.obj!(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end

    end # end while loop

    return j

end

"""
"""
function watchdog_fixed_mnewton_gd(
    optData::WatchdogFixedMNewtonGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.hess!(progData, precomp, store, x)
    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.grad_val_hist[1] = norm(store.grad)

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
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            max_iterations = optData.inner_loop_max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.ρ * optData.max_distance_squared

            # revert to previous iterate
            x .= optData.iter_hist[iter]

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

            Fx = F(x)
        end

        # update the objective_hist
        optData.objective_hist[iter + 1] = Fx
        if (iter % M) + 1 == optData.reference_value_index
            optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
        end

        # update iter and grad value history
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        OptimizationMethods.hess!(progData, precomp, store, ψjk)
        optData.grad_val_hist = norm(store.grad)
        optData.iter_hist[iter + 1] .= x
    end

    optData.stop_iteration = iter

    return x
end