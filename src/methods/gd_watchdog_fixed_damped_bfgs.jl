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

    WatchdogFixedDampedBFGSGD(::Type{T}; x0::Vector{T}, c::T, α::T, δ::T, ρ::T,
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
    x0::Vector{T},
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
    inner_loop!(ψjk::S, θk::S, optData::WatchdogFixedDampedBFGSGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64; max_iterations = 100) where {T}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and `store` in
    place. `ψjk` gets updated to be the terminal iterate of the inner loop.
    This inner loop function uses fixed step sizes with the damped BFGS step.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\alpha =`` `optData.α`.

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\alpha d_i^k,
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which 
at least one of the conditions are satisfied: 

1. ``j_k == `` `optData.inner_loop_max_iterations`
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\leq \\eta (1 + |F(\\theta_k)|)`` and
    ``|F(\\psi_{j_k}^k| \\leq `` `optData.reference_value`.

The step direction ``d_i^k`` is the damped BFGS step. In particular, let
``B_i^k`` be the damped BFGS approximation to the Hessian using
[OptimizationMethods.update_bfgs!](@ref). Then,

```math
    d_i^k = (B_i^k)^{-1} \\dot F(\\psi_i^k).
```

If the inner loop iterate is accepted (the watchdog condition is satisfied) at
time ``k \\in \\mathbb{N}``, then ``B_0^k = B_{j_{k-1}}^{k-1}``; otherwise,
the ``B_0^k`` is ``B_0^{k-1}``  updated using the damped
approximation between ``\\theta_{k-1}`` and ``\\theta_k`` where ``\\theta_k``
was produced through backtracking.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::WatchdogFixedDampedBFGSGD{T}`, `struct` that specifies the optimization
    algorithm. Fields are modified during the inner loop.
- `progData::P1 where P1 <: AbstractNLPModel{T, S}`, `struct` that specifies the
    optimization problem. Fields are modified during the inner loop.
- `precomp::P2 where P2 <: AbstractPrecompute{T}`, `struct` that has precomputed
    values. Required to take advantage of this during the gradient computation.
- `store::P3 where P3 <: AbstractProblemAllocate{T}`, `struct` that contains
    buffer arrays for computation.
- `k::Int64`, outer loop iteration for computation of the local Lipschitz
    approximation scheme.

## Optional Keyword Arguments

- `max_iteration = 100`, maximum number of allowable iteration of the inner loop.

# Returns

- `j::Int64`, the iteration for which a triggering event evaluated to true.
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
) where {T, S}

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
        if j == 1
            optData.d0k .= optData.djk
        end

        # take step
        ψjk .-= optData.α * optData.djk

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
            if OptimizationMethods.obj(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end
    end

    return j
end

"""
    watchdog_fixed_damped_bfgs_gd(optData::WatchdogFixedDampedBFGSGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of 

# Reference(s)

[Grippo L. and Sciandrone M. "Nonmonotone Globalization Techniques
    for the Barzilai-Borwein Gradient Method". 
    Computational Optimization and Applications.](@cite grippo2002Nonmonotone)

[Nocedal and Wright, "Numerical Optimization". Springer. 2nd Edition. 
    Chapter 6 and 18.](@cite nocedal2006Numerical)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\alpha =`` `optData.α`.

Let ``\\psi_0^k = \\theta_k``, then recursively define for ``j \\in \\mathbb{N}``
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\alpha d_i^k.
```
For more information on ``d_i^k``, see the documentation for 
[OptimizationMethods.inner_loop!](@ref) when 
`optData::WatchdogFixedDampedBFGSGD{T}`.

Let ``j_k \\in \\mathbb{N}`` is the smallest iteration for which 
at least one of the conditions are satisfied: 

1. ``j_k == `` `optData.inner_loop_max_iterations`
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\leq \\eta (1 + |F(\\theta_k)|)`` and
    ``|F(\\psi_{j_k}^k| \\leq `` `optData.reference_value`.

Let 
``\\tau_{\\mathrm{obj}}^k = \\max_{0 \\leq i \\leq max(0, M - 1)} F(\\theta_{k - i})``.

If the watchdog condition

```math
    F(\\psi_{j_k}^k) \\leq \\tau_{\\mathrm{obj}}^k - 
        \\rho \\max_{0 \\leq j \\leq j_k} ||\\psi_j^k - \\theta_k||_2^2.
```

then ``\\theta_{k+1} = \\psi_{j_k}^k``; otherwise, we find a 
``t + 1 \\in \\mathbb{N}`` such that the following condition is satisfied

```math
    F(\\theta_{k} - \\delta^t \\alpha \\dot F(\\theta_k)) \\leq
    \\tau_{\\mathrm{obj}}^k - \\rho\\delta^t\\alpha||\\dot F(\\theta_k)||_2^2,
```

and set ``\\theta_{k} = \\theta_{k} - \\delta^t \\alpha \\dot F(\\theta_k)``.
That is, the algorithm tries to find the next iterate through a backtracking
routine. 

# Arguments

- `optData::WatchdogFixedDampedBFGSGD{T}`, the specification 
    for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.

# Return

- `x::S`, final iterate of the optimization algorithm.
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