# Date: 2025/05/02
# Author: Christian Varner
# Purpose: Implementation of the watchdog technique
# with (non-)monotone line search fall back

"""
    WatchdogFixedGD{T} <: AbstractOptimizerData{T}

A structure for storing data about gradient descent with fixed step size,
    globalized through a watchdog technique. The structure also stores values 
    during the progression of its application on an optimization problem.

# Fields

- `name::String`, name of the optimizer for reference.
- `F_θk::T`, objective function value at the beginning of the inner loop
    for one of the inner loop stopping conditions.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `α::T`, step size used in the inner loop. Also used to initalize the
    line search when the watchdog condition fails.
- `δ::T`, the step size reduction factor used in line search.
- `ρ::T`, parameter used in backtracking and the watchdog condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `line_search_max_iterations::Int64`, maximum iteration limit for the
    backtracking routine
- `max_distance_squared::T`, term used in the watchdog condition. Corresponds
    to the maximum distance between the inner loop iterates and the starting
    inner loop iterate.
- `η::T`, term used in the stopping conditions for the inner loop.
- `inner_loop_max_iterations::Int64`, maximum iteration limit
    of the inner loop subroutine.
- `objective_hist::CircularVector{T, Vector{T}}`, objective cache used
    in the non-monotone line search conditions
- `reference_value::T`, maximum value of the objective history.
- `reference_value_index::Int64`, index corresponds to the maximum value of the
    objective history
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

    WatchdogFixedGD(::Type{T}; x0::Vector{T}, α::T, δ::T, ρ::T, window_size::Int64,
        η::T, line_search_max_iterations::Int64, inner_loop_max_iterations::Int64,
        threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `α::T`, step size used in the inner loop.
- `δ::T`, the step size reduction factor used in line search.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `window_size::Int64`, number of objective function considered in the
    computation of the reference value (i.e., size of `objective_hist`).
- `η::T`, term used in the stopping conditions for the inner loop.
- `line_search_max_iterations::Int64`, maximum iteration limit for the
    backtracking routine
- `inner_loop_max_iterations::Int64`, maximum iteration limit
    of the inner loop subroutine.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    is at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
"""
mutable struct WatchdogFixedGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    # line search helpers
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
function WatchdogFixedGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iterations::Int64,
    window_size::Int64,
    η::T,
    inner_loop_max_iterations::Int64,
    threshold::T,
    max_iterations::Int64
) where {T}

    # name for recording purposes
    name::String = "Gradient Descent with Fixed Step Size, Globalized by"*
    " Watchdog"

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return WatchdogFixedGD{T}(name,
        T(0),                                              # F_θk
        zeros(T, d),                                       # ∇F_θk
        T(0),                                              # norm_∇F_ψ
        α, 
        δ,                                                 
        ρ,
        line_search_max_iterations,
        T(0),                                              # max_distance_squared
        η,
        inner_loop_max_iterations,
        CircularVector(zeros(T, window_size)),             # objective_hist
        T(0),                                              # reference_value
        -1,                                                # reference_value_index
        threshold,                                          
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    ) 
end

"""
    inner_loop!(ψjk::S, θk::S, optData::WatchdogFixedGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64) where {T, S}

Inner loop iteration, modifying `ψjk`, `optData`, and `store` in place.
`ψjk` get updated to be the terminal iterate of the inner loop;
the fields `norm_∇F_ψ` and `max_distance_squared`
are updated in `optData`; the fields `grad` and potentially 
`obj` in `store` gets updated to be the gradient and objective at an inner
loop iterate.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\alpha \\in \\mathbb{R},~\\alpha > 0``
be stored in `optData.α`.

Let ``\\psi_0^k = \\theta_k``, then this method computes
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\alpha \\dot F(\\psi_i^k),
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``j_k == `` `optData.inner_loop_max_iterations`
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\leq \\eta (1 + |F(\\theta_k)|)`` and
    ``|F(\\psi_{j_k}^k| \\leq `` `optData.reference_value`.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::WatchdogFixedGD{T}`, `struct` that specifies the optimization
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
    optData::WatchdogFixedGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    max_iterations = 100) where {T, S}

    # initialization for inner loop
    j::Int64 = 0
    dist::T = T(0)
    optData.max_distance_squared = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # take step
        ψjk .-= optData.α .* store.grad

        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # other stopping condition
        if optData.norm_∇F_ψ <= optData.η * (1 + abs(optData.F_θk))
            if OptimizationMethods.obj(progData, precomp, store, ψjk) <= 
                    optData.reference_value
                return j
            end
        end

    end

    return j
end

"""
    watchdog_fixed_gd(optData::WatchdogFixedGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with fixed step sizes, negative
    gradient directions, which is globalized through a 
    watchdog technique. The optimization algorithm is specified
    through `optData`, and applied to the problem `progData`.

# Reference(s)

[Grippo L. and Sciandrone M. "Nonmonotone Globalization Techniques
    for the Barzilai-Borwein Gradient Method". 
    Computational Optimization and Applications.](@cite grippo2002Nonmonotone)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\alpha \\in \\mathbb{R},~\\alpha > 0``
be stored in `optData.α`, let ``\\rho=`` `optData.ρ`, and ``\\delta=``
`optData.δ`.

Let ``\\psi_0^k = \\theta_k``, and recursively define for ``j \\in \\mathbb{N}``
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\alpha \\dot F(\\psi_i^k).
```

Let ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

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

!!! note
    There is a maximum iteration limit on the line search routine, and if
    this is reached without the descent condition being satisfied, then
    the algorithm terminates.

# Arguments

- `optData::WatchdogFixedGD{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.

# Return

- `x::S`, final iterate of the optimization algorithm.
"""
function watchdog_fixed_gd(
    optData::WatchdogFixedGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.grad_val_hist[1] = norm(store.grad)

    # Initialize the objective history
    M = length(optData.objective_hist)
    optData.objective_hist[1] = F(x) 
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

            # backtrack on the previous iterate
            x .= optData.iter_hist[iter]
            backtrack_success = OptimizationMethods.backtracking!(
                x,
                optData.iter_hist[iter],
                F,
                optData.∇F_θk,
                optData.grad_val_hist[iter] ^ 2,
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

            # get new objective for history
            Fx = F(x)

            # update history of gradient
            OptimizationMethods.grad!(progData, precomp, store, x)
            optData.grad_val_hist[iter + 1] = norm(store.grad)
        else
            # update history of gradient as the iterate was accepted
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