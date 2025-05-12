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
    inner_loop!(ψjk::S, θk::S, optData::WatchdogFixedMNewtonGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64; max_iterations = 100) where {T}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and `store` in
    place. `ψjk` gets updated to be the terminal iterate of the inner loop.
    This inner loop function uses fixed step sizes with modified Newton
    directions.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\alpha =`` `optData.α`.

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\alpha_i^k d_i^k,
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which 
at least one of the conditions are satisfied: 

1. ``j_k == `` `optData.inner_loop_max_iterations`
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\leq \\eta (1 + |F(\\theta_k)|)`` and
    ``|F(\\psi_{j_k}^k| \\leq `` `optData.reference_value`.

The step direction ``d_i^k`` is one of the following. Let ``\\ddot F(\\psi_i^k)``
be the hessian matrix of ``F`` at ``\\psi_i^k`` for ``i + 1 \\in \\mathbb{N}``
and ``k + 1 \\in \\mathbb{N}``. Then, if 
[OptimizationMethods.add_identity_until_pd!](@ref) successful modifies
``\\ddot F(\\psi_i^k)``, returning ``H_i^k``, then

```math
    d_i^k = (H_i^k)^{-1} \\dot F(\\psi_i^k).
```

Otherwise, if this routine is not successful ``d_i^k = \\dot F(\\psi_i^k)``.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::WatchdogFixedMNewtonGD{T}`, `struct` that specifies the optimization
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
    optData::WatchdogFixedMNewtonGD{T}, 
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
            if OptimizationMethods.obj(progData, precomp, store, ψjk) <= optData.reference_value
                return j
            end
        end

    end # end while loop

    return j

end

"""
    watchdog_fixed_mnewton_gd(optData::WatchdogFixedMNewtonGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

# Reference(s)

[Grippo L. and Sciandrone M. "Nonmonotone Globalization Techniques
    for the Barzilai-Borwein Gradient Method". 
    Computational Optimization and Applications.](@cite grippo2002Nonmonotone)

[Nocedal and Wright. "Numerical Optimization". Edition 2, Springer, 2006, 
    Page 51.](@cite nocedal2006Numerical)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\alpha =`` `optData.α`, 
``\\rho =`` `optData.ρ`, and ``\\delta=`` `optData.δ`.

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\alpha_i^k d_i^k.
```
For more information on ``d_i^k``, see the documentation for
[OptimizationMethods.inner_loop!](@ref) when
`optData::WatchdogFixedMNewtonGD{T}`.

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

- `optData::WatchdogFixedMNewtonGD{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

# Return

- `x::S`, final iterate of the optimization algorithm.
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