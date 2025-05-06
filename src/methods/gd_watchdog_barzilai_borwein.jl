# Date: 2025/05/05
# Author: Christian Varner
# Implementation of gradient descent with Barzilai-Borwein steps
# globalized through a watchdog technique

"""
    WatchdogBarzilaiBorweinGD{T} <: AbstractOptimizerData{T}

A structure for storing data about gradient descent with barzilai-borwein
    step size and negative gradient steps, globalized through the
    watchdog framework. The structure also stores values during the progression
    of its application on an optimization problem.

# Fields

- `name::String`, name of the optimizer for reference.
- `F_θk::T`, objective function value at the beginning of the inner loop
    for the inner loop stopping condition.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `init_stepsize::T`, initial step size used to start the Barzilai-Borwein
    method.
- `bb_step_size::Function`, Barzilai-Borwein step size function. See
    [the long step size function](@ref OptimizationMethods.bb_long_step_size) and
    [the short step size function](@ref OptimizationMethods.bb_short_step_size).
- `α_lower::T`, used to compute a safeguard on the Barzilai-Borwein step size.
- `α_default::T`, If the Barzilai-Borwein step size is smaller than `α_lower` or
    larger than `1/α_lower`, then it is set to `α_default`.
- `iter_diff_checkpoint::Vector{T}`, buffer array for difference between
    iterates before the start of an inner loop. Values are saved because of 
    potential restarts.
- `grad_diff_checkpoint::Vector{T}`, buffar array for difference between
    gradients before the start of an inner loop. Values are saved because of 
    potential restarts.
- `iter_diff::Vector{T}`, buffer array for difference between iterates 
    used to calculate the step size.
- `grad_diff::Vector{T}`, buffer array for difference between gradients
    used to calculate the step size.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `line_search_max_iterations::Int64`, maximum number of line search
    iterations
- `max_distance_squared::T`, maximum distance between the starting
    and inner loop iterates. Used in the watchdog condition.
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

    WatchdogBarzilaiBorweinGD(::Type{T}, x0::Vector{T}, init_stepsize::T,
        long_stepsize::Bool, α_lower::T, α_default::T, ρ::T,
        line_search_max_iterations::Int64, η::T,
        inner_loop_max_iterations::Int64, window_size::Int64, threshold::T,
        max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `init_stepsize::T`, initial step size used to start the Barzilai-Borwein
    method.
- `long_stepsize::Bool`, if `true` use the long form of the step size,
    otherwise use the short form.
- `α_lower::T`, used to compute a safeguard on the Barzilai-Borwein step size.
- `α_default::T`, If the Barzilai-Borwein step size is smaller than `α_lower` or
    larger than `1/α_lower`, then it is set to `α_default`.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
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
mutable struct WatchdogBarzilaiBorweinGD{T} <: AbstractOptimizerData{T}
    name::String
    F_θk::T
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    # step size helpers
    init_stepsize::T
    bb_step_size::Function
    α_lower::T
    α_default::T
    iter_diff_checkpoint::Vector{T}
    grad_diff_checkpoint::Vector{T}
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    # line search parameters
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
function WatchdogBarzilaiBorweinGD(::Type{T},
    x0::Vector{T},
    init_stepsize::T,
    long_stepsize::Bool,
    α_lower::T,
    α_default::T,
    ρ::T,
    line_search_max_iterations::Int64,
    η::T,
    inner_loop_max_iterations::Int64,
    window_size::Int64,
    threshold::T,
    max_iterations::Int64
    ) where {T}

    name::String = "Gradient Descent with Barzilai-Borwein Step Size, Globalized"*
        " by Watchdog."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # initalize step size function
    step_size = long_stepsize ? bb_long_step_size : bb_short_step_size

    return WatchdogBarzilaiBorweinGD{T}(
        name,
        T(0),
        zeros(T, d),
        T(0),
        init_stepsize,
        step_size,
        α_lower,
        α_default,
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
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

    inner_loop!(ψjk::S, θk::S, optData::WatchdogBarzilaiBorweinGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, k::Int64; 
        max_iterations = 100) where {T}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and `store` in
    place. `ψjk` gets updated to be the terminal iterate of the inner loop.
    This inner loop function uses negative gradient directions with a 
    safeguarded Barzilai-Borwein step size.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm.

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\alpha_j^k \\dot F(\\psi_i^k),
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``j_k == `` `optData.inner_loop_max_iterations`
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\leq \\eta (1 + |F(\\theta_k)|)`` and
    ``|F(\\psi_{j_k}^k| \\leq `` `optData.reference_value`.

The step size ``\\alpha_j^k`` is calculated using the safeguarded Barzilai-
Borwein method. To explain the step size computation, define

Suppose that `long_stepsize = true` and let ``k + 1 \\in \\mathbb{N}``. We
now describe how the initial step size for each inner loop is calculated, and
then subsequent step sizes.
The initial step size ``\\alpha_0^0`` is `optData.init_stepsize`.

For ``k > 0``, if the watchdog condition was satisfied at ``\\psi_{j_{k-1}}^k``,
then
```math
    \\gamma_0^k = 
        \\frac{
        ||\\psi_{j_{k-1}}^{k-1} - \\psi_{j_{k-1} - 1}^{k-1}||_2^2} 
        {(\\psi_{j_{k-1}}^{k-1} - \\psi_{j_{k-1} - 1}^{k-1})^\\intercal 
        (\\dot F(\\psi_{j_{k-1}}^{k-1}) - 
        \\dot F(\\psi_{j_{k-1} - 1}^{k-1}))},
```
otherwise, ``\\theta_{k}`` was produced by backtracking on ``\\theta_{k-1}``
and 
```math
    \\gamma_0^k = 
        \\frac{
        ||\\theta_{k} - \\theta_{k-1}||_2^2} 
        {(\\theta_{k} - \\theta_{k-1})^\\intercal 
        (\\dot F(\\theta_{k}) - \\dot F(\\theta_{k-1}))},
```
then if ``\\gamma_0^k \\in [\\underline{\\alpha}, 1/\\underline{\\alpha}]`` then
``\\alpha_0^k = \\gamma_0^k``, otherwise ``\\alpha_0^k = \\alpha``.

In all cases, for ``j \\in \\mathbb{N}`` and ``k + 1 \\in \\mathbb{N}``
```math
    \\gamma_j^k = 
        \\frac{
        ||\\psi_j^k - \\psi_{j-1}^k||_2^2}{ 
        (\\psi_j^k - \\psi_{j-1}^k)^\\intercal 
        (\\dot F(\\psi_j^k) - \\dot F(\\psi_{j-1}^k))},
```
and if ``\\gamma_j^k \\in [\\underline{\\alpha}, 1/\\underline{\\alpha}]`` then
``\\alpha_j^k = \\gamma_j^k``, otherwise ``\\alpha_0^k = \\alpha``.

When `long_stepsize = false`, the cases remain the same but the step size formula
changes to the short form of the Barzilai-Borwein step size.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::WatchdogBarzilaiBorweinGD{T}`, `struct` that specifies the optimization
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
    Should be kept at `100` as that is what is specified in the paper, but
    is useful to change for testing.

# Returns

- `j::Int64`, the iteration for which a triggering event evaluated to true.
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::WatchdogBarzilaiBorweinGD{T}, 
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

    # compute the initial step size
    step_size = k == 1 ? optData.init_stepsize : 
        optData.bb_step_size(optData.iter_diff, optData.grad_diff)
    if step_size < optData.α_lower || step_size > (1/optData.α_lower)
        step_size = optData.α_default
    end

    # stopping conditions
    while j < max_iterations

        # Increment the inner loop counter
        j += 1

        # update iter diff and grad diff
        optData.iter_diff .= -ψjk
        optData.grad_diff .= -store.grad

        # take step
        ψjk .-= step_size .* store.grad

        dist = norm(ψjk - θk)
        optData.max_distance_squared = max(dist^2, optData.max_distance_squared)

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # update values in iter_diff and grad_diff
        optData.iter_diff .+= ψjk
        optData.grad_idff .+= store.grad

        # safe guard against too large or too small step sizes
        step_size = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end

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
    watchdog_barzilai_borwein_gd(optData::WatchdogBarzilaiBorweinGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with the Barzilai-Borwein step size and
    negative gradient directions, which is globalized through the
    watchdog technique. The optimization algorithm is specified
    through `optData`, and applied to the problem `progData`.

# Reference(s)

[Grippo L. and Sciandrone M. "Nonmonotone Globalization Techniques
    for the Barzilai-Borwein Gradient Method". 
    Computational Optimization and Applications.](@cite grippo2002Nonmonotone)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm.

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\alpha_j^k \\dot F(\\psi_i^k),
```
To see how the inner loop steps are performed for this method, see
    the documentation for [OptimizationMethods.inner_loop!](@ref) for
    `optData::WatchdogBarzilaiBorweinGD{T}`.

Let ``j_k \\in \\mathbb{N}`` be the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``j_k == `` `optData.inner_loop_max_iterations`
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\leq \\eta (1 + |F(\\theta_k)|)`` and
    ``|F(\\psi_{j_k}^k| \\leq `` `optData.reference_value`.

Let 
``\\tau_{\\mathrm{obj}}^k = \\max_{0 \\leq i \\leq max(0, M - 1)} F(\\theta_{k - i})``.

If the watchdog condition

```math
    F(\\psi_{j_k}^k) \\leq \\tau_{\\mathrm{obj}}^k - 
        \\max_{0 \\leq j \\leq j_k} ||\\psi_j^k - \\theta_k||_2^2.
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

- `optData::WatchdogBarzilaiBorweinGD{T}`, the specification for the optimization 
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
function watchdog_barzilai_borwein_gd(
    optData::WatchdogBarzilaiBorweinGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, optData.iter_hist[1])
    optData.grad_val_hist[1] = norm(store.grad)

    # update constants needed for triggering events
    optData.τ_lower = optData.grad_val_hist[1] / sqrt(2)  
    optData.τ_upper = sqrt(10) * optData.grad_val_hist[1]   

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
        optData.iter_diff_checkpoint .= optData.iter_diff
        optData.grad_diff_checkpoint .= optData.grad_diff
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            max_iterations = optData.inner_loop_max_iterations)
        Fx = F(x)

        # if watchdog not successful, try to backtrack
        if Fx > optData.reference_value - optData.max_distance_squared

            # revert to previous iterate
            x .= optData.iter_hist[iter]

            # compute the initial step size
            step_size = (iter == 1) ? optData.init_stepsize : 
                optData.bb_step_size(optData.iter_diff_checkpoint, 
                    optData.grad_diff_checkpoint)
            if step_size < optData.α_lower || step_size > (1/optData.α_lower)
                step_size = optData.α_default
            end

            # update iter_diff and grad_diff
            optData.iter_diff .= -x
            optData.grad_diff .= -optData.∇F_θk

            # backtrack on the previous iterate
            backtrack_success = OptimizationMethods.backtracking!(
                x,
                optData.iter_hist[iter],
                F,
                optData.∇F_θk,
                optData.grad_val_hist[iter] ^ 2,
                optData.reference_value,
                step_size,
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

            optData.iter_diff .+= x
            optData.grad_diff .+= store.grad

            Fx = F(x)
        else
            OptimizationMethods.grad!(progData, precomp, store, x)
            optData.grad_val_hist[iter + 1] = norm(store.grad)
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