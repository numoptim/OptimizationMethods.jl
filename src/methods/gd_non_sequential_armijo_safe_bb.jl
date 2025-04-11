# Date: 2025/31/03
# Author: Christian Varner
# Purpose: Implementation of nonsequential armijo condition
# with the barzilai borwein step size

"""
    NonsequentialArmijoSafeBBGD{T} <: AbstractOptimizerData{T}

A mutable struct that represents gradient descent with non-sequential armijo
    line search and triggering events. The inner loop is composed of negative
    gradient directions, and a safe-guarded Barzilai-Borwein step size. 

# Fields

- `name::String`, name of the optimizer for reference.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `init_stepsize::T`, initial step size used to start the Barzilai-Borwein
    method.
- `bb_step_size::Function`, Barzilai-Borwein step size function. See
    [the long step size function](@ref OptimizationMethods.bb_long_step_size) and
    [the short step size function](@ref OptimizationMethods.bb_short_step_size).
- `α0k::T`, initial step size used in the inner loop. Used in the non-monotone
    non-sequential Armijo condition.
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
- `δk::T`, scaling factor used to condition the step size.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `objective_hist::Vector{T}`, vector of previous accepted objective values
    for non-monotone cache update.
- `reference_value::T`, the maximum objective value in `objective_hist`.
- `reference_value_index::T`, the index of the maximum value in `objective_hist`.
- `acceptance_cnt::Int64`, the number of accepted iterates.
- `τ_lower::T`, lower bound on the gradient interval triggering event.
- `τ_upper::T`, upper bound on the gradient interval triggering event.
- `second_acceptance_occurred::Bool`, if a second accepted iterate has occured.
    Used to correctly compute the Barzilai-Borwein step size.
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

    NonsequentialArmijoSafeBBGD(::Type{T}; x0::Vector{T}, init_stepsize::T,
        long_stepsize::Bool, α_lower::T, α_upper::T, δ0::T, δ_upper::T 
        ρ::T, M::Int64, threshold::T, max_iterations::Int64) where {T}

Constructs an instance of type `NonsequentialArmijoSafeBBGD`.

## Arguments

- `T::DataType`, type for data and computation.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `init_stepsize::T`, initial step size used to form Barzilai-Borwein
    step size.
- `long_stepsize::Bool`, if `true`, then 
    [the long step size function](@ref OptimizationMethods.bb_long_step_size) 
    is used. If `false`, then
    [the short step size function](@ref OptimizationMethods.bb_short_step_size). 
- `α_lower::T`, used to compute a safeguard on the Barzilai-Borwein step size.
- `α_default::T`, If the Barzilai-Borwein step size is smaller than `α_lower` or
    larger than `1/α_lower`, then it is set to `α_default`.
- `δ0::T`, starting scaling factor.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `M::Int64`, number of objective function values from accepted iterates utilized
    in the non-monotone cache.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
"""
mutable struct NonsequentialArmijoSafeBBGD{T} <: AbstractOptimizerData{T}
    name::String
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    init_stepsize::T
    bb_step_size::Function
    α0k::T
    α_lower::T
    α_default::T
    iter_diff_checkpoint::Vector{T}
    grad_diff_checkpoint::Vector{T}
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    δk::T
    δ_upper::T
    ρ::T
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    acceptance_cnt::Int64
    τ_lower::T
    τ_upper::T
    second_acceptance_occurred::Bool
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonsequentialArmijoSafeBBGD(::Type{T};
    x0::Vector{T},
    init_stepsize::T,
    long_stepsize::Bool,
    α_lower::T,
    α_default::T,
    δ0::T,
    δ_upper::T,
    ρ::T,
    M::Int64,
    threshold::T,
    max_iterations::Int64
    ) where {T}

    # error checking
    @assert 0 < δ0 "Initial scaling factor $(δ0) needs to be positive."

    @assert δ0 <= δ_upper "Initial scaling factor $(δ0) needs to be smaller"*
    " than its upper bound $(δ_upper)."

    @assert 0 < α_lower "Step size lower bound $(α_lower) needs to be positive."

    @assert 0 < α_default "Default step size $(α_default) needs to be positive."

    @assert (α_lower <= init_stepsize && init_stepsize <= 1/α_lower) "The initial"*
    " step size $(init_stepsize) is not in the interval [$(α_lower), $(1/α_lower)]." 

    # name for recording purposes
    name::String = "Gradient Descent with Triggering Events and Nonsequential"*
        "Armijo with Barzilai-Borwein step size and gradient directions."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # return struct
    return NonsequentialArmijoSafeBBGD{T}(
        name,
        zeros(T, d),    # ∇F_θk
        T(0),           # norm_∇F_ψ
        init_stepsize,
        long_stepsize ? bb_long_step_size : bb_short_step_size,
        T(0),           # α0k
        α_lower,
        α_default,
        zeros(T, d),    # iter_diff_checkpoint
        zeros(T, d),    # grad_diff_checkpoint
        zeros(T, d),    # iter_diff
        zeros(T, d),    # grad_diff
        δ0,
        δ_upper,
        ρ,
        CircularVector(zeros(T, M)),    # objective_hist
        T(-1),          # reference_value
        -1,             # reference_value_index
        0,
        T(-1),          # τ_lower
        T(-1),          # τ_upper
        false,          # second_acceptance_occurred
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    )
end

################################################################################
# Utility
################################################################################

"""
    inner_loop!(ψjk::S, θk::S, optData::NonsequentialArmijoSafeBBGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64; radius = 10, max_iteration = 100) where {T, S}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and
`store` in place. `ψjk` gets updated to be the terminal iterate of the inner loop.
This inner loop function uses negative gradient directions with a safe-guarded
    Barzilai-Borwein step size.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method.

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\delta_k \\alpha_j^k \\dot F(\\psi_i^k),
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``||\\psi_{j_k}^k - \\theta_k||_2 > 10``, 
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, 
3. ``j_k == 100``.

The step size ``\\alpha_j^k`` is calculated using the safeguarded Barzilai-
Borwein method. To explain the step size computation, define

```math
    L(k) = \\max\\lbrace 0 < t \\leq k : \\theta_t \\not= \\theta_{t-1} \\rbrace,
```
where ``\\max \\emptyset = 0``. Let 
``\\underline{\\alpha} \\in \\mathbb{R},~\\underline{\\alpha} \\in (0, 1)``, 
and ``\\alpha \\in \\mathbb{R}_{> 0}``.

Suppose that `long_stepsize = true` and let ``k + 1 \\in \\mathbb{N}``. We
now describe how the initial step size for each inner loop is calculated, and
then subsequent step sizes.
The initial step size, ``\\alpha_0^k``, depends on if ``L(k) = k`` and the value
of ``k``. For (case 1) ``k = 0`` or (case 2) ``k > 0`` and ``L(k) = 0``, 
then ``\\alpha_0^k`` is `optData.init_stepsize`.

For ``k > 0`` and ``0 < L(k) \\leq k``, let
```math
    \\gamma_0^k = 
        \\frac{
        ||\\psi_{j_{L(k)-1}}^{L(k)-1} - \\psi_{j_{L(k)-1} - 1}^{L(k)-1}||_2^2} 
        {(\\psi_{j_{L(k)-1}}^{L(k)-1} - \\psi_{j_{L(k)-1} - 1}^{L(k)-1})^\\intercal 
        (\\dot F(\\psi_{j_{L(k)-1}}^{L(k)-1}) - 
        \\dot F(\\psi_{j_{L(k)-1} - 1}^{L(k)-1}))},
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
- `optData::NonsequentialArmijoSafeBBGD{T}`, `struct` that specifies the optimization
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

- `radius = 10`, the radius of the bounding ball event. Should be kept at `10`,
    however we allow it to be a parameter for future purposes.
- `max_iteration = 100`, maximum number of allowable iteration of the inner loop.
    Should be kept at `100` as that is what is specified in the paper, but
    is useful to change for testing.

# Returns

- `j::Int64`, the iteration for which a triggering event evaluated to true.
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::NonsequentialArmijoSafeBBGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    radius = 10,
    max_iteration = 100) where {T, S}

    # inner loop iteration counter
    j::Int64 = 0

    # step size initialization
    step_size::T = (optData.second_acceptance_occurred) ?
        optData.bb_step_size(optData.iter_diff, optData.grad_diff) : 
        optData.init_stepsize 
    if step_size < optData.α_lower || step_size > (1/optData.α_lower)
        step_size = optData.α_default
    end
    optData.α0k = step_size
    
    # update the value of the norm gradient
    optData.norm_∇F_ψ = optData.grad_val_hist[k]
    
    # inner loop
    while ((norm(ψjk - θk) <= radius) &&
        (optData.τ_lower < optData.norm_∇F_ψ && optData.norm_∇F_ψ < optData.τ_upper) &&
        (j < max_iteration))

        # Increment the inner loop counter
        j += 1

        # needed for calculation of next step size
        optData.iter_diff .= -ψjk
        optData.grad_diff .= -store.grad

        # take step
        ψjk .-= (optData.δk * step_size) .* store.grad

        # store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)

        # calculate values needed for next iteration
        optData.iter_diff .+= ψjk
        optData.grad_diff .+= store.grad

        # compute step size for next iteration
        step_size = optData.bb_step_size(optData.iter_diff, optData.grad_diff)
        if step_size < optData.α_lower || step_size > (1/optData.α_lower)
            step_size = optData.α_default
        end
    end

    return j
end

"""
    nonsequential_armijo_safe_bb_gd(optData::NonsequentialArmijoSafeBBGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of the nonsequential armijo globalization framework with
    a safe version of the Barzilai-Borwein step size and negative
    gradient directions as specified by `optData` on the optimization problem
    specified by `progData`.

# Reference(s)

[Barzilai and Borwein. "Two-Point Step Size Gradient Methods". IMA Journal of 
    Numerical Analysis.](@cite barzilai1988Twopoint)

[Marcos Raydan. "The Barzilai and Borwein Gradient Method for the Large
    Scale Unconstrained Minimization Problem". 
    SIAM Journal of Optimization.](@cite raydan1997The)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method.
The ``k+1^{th}`` iterate and parameters are produced by the following procedure. 

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\delta_k \\alpha_j^k \\dot F(\\psi_i^k).
```
To see how the inner loop steps are performed for this method,
see documentation for [OptimizationMethods.inner_loop!](@ref) where 
`optData::NonsequentialArmijoSafeBBGD{T}`.

Let ``j_k \\in \\mathbb{N}`` be the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``||\\psi_{j_k}^k - \\theta_k||_2 > 10``, 
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, 
3. ``j_k == 100``.

The next iterate and algorithmic parameters in `optData` are updated based on 
the result of the non-sequential Armijo condition
```math
    F(\\psi_{j_k}^k) \\leq 
    F(\\theta_k) - \\rho\\delta_k\\alpha_0^k||\\dot F(\\theta_k)||_2.
```

When this condition is not satisfied, the following quantities are updated.

1. The iterate ``\\psi_{j_k}^k`` is rejected, and ``\\theta_{k+1} = \\theta_k``
2. The scaling factor ``\\delta_{k+1} = .5\\delta_k``

When this condition is satisfied, the following quantities are updated.

1. The iterate ``\\psi_{j_k}^k`` is accepted, and ``\\theta_{k+1} = \\psi_{j_k}^k``.
2. The scaling factor is updated as ``\\delta_{k+1} = \\min(1.5*\\delta_k, \\bar\\delta)``
    when ``||\\dot F(\\psi_{j_k}^k)||_2 > \\tau_{\\mathrm{grad},\\mathrm{lower}}^k``,
    otherwise ``\\delta_{k+1} = \\delta_k``.
3. If ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, then 
    ``\\tau_{\\mathrm{grad},\\mathrm{lower}}^{k+1} = 
    ||\\dot F(\\psi_{j_k}^k)||_2/\\sqrt{2}`` and 
    ``\\tau_{\\mathrm{grad},\\mathrm{upper}}^{k+1} = 
    \\sqrt{10}||\\dot F(\\psi_{j_k}^k)||_2``. Otherwise, this parameters are held
    constant.

# Arguments

- `optData::NonsequentialArmijoSafeBBGD{T}`, the specification for the optimization
    method.
- `progData <: AbstractNLPModel{T, S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.

# Return

- `x::Vector{T}`, final iterate of the optimization algorithm.
"""
function nonsequential_armijo_safe_bb_gd(
    optData::NonsequentialArmijoSafeBBGD{T},
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
    optData.acceptance_cnt += 1
    optData.objective_hist[1] = F(optData.iter_hist[1]) 
    optData.reference_value, optData.reference_value_index = 
        optData.objective_hist[1], 1

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration counter
        iter += 1

        # inner loop
        optData.∇F_θk .= store.grad
        optData.iter_diff_checkpoint .= optData.iter_diff
        optData.grad_diff_checkpoint .= optData.grad_diff
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter)

        # check non-sequential armijo condition
        Fx = F(x)
        achieved_descent = 
        OptimizationMethods.non_sequential_armijo_condition(Fx, optData.reference_value, 
            optData.grad_val_hist[iter], optData.ρ, optData.δk, optData.α0k)
        
        # update the algorithm parameters and current iterate
        update_algorithm_parameters!(x, optData, achieved_descent, iter)
        
        # update histories
        optData.iter_hist[iter + 1] .= x
        if achieved_descent

            ## update the objective cache
            optData.acceptance_cnt += 1
            optData.objective_hist[optData.acceptance_cnt] = Fx
            if ((optData.acceptance_cnt - 1) % M) + 1 == optData.reference_value_index
                optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
            end

            ## update values for the next iteration
            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
            optData.second_acceptance_occurred = true
        else
            # rejected case
            store.grad .= optData.∇F_θk
            optData.grad_val_hist[iter + 1] = optData.grad_val_hist[iter]
            optData.iter_diff .= optData.iter_diff_checkpoint
            optData.grad_diff .= optData.grad_diff_checkpoint
        end
    end

    optData.stop_iteration = iter

    return x

end