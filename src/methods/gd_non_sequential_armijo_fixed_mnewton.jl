# Date: 2025/04/10
# Author: Christian Varner
# Purpose: Implementation of non-sequential armijo line search
# with modified newton steps and fixed step size

"""
    NonsequentialArmijoFixedMNewtonGD{T} <: AbstractOptimizerData{T}

A mutable struct that represents gradient descent with non-sequential Armijo
    line search and triggering events. The inner loop uses fixed step sizes
    with modified newton steps. This `struct` stores the specification of
    the method, and important values throughout iteration.

# Fields

- `name::String`, name of the optimizer for reference.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate. This is saved to avoid recomputation if there is a rejection.
- `∇∇F_θk::Matrix{T}`, buffer matrix for the Hessian of the initial inner loop
    iterate. This is saved to avoid recomputation if there is a rejection.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `α::T`, step size used in the inner loop. 
- `δk::T`, scaling factor used to condition the step size.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `β::T`, argument for the function used to modify the Hessian.
- `λ::T`, argument for the function used to modify the Hessian.
- `hessian_modification_max_iteration::Int64`, max number of attempts
    at modifying the Hessian per-step.
- `objective_hist::CircularVector{T, Vector{T}}`, circular vector of 
    previous accepted objective value for non-monotone cache update.
- `reference_value::T`, the maximum objective value in `objective_hist`.
- `reference_value_index::Int64`, the index of the maximum value in `objective_hist`.
- `acceptance_cnt::Int64`, tracks the number of accepted iterates. 
    This is used to update the non-monotone cache.
- `τ_lower::T`, lower bound on the gradient interval triggering event.
- `τ_upper::T`, upper bound on the gradient interval triggering event.
- `inner_loop_radius::T`, inner loop radius for the bounding ball event.
- `inner_loop_max_iterations::Int64`, inner loop max number of iterations.
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

    NonsequentialArmijoFixedMNewtonGD(::Type{T}; x0::Vector{T}, α::T,
        δ0::T, δ_upper::T, ρ::T, β::T, λ::T,
        hessian_modification_max_iteration::Int64, M::Int64,
        inner_loop_radius::T, inner_loop_max_iterations::Int64, threshold::T,
        max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, type for data and computation.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `α::T`, step size used in the inner loop. 
- `δ0::T`, initial scaling factor.
- `δ_upper::T`, upper bound on scaling factor 
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `β::T`, argument for the function used to modify the Hessian.
- `λ::T`, argument for the function used to modify the Hessian.
- `hessian_modification_max_iteration::Int64`, max number of attempts
    at modifying the Hessian per-step.
- `M::Int64`, number of objective function values from accepted iterates utilized
    in the non-monotone cache.
- `inner_loop_radius::T`, inner loop radius for the bounding ball event.
- `inner_loop_max_iterations::Int64`, inner loop max number of iterations.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.

"""
mutable struct NonsequentialArmijoFixedMNewtonGD{T} <: AbstractOptimizerData{T}
    name::String
    ∇F_θk::Vector{T}
    ∇∇F_θk::Matrix{T}
    norm_∇F_ψ::T
    α::T
    δk::T
    δ_upper::T
    ρ::T
    β::T
    λ::T
    hessian_modification_max_iteration::Int64
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    acceptance_cnt::Int64
    τ_lower::T
    τ_upper::T
    inner_loop_radius::T
    inner_loop_max_iterations::Int64
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonsequentialArmijoFixedMNewtonGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ0::T,
    δ_upper::T,
    ρ::T,
    β::T,
    λ::T,
    hessian_modification_max_iteration::Int64,
    M::Int64,
    inner_loop_radius::T,
    inner_loop_max_iterations::Int64,
    threshold::T,
    max_iterations::Int64
) where {T}

    # error checking
    @assert α > 0 "Step size $(α) needs to be positive."

    @assert δ0 > 0 "Step size $(δ0) needs to be positive."
    
    @assert δ_upper >= δ0 "The upper bound $(δ_upper) is smaller than $(δ0)."

    @assert ρ > 0 "Line search parameter ρ is not positive."

    @assert β > 0 "Modified Newton Parameter β is not positive."

    @assert λ >= 0 "Modified Newton Parameter λ is not non-negative."
    
    @assert M > 0 "Objective history length $(M) is zero or negative."

    @assert inner_loop_radius > 0 "Inner loop radius is zero or negative."

    @assert inner_loop_max_iterations > 0 "Inner loop max iteration is zero"*
        "or negative."

    @assert max_iterations >= 0 "Max iterations is negative."

    # initializations for struct
    name::String = "Gradient Descent with Triggering Events and Nonsequential"*
        "Armijo with fixed step size and Modified Newton Directions."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return NonsequentialArmijoFixedMNewtonGD{T}(
        name,
        zeros(T, d),
        zeros(T, d, d),
        T(0),
        α,
        δ0,
        δ_upper,
        ρ,
        β,
        λ,
        hessian_modification_max_iteration,
        CircularVector(zeros(T, M)),
        T(-1),
        -1,
        0,
        T(-1),
        T(-1),
        inner_loop_radius,
        inner_loop_max_iterations,
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    )
end

"""
    inner_loop!(ψjk::S, θk::S, optData::NonsequentialArmijoFixedMNewtonGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64; 
        radius = 10,
        max_iteration = 100) where {T, S}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and
`store` in place. `ψjk` gets updated to be the terminal iterate of the inner loop.
The inner loop uses a constant step size and modified newton steps

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. Let ``\\alpha`` be 
a user defined step size that remains fixed.

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\delta_k 
        \\alpha \\left(H_{i}^k\\right)^{-1}\\dot F(\\psi_i^k),
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``||\\psi_{j_k}^k - \\theta_k||_2 > 10``, 
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, 
3. ``j_k == 100``.

And, `H_{i}^k` is the modified Hessian of `F(\\psi_i^k)` if the subroutine is
successful (i.e., a constant was found such that 
``\\ddot F(\\psi_i^k) + \\lambda I``). If the subroutine was unsuccessful, then
`H_{i}^k` is set to the identity matrix (i.e., a negative gradient step is taken).

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::NonsequentialArmijoFixedMNewtonGD{T}`, `struct` that specifies the optimization
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
    optData::NonsequentialArmijoFixedMNewtonGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    k::Int64; 
    radius = 10,
    max_iteration = 100) where {T, S}

    # inner loop
    j::Int64 = 0
    optData.norm_∇F_ψ = optData.grad_val_hist[k]
    while ((norm(ψjk - θk) <= radius) &&
        (optData.τ_lower < optData.norm_∇F_ψ && optData.norm_∇F_ψ < optData.τ_upper) &&
        (j < max_iteration))

        # Increment the inner loop counter
        j += 1

        # modify Hessian and return the result
        res = add_identity_until_pd!(store.hess;
            λ = optData.λ,
            β = optData.β, 
            max_iterations = optData.hessian_modification_max_iteration)
        
        # take a gradient step if this was not successful
        if !res[2]
            ψjk .-= (optData.δk * optData.α) .* store.grad
        else
            optData.λ = res[1] / 2
            lower_triangle_solve!(store.grad, store.hess')
            upper_triangle_solve!(store.grad, store.hess)
            ψjk .-= (optData.δk * optData.α) .* store.grad
        end

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        OptimizationMethods.hess!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)
    end

    return j
end

"""
    nonsequential_armijo_mnewton_fixed_gd(
        optData::NonsequentialArmijoFixedMNewtonGD,
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with non-sequential armijo and triggering
    events. The inner loop carries out a fixed step size, modified newton step.
    The optimization algorithm is specified through `optData`, and applied to the
    problem `progData`.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. Let ``\\alpha`` be
a user selected constant step size.
The ``k+1^{th}`` iterate and parameters are produced by the following procedure. 

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\delta_k 
        \\alpha \\left(H_{i}^k\\right)^{-1}\\dot F(\\psi_i^k),
```
where `H_{i}^k` is the modified Hessian of `F(\\psi_i^k)` if the subroutine is
successful (i.e., a constant was found such that 
``\\ddot F(\\psi_i^k) + \\lambda I``). If the subroutine was unsuccessful, then
`H_{i}^k` is set to the identity matrix (i.e., a negative gradient step is taken).

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

- `optData::NonsequentialArmijoFixedMNewtonGD{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

# Return

- `x::S`, final iterate of the optimization algorithm.
"""
function nonsequential_armijo_mnewton_fixed_gd(
    optData::NonsequentialArmijoFixedMNewtonGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data and save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, x)
    OptimizationMethods.hess!(progData, precomp, store, x)
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
        optData.∇∇F_θk .= store.hess
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; 
            radius = optData.inner_loop_radius,
            max_iteration = optData.inner_loop_max_iterations)

        # check non-sequential armijo condition
        Fx = F(x)
        achieved_descent = 
        OptimizationMethods.non_sequential_armijo_condition(Fx, optData.reference_value, 
            optData.grad_val_hist[iter], optData.ρ, optData.δk, optData.α)
        
        # update the algorithm parameters and current iterate
        update_algorithm_parameters!(x, optData, achieved_descent, iter)
        
        # update histories
        optData.iter_hist[iter + 1] .= x
        if achieved_descent
            # accepted case
            optData.acceptance_cnt += 1
            optData.objective_hist[optData.acceptance_cnt] = Fx
            if ((optData.acceptance_cnt - 1) % M) + 1 == optData.reference_value_index
                optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
            end

            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
        else
            # rejected case
            store.grad .= optData.∇F_θk
            store.hess .= optData.∇∇F_θk
            optData.grad_val_hist[iter + 1] = optData.grad_val_hist[iter]
        end
    end

    optData.stop_iteration = iter

    return x
end