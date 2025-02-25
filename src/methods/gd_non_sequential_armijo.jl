# Date: 02/11/2025
# Author: Christian Varner
# Purpose: Implementation of gradient descent with novel gradient descent
# non-sequential Armijo condition

################################################################################
# Implementation Notes (for a v2 of this method):
#   
#   1) Ideally, the inner loop code is replaced with a function that is a field
#   in the struct. This function should return a iterate, and it should
#   carry out any of the gradient methods we have implemented so far.
#
#   2) The method should potentially seperate out the non-sequential armijo
#   condition into a function to follow the other implementations of line search
################################################################################ 

"""
    NonsequentialArmijoGD{T} <: AbstractOptimizerData{T}

A mutable struct that represents gradient descent with non-sequential armijo
    line search and triggering events. It stores the specification for the
    method and records values during iteration.

# Fields

- `name::String`, name of the optimizer for reference.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `prev_∇F_ψ::Vector{T}`, buffer array for the previous gradient in the inner 
    loop. Necessary for updating the local Lipschitz approximation.
- `prev_norm_step::T`, norm of the step between inner loop iterates. 
    Used for updating the local Lipschitz approximation.
- `α0k::T`, first step size used in the inner loop. 
- `δk::T`, scaling factor used to condition the step size.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `τ_lower::T`, lower bound on the gradient interval.
- `τ_upper::T`, upper bound on the gradient interval.
- `local_lipschitz_estimate::T`, local Lipshitz approximation.
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

    NonsequentialArmijoGD(::Type{T}; x0::Vector{T}, δ0::T, δ_upper::T, ρ::T,
        threshold::T, max_iterations::Int64) where {T}

Constructs an instance of type `NonsequentialArmijoGD{T}`.

## Arguments

- `T::DataType`, type for data and computation.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `δ0::T`, starting scaling factor.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.

"""
mutable struct NonsequentialArmijoGD{T} <: AbstractOptimizerData{T}
    name::String
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    prev_∇F_ψ::Vector{T}
    prev_norm_step::T
    α0k::T
    δk::T
    δ_upper::T
    ρ::T
    τ_lower::T
    τ_upper::T
    local_lipschitz_estimate::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonsequentialArmijoGD(
    ::Type{T};
    x0::Vector{T},
    δ0::T,
    δ_upper::T,
    ρ::T,
    threshold::T,
    max_iterations::Int64
) where {T}

    # error checking
    @assert 0 < δ0 "Initial scaling factor $(δ0) needs to be positive."

    @assert δ0 <= δ_upper "Initial scaling factor $(δ0) needs to be smaller"*
    " than its upper bound $(δ_upper)."

    # name for recording purposes
    name::String = "Gradient Descent with Triggering Events and Nonsequential Armijo"

    d = length(x0)

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return NonsequentialArmijoGD(
        name,
        zeros(T, d),
        T(0),
        zeros(T, d),
        T(0),
        T(0),
        δ0,
        δ_upper,
        ρ,
        T(-1),
        T(-1),
        T(1.0),
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
    update_local_lipschitz_approximation(j::Int64, k::Int64, djk::S, curr_grad::S,
        prev_grad::S, prev_approximation::T, prev_acceptance::Bool) where {T, S}

Given the previous approximation of the local Lipschitz constant,
    `prev_approximation::T`, update the current estimate. 
    That is, return the Lipschitz approximation for inner loop iteration `j` 
    and outer loop iteration `k`.

# Method

The local Lipschitz approximation method is conducted as follows. Let ``j``
    be the current inner loop iteration counter, and let ``k`` be the current
    outer loop iteration counter. Let ``\\psi_j^k`` be the ``j^{th}`` iterate
    of the ``k^{th}`` outer loop, and let ``\\hat L_{j}^k`` be the ``j^th`` estimate
    of the ``k^{th}`` outer loop of the local Lipschitz constant. The local
    estimate is updated according the following five cases.

1. When ``j == 1`` and ``k == 1``, this is the first iteration of the first 
    inner loop, and as there is no information available we set it to `1.0`.
2. When ``j == 1`` and ``k > 1``, this is the first iteration of the ``k^{th}``
    inner loop, and we return ``L_{j_{k-1}}^{k-1}`` which is the local Lipschitz
    estimates formed using information at the terminal iteration of the ``k-1^{th}``
    inner loop (i.e., this is the latest estimate).
3. When ``j > 1`` and ``k == 1``, this is an inner loop iteration where we have
    possible taken multiple steps, so we return the most 'local' estimate
    of the local Lipschitz constant which is
    ``\\frac{
        ||\\dot F(\\psi_{j}^k) - \\dot F(\\psi_{j-1}^k)||_2}{ 
        ||\\psi_{j}^k - \\psi_{j-1}^k||_2}.
    ``

4. When ``j > 1`` and ``k > 1`` and ``\\psi_{j_{k-1}}^{k-1}`` satisfied the
    descent condition, then we return 
    ``\\frac{
    ||\\dot F(\\psi_{j}^k) - \\dot F(\\psi_{j-1}^k)||_2}{
        ||\\psi_{j}^k - \\psi_{j-1}^k||_2}.
    ``

5. When ``j > 1`` and ``k > 1`` and ``\\psi_{j_{k-1}}^{k-1}`` did not satisfy the
    descent condition, then we return 
    ``\\max
    \\left( \\frac{||\\dot F(\\psi_{j}^k) - \\dot F(\\psi_{j-1}^k)||_2}{
        ||\\psi_{j}^k - \\psi_{j-1}^k||_2}, \\hat L_{j-1}^k \\right).
    ``

# Arguments

- `j::Int64`, inner loop iteration.
- `k::Int6`, outer loop iteration.
- `norm_djk::T`, norm of difference between `\\psi_j^k` and ``\\psi_{j-1}^k``.
    On the first iteration of the inner loop this will not matter.
- `curr_grad::S`, gradient at ``\\psi_j^k`` (i.e., the current iterate).
- `prev_grad::S`, gradient at ``\\psi_{j-1}^k`` (i.e., the previous iterate).
- `prev_approximation::T`, the local Lipschitz approximation from the previous 
    iteration
- `prev_acceptance::Bool`, whether or not the previous inner loop resulted in
    an accepted iterate.

# Return

`estimate::T`, estimate of the local Lipschitz constant.
"""
function update_local_lipschitz_approximation(j::Int64, k::Int64,
    norm_djk::T, curr_grad::S, prev_grad::S, prev_approximation::T, 
    prev_acceptance::Bool) where {T, S}

    # compute and return local estimate
    if j == 1 && k == 1
        return T(1)
    elseif j == 1 && k > 1
        return prev_approximation
    elseif (j > 1 && k == 1) 
        return T(norm(curr_grad - prev_grad) / norm_djk)
    elseif (j > 1 && k > 1 && prev_acceptance)
        return T(norm(curr_grad - prev_grad) / norm_djk)
    else
        return T(max(norm(curr_grad - prev_grad) / norm_djk, prev_approximation))
    end

end

"""
    compute_step_size(τ_lower::T, norm_grad::T, local_lipschitz_estimate::T
        ) where {T}

Computes the step size for the inner loop iterates.

# Method

The inner loop iterates are generated according to the formula

```math
    \\psi_{j+1}^k = \\psi_j^k - \\delta_k\\alpha_j^k \\dot F(\\psi_j^k).
```

The step size, ``\\alpha_j^k`` is computed as

```math
    \\alpha_j^k = 
    \\min \\left( 
    (\\tau_{\\mathrm{grad}, \\mathrm{lower}}^k)^2 / C_{j,1}^k, 
    1 / C_{j,2}^k 
    \\right)
```

where 

```math
    C_{j,1}^k = ||\\dot F(\\psi_j^k)||_2^3 + .5 \\hat{L}_j^k ||\\dot F(\\psi_j^k)||_2^2 + 1e-16
```

and

```math
    C_{j,2}^k = ||\\dot F(\\psi_j^k)||_2 + .5 \\hat{L}_j^k + 1e-16.
```

# Arguments

- `τ_lower::T`, lower bound on the gradient. 
- `norm_grad::T`, norm of the gradient at ``\\psi_j^k``.
- `local_lipschitz_estimate::T`, local lipschitz approximation at inner loop
    iteration `j` and outer loop iteration `k`.

# Return

- `αjk::T`, the step size.
"""
function compute_step_size(τ_lower::T, norm_grad::T, local_lipschitz_estimate::T) where {T}

    # compute and return step size
    cj1k = norm_grad ^ 3 + .5 * local_lipschitz_estimate * norm_grad ^ 2 + 1e-16
    cj2k = norm_grad + .5 * local_lipschitz_estimate + 1e-16
    αjk = min( (τ_lower)^2 / cj1k, 1/cj2k ) + 1e-16

    return T(αjk)
end

"""
    inner_loop!(ψjk::S, θk::S, optData::NonsequentialArmijoGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        past_acceptance::Bool, k::Int64; max_iteration = 100) where {T, S}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and
`store` in place. `ψjk` gets updated to be the terminal iterate of the inner loop;
the fields `local_lipschitz_estimate`, `norm_∇F_ψ`, and `prev_∇F_ψ` are updated in
`optData`; the fields `grad` in `store` gets updated to be the gradient at `ψjk`.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::NonsequentialArmijoGD{T}`, `struct` that specifies the optimization
    algorithm. Fields are modified during the inner loop.
- `progData::P1 where P1 <: AbstractNLPModel{T, S}`, `struct` that specifies the
    optimization problem. Fields are modified during the inner loop.
- `precomp::P2 where P2 <: AbstractPrecompute{T}`, `struct` that has precomputed
    values. Required to take advantage of this during the gradient computation.
- `store::P3 where P3 <: AbstractProblemAllocate{T}`, `struct` that contains
    buffer arrays for computation.
- `past_acceptance::Bool`, flag indicating if the previous inner loop resulted
    in a success (i.e., ``F(\\theta_k) < F(\\theta_{k-1})``).
- `k::Int64`, outer loop iteration for computation of the local Lipschitz
    approximation scheme.

## Optional Keyword Arguments

- `max_iteration = 100`, maximum number of allowable iteration of the inner loop.
    Should be kept at `100` as that is what is specified in the paper, but
    is useful to change for testing.
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::NonsequentialArmijoGD{T}, 
    progData::P1 where P1 <: AbstractNLPModel{T, S}, 
    precomp::P2 where P2 <: AbstractPrecompute{T}, 
    store::P3 where P3 <: AbstractProblemAllocate{T}, 
    past_acceptance::Bool, 
    k::Int64; 
    max_iteration = 100) where {T, S}

    # inner loop
    j::Int64 = 0
    αjk::T = T(0)
    optData.norm_∇F_ψ = optData.grad_val_hist[k]
    optData.prev_norm_step = T(0)
    while ((norm(ψjk - θk) <= 10) &&
        (optData.τ_lower < optData.norm_∇F_ψ && optData.norm_∇F_ψ < optData.τ_upper) &&
        (j < max_iteration))

        # Increment the inner loop counter
        j += 1

        # update local lipschitz estimate
        optData.local_lipschitz_estimate = update_local_lipschitz_approximation(
            j, k, optData.prev_norm_step, store.grad, optData.prev_∇F_ψ, 
            optData.local_lipschitz_estimate, past_acceptance)

        # compute the step size
        αjk = compute_step_size(optData.τ_lower, optData.norm_∇F_ψ, 
            optData.local_lipschitz_estimate)

        # save the first step size for non-sequential armijo
        if j == 1
            optData.α0k = αjk
        end

        # take step
        ψjk .-= (optData.δk * αjk) .* store.grad

        ## store values for next iteration
        optData.prev_norm_step = (optData.δk * αjk) * optData.norm_∇F_ψ 
        optData.prev_∇F_ψ .= store.grad
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)
    end

    optData.local_lipschitz_estimate = update_local_lipschitz_approximation(
            j, k, optData.prev_norm_step, store.grad,
            optData.prev_∇F_ψ, optData.local_lipschitz_estimate, past_acceptance)
end

"""
    update_algorithm_parameters!(θkp1::S, optData::NonsequentialArmijoGD{T},
        achieved_descent::Bool, iter::Int64) where {T, S}

Given that the non-sequential Armijo condition is checked, update the parameters
    the optimization method. The method updates the following variables in place.
    
- `θkp1` is updated to be the next outer loop iterate.
- `optData` has (potentially) the following fields updated: `δk`, `τ_lower`,
    `τ_upper`.

# Arguments

- `θkp1::S`, buffer array for the storage of the next iterate.
- `optData::NonsequentialArmijoGD{T}`, `struct` that specifies the optimization
    algorithm. 
- `achieved_descent::Bool`, boolean flag indicating whether or not the
    descent condition was achieved.
- `iter::Int64`, the current iteration of the method. The outer loop iteration.
    This is requried as it is used to overwrite `θkp1` with the previous iterate.
"""
function update_algorithm_parameters!(θkp1::S, optData::NonsequentialArmijoGD{T},
    achieved_descent::Bool, iter::Int64) where {T, S}
    if !achieved_descent
        θkp1 .= optData.iter_hist[iter]
        optData.δk *= .5
        return false 
    elseif optData.norm_∇F_ψ <= optData.τ_lower
        optData.τ_lower = optData.norm_∇F_ψ/sqrt(2) 
        optData.τ_upper = sqrt(10) * optData.norm_∇F_ψ
        return true
    elseif optData.norm_∇F_ψ >= optData.τ_upper
        optData.δk = min(1.5 * optData.δk, optData.δ_upper)
        optData.τ_lower = optData.norm_∇F_ψ/sqrt(2) 
        optData.τ_upper = sqrt(10) * optData.norm_∇F_ψ
        return true
    else
        optData.δk = min(1.5 * optData.δk, optData.δ_upper)
        return true
    end 
end

"""
    nonsequential_armijo_gd(optData::NonsequentialArmijoGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with non-sequential armijo and triggering
    events. The optimization algorithm is specified through `optData`, and
    applied to the problem `progData`.

# Reference(s)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. 
The ``k+1^{th}`` iterate and parameters are produced by the following procedure. 

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\delta_k \\alpha_i^k \\dot F(\\psi_i^k).
```
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

- `optData::NesterovAcceleratedGD{T}`, the specification for the optimization 
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
function nonsequential_armijo_gd(
    optData::NonsequentialArmijoGD{T},
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

    # constants required by the algorithm
    past_acceptance = true
    reference_value = F(optData.iter_hist[1])

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration counter
        iter += 1

        # inner loop
        optData.∇F_θk .= store.grad
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, past_acceptance, iter)

        # check non-sequential armijo condition
        Fx = F(x)
        achieved_descent = 
        OptimizationMethods.non_sequential_armijo_condition(Fx, reference_value, 
            optData.grad_val_hist[iter], optData.ρ, optData.δk, optData.α0k)
        
        # update the algorithm parameters and current iterate
        update_algorithm_parameters!(x, optData, achieved_descent, iter)
        
        # update histories
        past_acceptance = achieved_descent
        optData.iter_hist[iter + 1] .= x
        if past_acceptance
            # accepted case
            reference_value = Fx
            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
        else
            # rejected case
            store.grad .= optData.∇F_θk
            optData.grad_val_hist[iter + 1] = optData.grad_val_hist[iter]
        end
    end

    optData.stop_iteration = iter

    return x
end