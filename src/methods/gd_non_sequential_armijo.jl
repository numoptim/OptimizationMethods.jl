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
TODO
"""
mutable struct NonsequentialArmijoGD{T} <: AbstractOptimizerData{T}
    name::String
    ψ::Vector{T}
    norm_∇F_ψ::T
    prev_∇F_ψ::Vector{T}
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

# Method

The local Lipschitz approximation method is conducted as follows. Let ``j``
    be the current inner loop iteration counter, and let ``k`` be the current
    outer loop iteration counter. Let ``\\psi_j^k`` be the ``j^{th}`` iterate
    of the ``k^{th}`` outer loop, and let ``\\hat L_{j}^k`` be the ``j^th`` estimate
    of the ``k^{th}`` outer loop of the local Lipschitz constant. The local
    estimate is updated according the following five cases.

1) When ``j == 1`` and ``k == 1``, this is the first iteration of the first 
inner loop, and as there is no information available we set it to `1.0`.

2) When ``j == 1`` and ``k > 1``, this is the first iteration of the ``k^{th}``
inner loop, and we return ``L_{j_{k-1}}^{k-1}`` which is the local Lipschitz
estimates formed using information at the terminal iteration of the ``k-1^{th}``
inner loop (i.e., this is the latest estimate).

3) When ``j > 1`` and ``k == 1``, this is an inner loop iteration where we have
possible taken multiple steps, so we return the most 'local' estimate
of the local Lipschitz constant which is

```math
||\\dot F(\\psi_{j}^k) - \\dot F(\\psi_{j-1}^k||_2 / 
    ||\\psi_{j}^k - \\psi_{j-1}^k||_2.
```

4) When ``j > 1`` and ``k > 1`` and ``\\psi_{j_{k-1}}^{k-1}`` satisfied the
descent condition, then we return 

```math
||\\dot F(\\psi_{j}^k) - \\dot F(\\psi_{j-1}^k||_2 / 
    ||\\psi_{j}^k - \\psi_{j-1}^k||_2.
```

5) When ``j > 1`` and ``k > 1`` and ``\\psi_{j_{k-1}}^{k-1}`` did not satisfy the
descent condition, then we return 

```math
max\\left( ||\\dot F(\\psi_{j}^k) - \\dot F(\\psi_{j-1}^k||_2 / 
    ||\\psi_{j}^k - \\psi_{j-1}^k||_2, \\hat L_{j-1}^k \\right).
```

# Arguments

- `j::Int64`, inner loop iteration.
- `k::Int6`, outer loop iteration.
- `djk::S`, difference between `\\psi_j^k` and ``\\psi_{j-1}^k``. On the first
    iteration of the inner loop this is set to `0`.
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
    djk::S, curr_grad::S, prev_grad::S, prev_approximation::T, 
    prev_acceptance::Bool) where {T, S}

    # compute and return local estimate
    if j == 1 && k == 1
        return T(1)
    elseif j == 1 && k > 1
        return prev_approximation
    elseif (j > 1 && k == 1) 
        return T(norm(curr_grad - prev_grad) / norm(djk))
    elseif (j > 1 && k > 1 && prev_acceptance)
        return T(norm(curr_grad - prev_grad) / norm(djk))
    else
        return T(max(norm(curr_grad - prev_grad) / norm(djk), prev_approximation))
    end

end

"""
    compute_step_size(τ_lower::T, norm_grad::T, local_lipschitz_estimate::T
        ) where {T}

Computes the step size for the inner loop iterates.

# Method

# Arguments

# Return
"""
function compute_step_size(τ_lower::T, norm_grad::T, local_lipschitz_estimate::T) where {T}

    # compute and return step size
    cj1k = norm_grad ^ 3 + .5 * local_lipschitz_estimate * norm_grad ^ 2 + 1e-16
    cj2k = norm_grad + .5 * local_lipschitz_estimate + 1e-16
    αjk = min( (τ_lower)^2 / cj1k, 1/cj2k ) + 1e-16

    return T(αjk)
end

"""
TODO
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
    j = 0
    αjk = 0
    optData.norm_∇F_ψ = optData.grad_val_hist[k]
    while ((norm(ψjk - θk) <= 10) &&
        (optData.τ_lower < optData.norm_∇F_ψ && optData.norm_∇F_ψ < optData.τ_upper) &&
        (j < max_iteration))

        # Increment the inner loop counter
        j += 1

        # update local lipschitz estimate
        optData.local_lipschitz_estimate = update_local_lipschitz_approximation(
            j, k, -optData.δk * αjk .* optData.prev_∇F_ψ, store.grad,
            optData.prev_∇F_ψ, optData.local_lipschitz_estimate, past_acceptance)

        # compute the step size
        αjk = compute_step_size(optData.τ_lower, optData.norm_∇F_ψ, 
            optData.local_lipschitz_estimate)

        # save the first step size for non-sequential armijo
        if j == 1
            optData.α0k = αjk
        end

        # take step
        ψjk .-= optData.δk * αjk * store.grad

        ## store values for next iteration
        optData.prev_∇F_ψ .= store.grad
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)
    end

    optData.local_lipschitz_estimate = update_local_lipschitz_approximation(
            j, k, -optData.δk * αjk .* optData.prev_∇F_ψ, store.grad,
            optData.prev_∇F_ψ, optData.local_lipschitz_estimate, past_acceptance)
end

"""
TODO
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
TODO
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
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, past_acceptance, iter)

        # check non-sequential armijo condition
        Fx = F(x)
        achieved_descent = 
        OptimizationMethods.non_sequential_armijo_condition(Fx, reference_value, 
            optData.grad_val_hist[iter], optData.ρ, optData.δk, optData.α0k)
        
        # update the algorithm parameters and current iterate
        update_algorithm_parameters!(x, optData, achieved_descent, iter)
        past_acceptance = achieved_descent
        if past_acceptance
            reference_value = Fx
        end 
    
        # save next gradient
        optData.iter_hist[iter + 1] .= x
        OptimizationMethods.grad!(progData, precomp, store, x)
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end