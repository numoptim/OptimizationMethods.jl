# Date: 2025/04/10
# Author: Christian Varner
# Purpose: Implementation of non-sequential armijo line search
# with modified newton steps and fixed step size

"""
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
    newton_step::Vector{T}
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    acceptance_cnt::Int64
    τ_lower::T
    τ_upper::T
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
    threshold::T,
    max_iterations::Int64
) where {T}

    # error checking

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
        zeros(T, d),
        CircularVector(zeros(T, M)),
        T(-1),
        -1,
        0,
        T(-1),
        T(-1),
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

        # modify hessian and return the result
        res = add_identity_until_psd!(store.hess;
            λ = optData.λ,
            β = optData.β, 
            max_iterations = optData.hessian_modification_max_iteration)
        
        # take a gradient step if this was not successful
        if !res[3]
            ψjk .-= (optData.δk * optData.α) .* store.grad
        else
            optData.λ = res[2] / 2
            optData.newton_step .= res[1].L \ store.grad
            optData.newton_step .= res[1].U \ optData.newton_step
            ψjk .-= (optData.δk * optData.α) .* optData.newton_step
        end

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        OptimizationMethods.hess!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)
    end

    return j
end

"""
"""
function nonsequential_armijo_mnewton_fixed_gd(
    optData::NonsequentialArmijoFixedMNewtonGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
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
            precomp, store, iter)

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
            if (optData.acceptance_cnt % M) + 1 == optData.reference_value_index
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