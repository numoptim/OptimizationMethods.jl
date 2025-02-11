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
    δ0::T
    ρ::T
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
    ρ::T,
    threshold::T,
    max_iterations::Int64
)

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
        ρ,
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    ) 
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
    
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, x)
    
    optData.grad_val_hist[1] = norm(store.grad)

    δiter = optData.δ0
    L_estimate = 1
    past_acceptance = true

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[1] > optData.grad_val_hist[iter + 1])

        iter += 1

        ## inner loop
        inner_loop_iter = 0
        optData.ψ .= optData.iter_hist[iter]
        optData.norm_∇F_ψ = optData.grad_val_hist[iter]
        while (norm(optData.ψ - x) <= 10) || 
            (τ_lower <= optData.norm_∇F_ψ && optData.norm_∇F_ψ <= τ_upper) ||
            (inner_loop_iter != 100)

            inner_loop_iter += 1

            ## step size computation
            if inner_loop_iter == 1 && iter == 1
                L_estimate = 1
            elseif inner_loop_iter == 1 && iter > 1
                nothing
            elseif (inner_loop_iter > 1 && iter == 1) ||
                (inner_loop_iter > 1 && iter > 1 && past_acceptance)
                L_estimate = norm(store.grad - prev_grad) / 
                    norm(δiter * αjk * prev_grad)
            else
                L_estimate = max(norm(store.grad - prev_grad) / 
                    norm(δiter * αjk * prev_grad), L_estimate)
            end       

            cj1k = optData.norm_∇F_ψ ^ 3 + .5 * L_jk * optData.norm_∇F_ψ ^ 2 + 1e-16
            cj2k = optData.norm_∇F_ψ + .5 * L_jk + 1e-16
            αjk = min( (τ_lower)^2 / cj1k, 1/cj2k ) + 1e-16

            if inner_loop_iter == 1
                optData.α_0k = αjk
            end

            ## take step
            optData.ψ .-= δiter * αjk * store.grad

            ## store values for next iteration
            optData.prev_∇F_ψ .= store.grad
            OptimizationMethods.grad!(progData, precomp, store, optData.ψ)
            optData.norm_∇F_ψ = norm(store.grad)
        end

        ## check non-sequential armijo condition
        if F(ψ) >= F(x) - optData.ρ * δiter * optData.α_0k * 
            (optData.grad_val_hist[iter]) ^ 2
            δiter *= .5 * δiter
            past_acceptance = false
        elseif norm_∇F_ψ <= τ_lower
            x .= ψ
            τ_lower, τ_upper = norm_∇F_ψ/sqrt(2), sqrt(10) * norm_∇F_ψ
            past_acceptance = true
        elseif norm_∇F_ψ >= τ_upper
            x .= ψ
            δiter = min(1.5 * δiter, optData.δ_upper)
            past_acceptance = true
        else
            x .= ψ
            δiter = min(1.5 * δiter, optData.δ_upper)
            past_acceptance = true
        end
    
        # save iterate
        optData.iter_hist[iter + 1] .= x

        OptimizationMethods.grad!(progData, precomp, store, x)
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end