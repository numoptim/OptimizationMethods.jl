# Date: 2025/04/23
# Author: Christian Varner
# Purpose: Implement the Non-sequential Armijo Line search with
# BFGS steps and fixed step sizes

"""
"""
mutable struct NonsequentialArmijoFixedBFGSGD{T} <: AbstractOptimizerData{T}
    name::String
    ∇F_θk::Vector{T}
    B_θk::Matrix{T}
    s_θk::Vector{T}
    y_θk::Vector{T}
    Bjk::Matrix{T}
    δBjk::Matrix{T}
    rjk::Vector{T}
    sjk::Vector{T}
    yjk::Vector{T}
    djk::Vector{T}
    c::T
    β::T
    norm_∇F_ψ::T
    α::T
    δk::T
    δ_upper::T
    ρ::T
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    acceptance_cnt::Int64
    τ_lower::T 
    τ_upper::T
    inner_loop_radius
    inner_loop_max_iterations
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonsequentialArmijoFixedBFGSGD(::Type{T};
    x0::Vector{T},
    c::T,
    β::T,
    α::T,
    δ0::T,
    δ_upper::T,
    ρ::T,
    M::Int64,
    inner_loop_radius::T,
    inner_loop_max_iterations::Int64,
    threshold::T,
    max_iterations::Int64
    ) where {T}

    # error checking

    # name for recording purposed
    name = "Gradient Descent with Triggering Events and Nonsequential"*
    " Armijo with fixed step size and BFGS steps."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # return the struct
    return NonsequentialArmijoFixedBFGSGD{T}(
        name,
        zeros(T, d),
        zeros(T, d, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        c,
        β,
        T(0),
        α,
        δ0,
        δ_upper,
        ρ,
        CircularVector(zeros(T, M)),
        T(0),
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
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::NonsequentialArmijoFixedBFGSGD{T},
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

        # store values for update
        optData.sjk .= -ψjk
        optData.yjk .= -store.grad

        # compute step
        optData.djk .= optData.Bjk \ store.grad

        # take step
        ψjk .-= (optData.δk * optData.α) .* store.grad

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)

        # update approximation
        optData.sjk .+= ψjk
        optData.yjk .+= store.grad
        update_success = OptimizationMethods.update_bfgs!(
            optData.Bjk, optData.rjk, optData.δBjk,
            optData.sjk, optData.yjk; damped_update = true)
        OptimizationMethods.add_identity(optData.Bjk, optData.β)

        optData.norm_∇F_ψ = norm(store.grad)
    end

    return j
end

"""
"""
function nonsequential_armijo_fixed_bfgs(
    optData::NonsequentialArmijoFixedBFGSGD{T},
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

    # Initialize hessian approximation
    fill!(optData.Bjk, 0)
    OptimizationMethods.add_identity(optData.Bjk,
        optData.c * optData.grad_val_hist[iter + 1])

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
        optData.B_θk .= optData.Bjk
        optData.s_θk .= optData.sjk
        optData.y_θk .= optData.yjk
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; radius = optData.inner_loop_radius,
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
            optData.Bjk .= optData.B_θk
            optData.sjk .= optData.s_θk
            optData.yjk .= optData.y_θk
            optData.grad_val_hist[iter + 1] = optData.grad_val_hist[iter]
        end
    end

    optData.stop_iteration = iter

    return x

end