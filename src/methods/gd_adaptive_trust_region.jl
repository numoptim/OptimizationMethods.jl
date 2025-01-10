# Date: 01/07/2025
# Author: Christian Varner
# Purpose: Implementation of the adaptive trust region method by
# Gratton

"""
    FirstOrderAdaptiveTrustRegionGD{T} <: AbstractOptimizerData{T}

A mutable `struct` for storing data about gradient descent using
    the (simplified) adaptive trust region framework, and the progress of its
    application on an optimization problem.

# Fields

- `name:String`, name of the solver for reference.
- `τ::T`, scalar that is analogous to the step size in our implementation.
- `μ::T`, scalar that is used for the trust region.
- `ζ::T`, scalar that is used for the trust region.
- `w::Vector{T}`, buffer array for weights used for the trust region.
- `Δ::vector{T}`, buffer array for the trust region (region is defined coordinate 
    wise).
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    correpsonds to the gradient norm at iteration `k`.
- `stop_iteration::Int64`, the iteration number that the solver stopped on.
    The terminal iterate is saved at `iter_hist[stop_iteration+1]`.

# Constructors

    FirstOrderAdaptiveTrustRegionGD(::Type{T}; x0::Vector{T}, τ::T, μ::T, ζ::T, 
        threshold::T, max_iterations::Int64) where {T}

Constructs the `struct` for the (simplified) Adaptive Trust Region optimization
    method.

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `τ::T`, scalar that is analogous to the step size in our implementation.
- `μ::T`, scalar that is used for the trust region.
- `ζ::T`, scalar that is used for the trust region.
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct FirstOrderAdaptiveTrustRegionGD{T} <: AbstractOptimizerData{T}
    name::String
    τ::T
    μ::T
    ζ::T
    w::Vector{T}
    Δ::Vector{T}
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64

    FirstOrderAdaptiveTrustRegionGD{T}(name, τ, μ, ζ, threshold, max_iterations,
        iter_hist, grad_val_hist, stop_iteration) where {T} = 
    begin
        d = length(iter_hist[1])
        return new(name, τ, μ, ζ, zeros(T, d), zeros(T, d), threshold,
             max_iterations, iter_hist, grad_val_hist, stop_iteration)
    end
end
function FirstOrderAdaptiveTrustRegionGD(
    ::Type{T};
    x0::Vector{T},
    τ::T,
    μ::T,
    ζ::T,
    threshold::T,
    max_iterations::Int64,
) where {T}

    name = "Gradient Descent with a (Simplified) Adaptive Trust Region"

    # create buffer arrays for iterate history and gradient norm
    d = length(x0)
    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:max_iterations + 1]
    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1 ## dummy value

    return FirstOrderAdaptiveTrustRegionGD{T}(name, τ, μ, ζ, threshold,
        max_iterations, iter_hist, grad_val_hist, stop_iteration)
end

"""
    first_order_adaptive_trust_region_gd(
        optData::FirstOrderAdaptiveTrustRegionGD{T},
        progData::P where P <: AbstractNLPModel{T, S}
    )

Implements gradient descent using the (simplified) adaptive trust region 
    framework and applies the method to the optimization problem specified
    by `progData`.

!!! warning
    Method is designed for non-convex functions that are globally Lipschitz
    continuous (i.e., the norm gradient is bounded on the entire domain).

# Reference(s)

[Gratton, Serge, et al. “First-Order Objective-Function-Free Optimization 
Algorithms and Their Complexity.” Arxiv, Mar. 2022, 
https://doi.org/10.48550/arXiv.2203.01757.](@cite gratton2022First-order)


# Method

The method that is implemented is a special case of 
[Algorithm 2.1](@cite gratton2022First-order) that is used in the papers 
numerical experiments.
For all ``k+1 \\in \\mathbb{N}``, the algorithm is composed of four steps.
First, the ``i^{th}`` coordinate in the ``k^{th}`` iteration of the trust region 
is defined as
```math
    \\Delta_{i, k} = \\frac{|g_{i, k}|}{w_{i, k}},
```
where ``g_{i,k}`` is the ``i^{th}`` coordinate of ``\\dot F(x_k)``, the gradient
vector of ``F`` at iterate ``x_k``, and 
```math
    w_{i,k} = \\left(\\zeta + \\sum_{l=0}^k g_{i,l}^2 \\right)^\\mu.
```
Second, an approximation to the hessian is computed which is denoted as ``B_k``,
which we simple set to ``0``.
Third, the step ``s_k``, is generated to satisfy the following
```math
    |s_{i,k}| \\leq \\Delta_{i,k}
```
and
```math
    g_k^\\intercal s_k + \\frac{1}{2} s_k^\\intercal B_k s_k \\leq
    \\tau \\left( g_k^\\intercal s_k^Q +
    \\frac{1}{2} (s_k^{Q})^\\intercal B_k s_k^Q \\right)
```
where
```math
    s_{i, k}^Q = -\\mathrm{sgn}(g_{i,k})\\Delta_{i,k}.
```
In the context of our implementation 
``s_{i,k} = -\\tau\\mathrm{sgn}(g_{i,k})\\Delta_{i,k}``.
Fourth, the step is taken such that ``x_{k+1} = x_{k} + s_{k}``.

!!! note
    This method is a special case of 
    [Algorithm 2.1](@cite gratton2022First-order), and appears in 
    the numerical section of the paper.

# Arguments

- `optData::BarzilaiBorweinGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function first_order_adaptive_trust_region_gd(
    optData::FirstOrderAdaptiveTrustRegionGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize storage and pre-computed values
    precomp, store = OptimizationMethods.initialize(progData)

    # iteration 0
    iter = 0

    x = copy(optData.iter_hist[iter + 1])

    grad!(progData, precomp, store, x)
    optData.grad_val_hist[iter + 1] = norm(store.grad)
    optData.w .= (optData.ζ .+ (store.grad .^ 2)) .^ optData.μ

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)
        
        iter += 1

        # compute trust region radius
        optData.Δ .= abs.(store.grad) ./ optData.w

        # take step
        x .-= optData.τ .* sign.(store.grad) .* optData.Δ
        grad!(progData, precomp, store, x)

        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # prep for next iteration
        optData.w .= ((optData.w .^ (1/optData.μ)) + 
            (store.grad .^ 2)) .^ optData.μ
    end

    optData.stop_iteration = iter

    return x
end