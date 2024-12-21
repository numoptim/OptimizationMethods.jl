# Date: 11/18/2024
# Author: Christian Varner
# Purpose: Implement Nesterov's Accelerated Gradient Descent

"""
    NesterovAcceleratedGD{T} <: AbstractOptimizerData

A structure that represents Nesterov Accelerated Gradient Descent. Stores
variables related to the method, and tracks quantities as the algorithm
progresses.

# Fields

- `name:String`, name of the solver for reference.
- `step_size::T`, step size used in the method. 
- `z::Vector{T}`, buffer array for auxiliary iterate sequence
- `y::Vector{T}`, buffer array for convex combination of iterate and auxiliary
    sequence 
- `B::T`, auxiliary quadratic scaling term for computing acceleration weights 
    and step size.
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
    
        NesterovAcceleratedGD(::Type{T}; x0::Vector{T}, step_size::T,
            threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `step_size::T`, step size used in the method. 
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct NesterovAcceleratedGD{T} <: AbstractOptimizerData{T}
    name::String
    step_size::T
    z::Vector{T}
    y::Vector{T}
    B::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64

    # inner constructor
    NesterovAcceleratedGD{T}(name, step_size, z, y, B, threshold, 
    max_iterations, iter_hist, grad_val_hist, stop_iteration) where {T} = 
    begin

        # test invariants
        @assert step_size > 0 "step size is non-positive."
        @assert threshold > 0 "`threshold` is non-positive."
        @assert max_iterations >= 0 "`max_iterations` is negative"
        @assert length(grad_val_hist) == max_iterations + 1 "`grad_val_hist`"*
        "should be of $(max_iterations+1) not $(length(grad_val_hist))"
        @assert length(iter_hist) == max_iterations + 1 "`iter_hist` should"*
        "be of length $(max_iterations+1) not $(length(iter_hist))"
        @assert length(z) == length(y) "Length of z and y need to be equal"

        # return new object
        return new(name, step_size, z, y, B, threshold, 
        max_iterations, iter_hist, grad_val_hist, stop_iteration)
    end
end
function NesterovAcceleratedGD(
    ::Type{T};
    x0::Vector{T},
    step_size::T,
    threshold::T,
    max_iterations::Int64,
) where {T}

    # name
    name::String = "Nesterov Accelerated Gradient Descent"

    d = length(x0)

    # initialization of buffers for the algorithm
    z = Vector{T}(undef, d)
    y = Vector{T}(undef, d)

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # initialize the structure
    return NesterovAcceleratedGD{T}(name, step_size, z, y, T(0), 
    threshold, max_iterations, iter_hist, grad_val_hist, stop_iteration)
end

"""
    nesterov_accelerated_gd(optData::NesterovAcceleratedGD{T}, progData::P
        where P <: AbstractNLPModel{T, S}) where {T, S}

Implements Nesterov's Accelerated Gradient Descent as specified by `optData` on
the problem specified by `progData`.

!!! warning
    This algorithm is designed for convex problems.

# Reference(s)

- See Algorithm 1 of [Li et. al. "Convex and Non-convex Optimization Under 
    Generalized Smoothness". arxiv, https://arxiv.org/abs/2306.01264.](@cite 
    li2023Convex)
- See line-search based approach in [Nesterov, Yurii. 1983. “A Method for 
    Solving the Convex Programming Problem 
    with Convergence Rate O(1/K^2).” Proceedings of the USSR Academy of Sciences 
    269:543–47.](@cite nesterov1983Method)

# Method

Let the objective function be denoted by ``F(\\theta)`` and ``\\nabla F(
    \\theta)`` denote its gradient function. Given ``\\theta_0`` and  a step 
    size ``\\alpha`` (equal to `optData.step_size`), the method produces five 
    sequences. At ``k=0``,

```math 
\\begin{cases}
    B_0 &= 0 \\\\
    z_0 &= \\theta_0 \\\\
    \\Delta_0 & = 1 \\\\
    y_0 &= \\theta_0;
\\end{cases}
```
and, for ``k\\in\\mathbb{N}``,
```math 
\\begin{cases}
    B_{k} &= B_{k-1} + \\Delta_{k-1} \\\\
    \\theta_k &= y_{k-1} - \\alpha \\nabla F(y_{k-1}) \\\\
    z_k &= z_{k-1} - \\alpha\\Delta_{k-1}\\nabla F(y_{k-1}) \\\\
    \\Delta_k &= \\frac{1}{2}\\left( 1 + \\sqrt{4 B_{k} + 1}  \\right) \\\\
    y_k &= \\theta_k + \\frac{\\Delta_{k}}{B_k + \\Delta_k + \\alpha^{-1}} 
    (z_k - \\theta_k).
\\end{cases}
```

The iterate sequence of interest is ``\\lbrace \\theta_k \\rbrace``.

# Arguments

- `optData::NesterovAcceleratedGD{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function nesterov_accelerated_gd(
    optData::NesterovAcceleratedGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data and save initial values
    precomp, store = OptimizationMethods.initialize(progData)

    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, x)

    grad_norm = norm(store.grad)
    optData.grad_val_hist[1] = grad_norm

    # Terms at Iteration 0
    step_size = optData.step_size        
    optData.B = T(0)
    optData.z = copy(x)
    Δ::T = 1            #1/2 × (1 + √(4 B₀ + 1  )) = 1/2 × (1 + 1) = 1
    optData.y = copy(x)


    while (iter < optData.max_iterations) && (grad_norm > optData.threshold)
        iter += 1

        # Update Sequences 
        optData.B += Δ
        x .= optData.y - step_size * store.grad 
        optData.z .-= step_size * Δ * store.grad 
        Δ = T(0.5 * (1 + sqrt(4*optData.B + 1)))
        optData.y .= x + Δ / (optData.B + Δ + 1/step_size) * 
            (optData.z - x)

        # update histories
        OptimizationMethods.grad!(progData, precomp, store, x)
        grad_norm = norm(store.grad)

        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = grad_norm

        # compute the gradient at y
        OptimizationMethods.grad!(progData, precomp, store, optData.y)
    end
    
    optData.stop_iteration = iter

    return x

end