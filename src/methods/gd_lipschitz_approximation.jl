# Date: 10/04/2024 
# Author: Christian Varner
# Purpose: Implement lipschitz approximation

"""
    LipschitzApproxGD{T} <: AbstractOptimizerData{T}

A structure for storing data about adaptive gradient descent
    using a Lipschitz Approximation scheme (AdGD), and the progress 
    of its application on an optimization problem.

# Fields

- `name::String`, name of the solver for reference
- `init_stepsize::T`, the initial step size for the method
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int64`, the maximum allowed iterations
- `iter_diff::Vector{T}`, a buffer for storing differences between subsequent
    iterate values that are used for computing the step size
- `grad_diff::Vector{T}`, a buffer for storing differences between gradient 
    values at adjacent iterates, which is used to compute the step size
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    corresponds to the gradient norm at iteration `k`
- `stop_iteration::Int64`, iteration number that the algorithm stopped at. 
   Iterate number `stop_iteration` is produced. 

# Constructors

    LipschitzApproxGD(::Type{T}; x0::Vector{T}, init_stepsize::T, threshold::T, 
        max_iterations::Int) where {T}

Constructs the `struct` for the optimizer.

## Arguments

- `T::DataType`, specific data type for the calculations

## Keyword Arguments

- `x0::Vector{T}`, the initial iterate for the optimizers
- `init_stepsize::T`, the initial step size for the method
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int`, the maximum number of iterations allowed  
"""
mutable struct LipschitzApproxGD{T} <: AbstractOptimizerData{T}
    name::String
    init_stepsize::T
    prev_stepsize::T
    theta::T
    lipschitz_approximation::T
    threshold::T
    max_iterations::Int64
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function LipschitzApproxGD(
    ::Type{T};
    x0::Vector{T},
    init_stepsize::T,
    threshold::T,
    max_iterations::Int,
) where {T}

    # Verify initial step size
    @assert init_stepsize > 0 "Initial step size must be positive."

    # Initialize iterate history 
    d = length(x0)
    iter_hist = Vector{T}[Vector{T}(undef, d) for i = 1:max_iterations + 1]
    iter_hist[1] = x0

    # Initialize buffers
    iter_diff = Vector{T}(undef, d)
    grad_diff = Vector{T}(undef, d)
    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1 # dummy value

    return LipschitzApproxGD(
        "Gradient Descent with Lipschitz Approximation (AdGD)", 
        init_stepsize, T(0), T(0), T(Inf), threshold, max_iterations, 
        iter_diff, grad_diff, iter_hist, grad_val_hist, stop_iteration 
    )
end


"""
    lipschitz_approximation_gd(optData::FixedStepGD{T}, progData::P where P 
        <: AbstractNLPModel{T, S}) where {T, S}
    
Implements gradient descent with adaptive step size formed through a lipschitz 
    approximation for the desired optimization problem specified by `progData`.

!!! warning 
    This method is designed for convex optimization problems.

# References(s)

Malitsky, Y. and Mishchenko, K. (2020). "Adaptive Gradient Descent without 
    Descent." 
    Proceedings of the 37th International Conference on Machine Learning, 
    in Proceedings of Machine Learning Research 119:6702-6712.
    Available from https://proceedings.mlr.press/v119/malitsky20a.html.


# Method

The iterates are updated according to the procedure,

```math
x_{k+1} = x_{k} - \\alpha_k \\nabla f(x_{k}),
```

where ``\\alpha_k`` is the step size and ``\\nabla f`` is the gradient function 
    of the objective function ``f``.

The step size is computed depending on ``k``. 
    When ``k = 0``, ``\\alpha_k = optData.init_stepsize``. 
    When ``k > 0``, 

```math
\\alpha_k = \\min\\left\\lbrace \\sqrt{1 + \\theta_{k-1}}\\alpha_{k-1}, 
    \\frac{\\Vert x_k - x_{k-1} \\Vert}{\\Vert \\nabla f(x_k) - 
    \\nabla f(x_{k-1})\\Vert} \\right\\rbrace,
```

where ``\\theta_0 = \\inf`` and ``\\theta_k = \\alpha_k / \\alpha_{k-1}``.

# Arguments 

- `optData::LipschitzApproxGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 
"""
function lipschitz_approximation_gd(
    optData::LipschitzApproxGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S} 

    # initialize storage and pre-compute values
    precomp, store = initialize(progData)

    # get initial iterate
    iter = 0
    x = copy(optData.iter_hist[iter + 1])

    # initialize additional buffer
    step_size::T = optData.init_stepsize # step size

    # store values
    grad!(progData, precomp, store, x)
    gra_norm = norm(store.grad)

    optData.grad_val_hist[iter+1] = gra_norm

    # main loop
    while (gra_norm > optData.threshold) && (iter < optData.max_iterations)
        iter += 1

        # store values
        optData.iter_diff .= -x
        optData.grad_diff .= -store.grad

        # compute the step size
        x .-= step_size * store.grad 
        optData.iter_hist[iter + 1] .= x
        
        grad!(progData, precomp, store, x)
        gra_norm = norm(store.grad)
        optData.grad_val_hist[iter + 1] = gra_norm

        # compute the step size for the next iteration
        optData.iter_diff .+= x
        optData.grad_diff .+= store.grad

        optData.theta = step_size / optData.prev_stepsize #Iter==1, Sets to Inf
        optData.prev_stepsize = step_size

        optData.lipschitz_approximation = norm(optData.grad_diff) / 
            norm(optData.iter_diff)
        step_size = min(sqrt(1 + optData.theta) * optData.prev_stepsize, 
            1 / (2 * optData.lipschitz_approximation))
    end

    optData.stop_iteration = iter
    return x
end