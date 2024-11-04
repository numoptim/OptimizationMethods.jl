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
- `max_iterations::Int`, the maximum allowed iterations
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    correpsonds to the gradient norm at iteration `k`
- `stop_iteration`, iteration number that the algorithm stopped at. 
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
    threshold::T
    max_iterations::Int64
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

    d = length(x0)
    iter_hist = Vector{T}[Vector{T}(undef, d) for i = 1:max_iterations + 1]
    iter_hist[1] = x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1 # dummy value

    return LipschitzApproxGD(
        "Gradient Descent with Lipschitz Approximation (AdGD)", 
        init_stepsize, threshold, max_iterations, iter_hist, grad_val_hist, stop_iteration 
    )
end


"""
    lipschitz_approximation_gd(optData::FixedStepGD{T}, progData::P where P <: AbstractNLPModel{T, S}) 
        where {T, S}
    
Implements gradient descent with adaptive step size formed through a lipschitz approximation for the
    desired optimization problem specified by `progData`.

# References(s)

Malitsky, Y. and Mishchenko, K. (2020). "Adaptive Gradient Descent without Descent." 
    Proceedings of the 37th International Conference on Machine Learning, 
    in Proceedings of Machine Learning Research 119:6702-6712.
    Available from https://proceedings.mlr.press/v119/malitsky20a.html.


# Method

The iterates are updated according the procedure,

```math
x_{k+1} = x_{k} - \\alpha_k \\nabla f(x_{k}),
```

where ``\\alpha_k`` is the step size and ``\\nabla f`` is the gradient of the objective function ``f``.

The step size is computed depending on ``k``. 
When ``k = 0``, ``\\alpha_k = optData.init_stepsize``. 
When ``k > 0``, 

```math
\\alpha_k = \\min\\left( \\sqrt{1 + \\theta_{k-1}}\\alpha_{k-1}, 
    \\frac{||x_k - x_{k-1}||}{||\\nabla f(x_k) - \\nabla f(x_{k-1})||} \\right),
```

where ``\\theta_0 = \\inf`` and ``\\theta_k = \\alpha_k / \\alpha_{k-1}``.

# Arguments 

- `optData::FixedStepGD{T}`, the specification for the optimization method.
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
    gprev = zeros(T, size(x))
    L :: T = zero(T)
    alfa_prev :: T = zero(T)
    alfa :: T = zero(T)
    theta :: T = zero(T)

    # store values
    grad!(progData, precomp, store, x)
    gra_norm = norm(store.grad)

    optData.grad_val_hist[iter+1] = gra_norm

    # first iteration
    iter += 1
    x .-= optData.init_stepsize .* store.grad
    gprev .= store.grad
    alfa_prev = optData.init_stepsize

    # update values
    grad!(progData, precomp, store, x)
    gra_norm = norm(store.grad)
    optData.iter_hist[iter + 1] .= x
    optData.grad_val_hist[iter + 1] = gra_norm

    # main loop
    while (gra_norm > optData.threshold) && (iter < optData.max_iterations)
        iter += 1

        # compute the step size
        L = norm(store.grad - gprev) / norm(x - optData.iter_hist[iter - 1])
        if iter == 2
            alfa = 1 / (2 * L)
        else
            alfa = min(sqrt(1 + theta) * alfa_prev, 1 / (sqrt(2) * L))
        end

        # take step and update values
        x .-= alfa * store.grad
        gprev .= store.grad
        theta = alfa / alfa_prev
        alfa_prev = alfa

        grad!(progData, precomp, store, x)
        gra_norm = norm(store.grad)

        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = gra_norm 
    end

    optData.stop_iteration = iter

    # return main iterate
    return x
end