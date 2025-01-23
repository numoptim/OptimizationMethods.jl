# Part of OptimizationMethods.jl
# Date: 01/22/2025
# Author: Christian Varner
# Purpose: Implementation of the two step negative curvature
# method.

"""
    NegativeCurvatureTwoStepGD{T} <: AbstractOptimizerData{T}

A mutable struct that represents gradient descent using negative curvature directions.
    It stores the specification for the mthod and records values during iteration.

# Fields

- `name::String`, name of the optimizer for recording purposes.
- `alpha::T`, step size for the negative gradient direction.
- `beta::T`, step size for the negative curvature direction.
- `eigenvector_min::Vector{T}`, buffer array for the eigenvector corresponding
    to the smallest eigenvalue.
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

    NegativeCurvatureTwoStepGD(::Type{T}; x0::Vector{T}, alpha::T, beta::T, 
        threshold::T, max_iterations::Int64) where {T}

Constructs an instance of type `NegativeCurvatureTwoStepGD{T}`.

## Arguments

- `T::DataType`, type for data and computation

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `alpha::T`, step size for the negative gradient direction.
- `beta::T`, step size for the negative curvature direction.
- `init_norm_damping_factor::T`, initial damping factor, which will correspond
    to the reciprocoal of the initial step size. 
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
"""
mutable struct NegativeCurvatureTwoStepGD{T} <: AbstractOptimizerData{T}
    name::String
    alpha::T
    beta::T
    eigenvector_min::Vector{T}
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NegativeCurvatureTwoStepGD(::Type{T};
    x0::Vector{T},
    alpha::T,
    beta::T,
    threshold::T,
    max_iterations::Int64
) where {T}
    
    # name of optimizer
    name::String = "Gradient Descent using Negative Curvature Information"

    # initialize iter_hist and grad_val_hist
    d::Int64 = length(x0)
    iter_hist::Vector{Vector{T}} = 
        Vector{Vector{T}}([Vector{T}(undef, d) for i in 1:(max_iterations + 1)])
    iter_hist[1] = x0
    
    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1

    # return progData
    return NegativeCurvatureTwoStepGD{T}(name, alpha, beta, zeros(T, d),
        threshold, max_iterations, iter_hist, grad_val_hist, 
        stop_iteration)
end

"""
    negative_curvature_gd(optData::NegativeCurvatureTwoStepGD{T},
        progData::P where P <: AbstractNLPModel{T,S}) where {T, S}

Implementation of gradient descent with negative curvature information using
    the two step method with the specification defined in `optData` on the problem
    specified by `progData`.

!!! note
    The implementation and method description is a special case of the 
    general framework outlined in [Algorithm 1](@cite curtis2018Exploiting)
    of the linked paper.

# Reference(s)

[Curtis, F.E., Robinson, D.P. "Exploiting negative curvature in deterministic
    and stochastic optimization". Math. Program. 176, 69-94 (2019). 
    https://doi.org/10.1007/s10107-018-1335-8](@cite curtis2018Exploiting)

# Method

Let ``\\theta_{k}`` be the ``k^{th}`` iterate, let ``\\alpha \\in \\mathbb{R}``, 
and let ``\\beta \\in \\mathbb{R}``. Let ``\\dot F`` be the gradient, and
``\\ddot F`` be the hessian of the function ``F``. The optimization method generates
the ``k+1^{th}`` iterate by taking two steps defined as follows.

The first step is the negative curvature step. In particular, when
``\\ddot F(\\theta_k)`` has a negative eigenvalue, let ``d_k`` be a scaled
version of the eigenvector, corresponding to the smallest eigenvalue ``\\lambda_k``, 
such that ``\\dot F(\\theta_k)^\\intercal d_k \\leq 0`` and 
``||d_k||_2 = |\\lambda_k|``. If the smallest eigenvalue is
non-negative, then ``d_k = 0``. The second step direction is 
then ``-\\dot F(\\theta_k + \\beta d_k)``.

Combining these two directions, the ``k+1^{th}`` is generated as
```math
    \\theta_{k+1} = \\theta_{k} + \\beta d_k - 
        \\alpha \\dot F(\\theta_k + \\beta d_k). 
```

!!! note
    Let ``L`` be the global Lipschitz constant for the gradient, and let
    ``\\sigma`` be the global Lipschitz constant for the hessian.
    For the method to theoretical converge, ``\\alpha \\in (0, 2/L)`` 
    and ``\\beta \\in (0, 3/\\sigma)``.

# Arguments

- `optData::WeightedNormDampingGD{T}`, specification for the optimization algorithm.
- `progData::P where P <: AbstractNLPModel{T, S}`, specification for the problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 
"""
function negative_curvature_two_step_gd(
    optData::NegativeCurvatureTwoStepGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem
    precomp, store = initialize(progData)

    # initialization of variables for optimization
    iter::Int64 = 0
    x::S = copy(optData.iter_hist[1])

    # compute the initial gradient
    grad!(progData, precomp, store, x)
    optData.grad_val_hist[1] = norm(store.grad)

    # main optimization loop
    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)
        
        # increment iteration
        iter += 1

        # compute negative curvature direction
        hess!(progData, precomp, store, x)
        lambda_min = eigmin(store.hess)
        if lambda_min < 0
            optData.eigenvector_min = eigvecs(store.hess, [lambda_min])
            
            # scale and take step
            optData.eigenvector_min ./= norm(optData.eigenvector_min)
            optData.eigenvector_min .*= abs(lambda_min)
            optData.eigenvector_min .*= dot(store.grad, 
                optData.eigenvector_min) < 0 ? 1 : -1

            x .+= progData.beta .* optData.eigenvector_min
        end

        # compute gradient direction
        grad!(progData, precomp, store, x)
        x .-= optData.alpha .* store.grad

        # update history
        grad!(progData, precomp, store, x)
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end
