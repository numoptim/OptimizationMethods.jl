# Date: 2025/04/15
# Author: Christian Varner
# Purpose: Implementation of fixed step size initialization, (non-)monotone
# modified newton gradient descent

"""
    FixedModifiedNewtonNLSMaxValGD{T} <: AbstractOptimizerData{T}

Mutable structure that represents, parameterizes, and stores values
for each iteration of a fixed step size, modified newton step direction,
gradient descent algorithm, which is globalized through a line search 
scheme.

# Fields
    
- `name::String`, name of the solver for reference.
- `α::T`, the initial step size for line search.  
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `step::Vector{T}`, buffer array for the step direction used.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference objective value for the line search criterion.
- `objective_hist::CircularVector{T, Vector{T}}`, buffer array of size     
    `window_size` that stores `window_size` previous objective values.
- `max_value::T`, maximum value of `objective_hist`. This is the reference 
    objective value used in the line search procedure.
- `max_index::Int64`, index of the maximum value that corresponds to the 
    reference objective value.
- `β::T`, argument for the function used to modify the hessian.
- `λ::T`, argument for the function used to modify the hessian.
- `hessian_modification_max_iteration::Int64`, max number of attempts
    at modifying the hessian per-step.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
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

    FixedModifiedNewtonNLSMaxValGD(::Type{T}; x0::Vector{T}, α::T, δ::T, ρ::T,
        line_search_max_iteration::Int64, window_size::Int64, β::T, λ::T,
        threshold::T, max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial starting point for the optimization algorithm.
- `α::T`, the initial step size for line search.     
- `δ::T`, backtracking decreasing factor applied to `α` when the line search
    criterion is not satisfied
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correpsond to stricter descetn conditions, and
    smaller values correspond to looser descent conditions.
- `line_search_max_iteration::Int64`, maximum number of iterations for
    line search.
- `window_size::Int64`, number of previous objective values that are used
    to construct the reference value for the line search criterion.
- `β::T`, argument for the function used to modify the hessian.
- `λ::T`, argument for the function used to modify the hessian.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct FixedModifiedNewtonNLSMaxValGD{T} <: AbstractOptimizerData{T}
    name::String
    # parameters for line search
    α::T
    δ::T
    ρ::T
    line_search_max_iteration::Int64
    step::Vector{T}
    # parameters for non-monotone line search
    window_size::Int64
    objective_hist::CircularVector{T, Vector{T}}
    max_value::T
    max_index::Int64
    # parameters for modified Newton
    β::T
    λ::T
    hessian_modification_max_iteration::Int64
    # default parameters
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function FixedModifiedNewtonNLSMaxValGD(::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    ρ::T,
    line_search_max_iteration::Int64,
    window_size::Int64,
    β::T,
    λ::T,
    hessian_modification_max_iteration::Int64,
    threshold::T,
    max_iterations::Int64) where {T}

    # error checking

    # length for initialization purposes
    d = length(x0)

    # initialization iterate history
    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 
        1:max_iterations + 1]
    iter_hist[1] = x0

    # initialization of gradient and dummy value for stop_iteration
    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    # name of the optimizer for reference
    name = "Gradient Descent with non-monotone line search using the max value"*
    " of the previous $(window_size) values" 

    return FixedModifiedNewtonNLSMaxValGD{T}(
        name,
        α,
        δ,
        ρ,
        line_search_max_iteration,
        zeros(T, d),                                                # step
        window_size,
        CircularVector(zeros(T, window_size)),                      # objective-hist
        T(0),                                                       # max_value
        -1,                                                         # max_index
        β,
        λ,
        hessian_modification_max_iteration,
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    )
end

"""
    fixed_modified_newton_nls_maxval_gd(
        optData::FixedModifiedNewtonNLSMaxValGD{T},
        progData::P where P <: AbstractNLPModel{T, S}
        ) where {T, S}

Implementation of a fixed step size, modified newton step direction,
gradient descent method, which is globalized through a (non-)monotone
Armijo line search scheme. The method and its parameters are specified
by `optData`, and the optimization routine is applied to a problem specified
by `progData`.

# Reference(s)

[Nocedal and Wright, "Numerical Optimization". Springer. 2nd Edition. 
    Page 48.](@cite nocedal2006Numerical)

# Method

The method we describe is a version of Algorithm 3.2 in the 
    [reference above](@cite nocedal2006Numerical).

Let ``k + 1 \\in \\mathbb{N}``, and ``\\theta_{k} \\in \\mathbb{R}^n``.
Let ``F(\\theta)`` be a function, and let ``\\dot F(\\theta)`` and
``\\ddot F(\\theta)`` be the gradient and hessian of ``F(\\theta)``,
respectively. Let ``\\alpha \\in \\mathbb{R}_{>0}``, ``\\delta \\in (0, 1)``,
``\\rho \\in (0, 1)`` be algorithmic parameters. Then, iterate ``\\theta_{k+1}``
is produced by the following procedure.

```math
    \\theta_{k+1} = \\theta_{k} - \\delta^j \\alpha d_k, ~d_k \\in\\mathbb{R}^n
```

where ``j`` is the smallest number such that ``j + 1 \\in \\mathbb{N}`` and ...
**finish this**

The vector ``d_k \\in \\mathbb{R}^n`` is defined as either the modified
Newton step, or the negative gradient step. In particular, a
[modification routine](@ref OptimizationMethods.add_identity_until_pd!)
is applied to the Hessian that adds a scaling of a identity matrix until
a positive definite matrix is obtained. If this subroutine is successful, 
returning the upper Cholesky factor ``L_k``, then

```math
    d_k = (L_k)^{-1}\\left(L_k^\\intercal\\right)^{-1} \\dot F(\\theta_k),
```

for which the solution is provided by an 
    [upper](@ref OptimizationMethods.upper_triangle_solve!) 
    and [lower](@ref OptimizationMethods.lower_triangle_solve!) 
    triangular matrix solve.

If this subroutine is not successful in the allotted number of iteration,  
then we simply set ``d_k = \\dot F(\\theta)``.

# Arguments

- `optData::FixedModifiedNewtonNLSMaxValGD`, the specification for 
    the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function fixed_modified_newton_nls_maxval_gd(
    optData::FixedModifiedNewtonNLSMaxValGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
    ) where {T, S}

    # initializations
    precomp, store = initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)

    # Iteration 0
    iter = 0

    # Update iteration 
    x = copy(optData.iter_hist[iter + 1]) 
    grad!(progData, precomp, store, x)
    hess!(progData, precomp, store, x)

    # Store Values
    optData.grad_val_hist[iter + 1] = norm(store.grad)

    # Update the objective cache
    optData.max_value = F(x)
    optData.objective_hist[iter + 1] = optData.max_value
    optData.max_index = iter + 1

    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # update the iteration number
        iter += 1

        # modify the current newton step
        res = OptimizationMethods.add_identity_until_pd!(
            store.hess;
            λ = optData.λ,
            β = optData.β,
            max_iterations = optData.hessian_modification_max_iteration
        )

        # subroutine success compute step, otherwise use negative gradient
        optData.step .= store.grad
        if res[2]
            optData.λ = res[1] / 2
            OptimizationMethods.lower_triangle_solve!(optData.step, store.hess')
            OptimizationMethods.upper_triangle_solve!(optData.step, store.hess)            
        end
        success = OptimizationMethods.backtracking!(
            x,
            optData.iter_hist[iter],
            F,
            store.grad,
            optData.step,
            optData.max_value,
            optData.α,
            optData.δ,
            optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        # if backtracking is not successful, return the previous point
        if !success
            optData.stop_iteration = (iter - 1)
            return optData.iter_hist[iter]
        end

        # compute the next gradient and hessian values
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.hess!(progData, precomp, store, x)
        
        # store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)

        # update the objective cache for non-monotone line search
        F_x = F(x)
        optData.objective_hist[iter + 1] = F_x
        if (iter % optData.window_size) + 1 == optData.max_index
            optData.max_value, optData.max_index = 
                findmax(optData.objective_hist)
        end
    end

    optData.stop_iteration = iter

    return x
end