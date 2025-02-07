# Date: 02/05/2025
# Author: Christian Varner
# Purpose: Implementation of gradient descent using a basic
# version of Wolfe line search

"""
    WolfeGD{T} <: AbstractOptimizerData{T}

Mutable `struct` that represents gradient descent using the ELBS routine to
satisfy weak wolfe conditions. 

# Fields

- `name::String`, name of the optimizer for recording purposes
- `α::T`, initial step size used in the line search procedure. 
- `δ::T`, inflation factor applied to the step size.
- `c1::T`, term used in the sufficient decrease condition. Larger values require
    higher amounts of descent per iteration, while smaller values indicate
    a less strict descent condition.
- `c2::T`, term used in the curvature condition. Larger values are stricter.
- `gkm1::Vector{T}`, buffer array for the gradient vector.
- `line_search_max_iterations::Int`, maximum number of iterations the
    line search procedure is allowed to take.
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

    WolfeEBLSGD(::Type{T}; x0::Vector{T}, α::T, δ::T, c1::T, c2::T, 
    line_search_max_iterations::Int, threshold::T, max_iterations::Int) 
    where {T}

## Arguments

- `T::DataType`, type for data and computation

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `α::T`, initial step size used in the line search procedure.
- `δ::T`, inflation factor applied to the step size.
- `c1::T`, term used in the sufficient decrease condition. Larger values require
    higher amounts of descent per iteration, while smaller values indicate
    a less strict descent condition.
- `c2::T`, term used in the curvature condition. Larger values are stricter.
- `line_search_max_iterations::Int`, maximum number of iterations the
    line search procedure is allowed to take.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
"""
mutable struct WolfeEBLSGD{T} <: AbstractOptimizerData{T}
    name::String
    α::T
    δ::T
    c1::T
    c2::T
    gkm1::Vector{T}
    line_search_max_iterations::Int
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function WolfeEBLSGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ::T,
    c1::T,
    c2::T, 
    line_search_max_iterations::Int,
    threshold::T,
    max_iterations::Int
) where {T}

    # name of the optimizer
    name = "Gradient Descent with Line Search for Weak Wolfe Conditions"

    # initialize iter_hist and grad_val_hist
    d::Int64 = length(x0)
    iter_hist::Vector{Vector{T}} = 
        Vector{Vector{T}}([Vector{T}(undef, d) for i in 1:(max_iterations + 1)])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1) 
    stop_iteration::Int64 = -1 ## dummy value

    return WolfeEBLSGD(name, α, δ, c1, c2, zeros(T, d),
        line_search_max_iterations, threshold,
        max_iterations, iter_hist, grad_val_hist, stop_iteration)
end

"""
    wolfe_ebls_gd(optData::WolfeEBLSGD{T}, progData::P 
        where P <: AbstractNLPModel{T, S})

Implementation of gradient descent using a line search procedure to satisfy 
    the weak wolfe conditions on the optimization problem specified by
    `progData`.

# Reference(s)

[Wright, Stephen J., and Benjamin Recht. "Optimization for Data Analysis." 
    Cambridge: Cambridge University Press, 2022. Print.](@cite wright2020Optimization)

# Method

The method generates an iterate sequence defined by
```math
    \\theta_{k} = \\theta_{k-1} - \\alpha_{k-1} \\dot F(\\theta_{k-1}),
    ~k \\in \\mathbb{N}. 
```

To select the step size, a line search routine is employed to satisfy the weak
Wolfe conditions. The line search strategy is the Extrapolation-Bisection Line
Search Routine (EBLS) which finds an ``\\alpha_{k-1}`` that satisfies 
two conditions. The first is the sufficient decrease condition

```math
    F(\\theta_k) \\leq F(\\theta_{k-1}) -
        c1 * \\alpha_{k-1} * ||\\dot F(\\theta_{k-1})||_2^2,
```
and the second condition is the curvature condition which enforces that

```math
    \\dot F(\\theta_k)^\\intercal (-\\dot F(\\theta_{k-1})) \\geq 
        -c_2 ||\\dot F(\\theta_{k-1})||_2^2,
```
where ``0 < c_1 < c_2 < 1`` are constant selected by the user.

To see more about the line search routine, see the documentation for
`EBLS!(...)`.

# Arguments

- `optData::WolfeEBLSGD{T}`, specification for the optimization algorithm.
- `progData::P where P <: AbstractNLPModel{T, S}`, specification for the problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 
"""
function wolfe_ebls_gd(
    optData::WolfeEBLSGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem
    precomp, store = initialize(progData)

    # initialization of variables for optimization
    iter::Int64 = 0
    x::S = copy(optData.iter_hist[1])

    # initial gradient information
    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.gkm1 .= store.grad
    optData.grad_val_hist[iter + 1] = norm(store.grad)


    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # update iteration number
        iter += 1

        # get point that satisfies wolfe conditions
        wolfe_condition_satisfied = OptimizationMethods.EBLS!(
            x, 
            optData.iter_hist[iter],
            progData,
            precomp,
            store,
            optData.gkm1,
            optData.grad_val_hist[iter] ^ 2, 
            OptimizationMethods.obj(progData, precomp, store, x),
            optData.α,
            optData.δ,
            optData.c1,
            optData.c2;
            max_iterations = optData.line_search_max_iterations
        )

        # save input
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x
end