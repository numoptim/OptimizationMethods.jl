# Date: 01/17/2025
# Author: Christian Varner
# Purpose: Implementation of the proximal point method
# which is a special case of the bregman distance method.

"""
    ProximalPointGD{T} <: AbstractOptimizerData{T}
    
A mutable struct that represents gradient descent using the proximal
    point algorithm. It stores the specification for the method and records
    values during iteration.

# Fields

- `name::String`, name of the optimizer for recording purposes. 
- `distance_penalty::T`, penalty applied to the distance function.
- `subproblem_solver::Function`, subproblem solver used to solve the
    proximal problem. Should take in a single argument that is a subtype
    of `AbstractOptimizerData{T}` and return a (local/approximate) solution to 
    the problem.
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

!!! warning
    The algorithm currently takes the solution of `subproblem_solver` without
    checking gradient tolerance conditions. 

# Constructors

    ProximalPointGD(::Type{T}; distance_penalty::T,
        subproblem_solver_struct::P where P <: AbstractOptimizerData{T},
        subproblem_solver_function::Function, threshold::T, max_iterations::Int64
    ) where {T}

Constructs an instance of type `ProximalPointGD{T}`.

## Arguments

- `T::DataType`, type for data and computation

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `distance_penalty::T`, penalty applied to the distance function
- `subproblem_solver_struct::P where P <: AbstractOptimizerData{T}`, struct
    that specifies the inner solver optimization algorithm.
- `subproblem_solver_function::Function`, method that implements the 
    optimization algorithm corresponding to `subproblem_solver_struct`.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
"""
mutable struct ProximalPointGD{T} <: AbstractOptimizerData{T}
    name::String
    distance_penalty::T
    subproblem_solver::Function
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function ProximalPointGD(
    ::Type{T};
    x0::Vector{T},
    distance_penalty::T,
    subproblem_solver_struct::P where P <: AbstractOptimizerData{T},
    subproblem_solver_function::Function,
    threshold::T,
    max_iterations::Int64
) where {T}

    # create subproblem solver
    subproblem_solver(progData) =
        subproblem_solver_function(subproblem_solver_struct, progData)

    # initialize iter_hist and grad_val_hist
    d::Int64 = length(x0)
    iter_hist::Vector{Vector{T}} = 
        Vector{Vector{T}}([Vector{T}(undef, d) for i in 1:(max_iterations + 1)])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1) 
    stop_iteration::Int64 = -1 ## dummy value

    return ProximalPointGD("Proximal Point Gradient Descent", distance_penalty,
        subproblem_solver, threshold, max_iterations, iter_hist, grad_val_hist,
        stop_iteration
    )
end

"""
    proximal_point_gd(optData::ProximalPointGD, 
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Method that implements the proximal point gradient descent algorithm using the
    specification in `optData` on the problem specified by `progData`.

# Reference(s)

[Bauschke et. al. "A Descent Lemma Beyond Lipschitz Gradient Continuity:
    First-Order Methods Revisited and Applications."
    Mathematics of Operations Research.](@cite bauschke2017a)

# Method

Let ``\\theta_k`` be the ``k^{th}`` iterate, and let ``\\lambda`` be the
penalty parameter on the distance function.
Then, ``\\theta_{k+1}`` is generated as the following

```math
    \\theta_{k + 1} = \\arg\\min_{\\theta} F(\\theta) + \\frac{\\lambda}{2}
    ||\\theta - \\theta_{k}||_2^2,
```
where ``||\\cdot||_2^2`` is the L2-norm.

!!! note
    This is a special case of the more general Bregman Distance method
    proposed in 
    ["A Descent Lemma Beyond Lipschitz Gradient Continuity"](@cite bauschke2017a) 
    with the L2-norm as the penalty function.

# Arguments

- `optData::WeightedNormDampingGD{T}`, specification for the optimization algorithm.
- `progData::P where P <: AbstractNLPModel{T, S}`, specification for the problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 
"""
function proximal_point_gd(
    optData::ProximalPointGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize for main problem
    precomp, store = initialize(progData)

    # initial iteration
    iter::Int64 = 0

    x::S = copy(optData.iter_hist[1])
    penalty::T = optData.distance_penalty

    # initialize for subproblem
    subproblem_prog_data = ProximalPointSubproblem(T; 
        progData = progData, progData_precomp = precomp, progData_store = store, 
        penalty = penalty, θkm1 = x)

    # store initial gradient
    grad!(progData, precomp, store, x)
    optData.grad_val_hist[1] = norm(store.grad)

    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration
        iter += 1

        # solve inner problem
        x .= optData.subproblem_solver(subproblem_prog_data)

        # store values
        grad!(progData, precomp, store, x)
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
        
        # get ready for next iteration
        subproblem_prog_data.θkm1 .= x
    end
    
    optData.stop_iteration = iter

    return x
end