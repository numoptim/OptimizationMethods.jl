# Date: 12/16/2024
# Author: Christian Varner
# Purpose: Implementation of diminishing step size
# gradient descent

"""
    DiminishingStepGD{T} <: AbstractOptimizerData{T}

A structure for storing data about gradient descent using diminishing step sizes,
    and recording the progress of its application on an optimization problem.

# Fields

- `name::String`, name of the solver for reference.
- `step_size_function::Function`, step size function. Should take the iteration 
    number and return the step size for that iteration.
- `step_size_scaling::T`, factor that is multipled with the amount of the step 
    size function.
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

!!! warning
    `step_size_function` should take in two arguments, the data type for
    computation `T` and the iteration number. For example, calling
    `step_size_function(Float64, 1)` should return the step size
    as a `Float64` for the iteration `1`.

# Constructors

    DiminishingStepGD(::Type{T}; x0::Vector{T}, step_size_function::Function,
        step_size_scaling::T, threshold::T, max_iterations::Int)

Constructs the `struct` for the diminishing step size gradient descent method.

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `step_size_function::Function`, step size function. Should take the iteration 
    number and return the step size for that iteration.
- `step_size_scaling::T`, factor that is multipled with the amount of the step 
    size function.
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct DiminishingStepGD{T} <: AbstractOptimizerData{T}
    name::String
    step_size_function::Function
    step_size_scaling::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function DiminishingStepGD(
    ::Type{T};
    x0::Vector{T},
    step_size_function::Function,
    step_size_scaling::T,
    threshold::T,
    max_iterations::Int,
) where {T}
    
    # error checking
    
    # initialization histories
    d = length(x0)
    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:(max_iterations + 1)]
    iter_hist[1] .= x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    # instantiate and return the struct
    return DiminishingStepGD("Gradient Descent with Diminishing Step Size",
                                step_size_function, step_size_scaling,
                                threshold, max_iterations, iter_hist, 
                                grad_val_hist, stop_iteration)
end

"""
    diminishing_step_gd(optData::DiminishingStepGD{T}, progData::P
        where P <: AbstractNLPModel{T, S}) where {T, S}

Implements gradient descent with diminishing step sizes and applies the method
    to the optimization problem specified by `progData`.

# Reference(s)

[Patel, Vivak, and Albert Berahas. “Gradient Descent in the Absence of Global 
    Lipschitz Continuity of the Gradients.” SIAM 6 (3): 579–846. 
    https://doi.org/10.1137/22M1527210.](@cite patel2024Gradient)

[Bertsekas, Dimitri. "Nonlinear Programming". 3rd Edition, Athena Scientific, 
    Chapter 1.](@cite bertsekas2016Nonlinear)

# Method

Given iterates ``\\lbrace x_0,\\ldots,x_k\\rbrace``, the iterate ``x_{k + 1}``
    is equal to ``x_k - \\alpha_k \\nabla f(x_k)``, where ``\\alpha_k`` is equal
    to `optData.step_size_scaling * optData.step_size_function(T, k)`.

!!! info "Step Size Function"
    The step size function should satisfy several conditions.
    First, ``\\alpha_k > 0`` for all ``k``.
    Second, ``\\lim_{k \\to \\infty} \\alpha_k = 0.``
    Finally, ``\\sum_{k=0}^{\\infty} \\alpha_k = \\infty.``
    See [Patel and Berahas (2024)](@cite patel2024Gradient) for details.

# Arguments

- `optData::DiminishingStepGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function diminishing_step_gd(
    optData::DiminishingStepGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}
    # initialization of storage and pre-computed values
    precomp, store = OptimizationMethods.initialize(progData)
    
    # update histories
    iter = 0
    x = copy(optData.iter_hist[iter + 1])
    grad!(progData, precomp, store, x)
    optData.grad_val_hist[iter + 1] = norm(store.grad)
    
    # Compute initial step size 
    α = optData.step_size_scaling*optData.step_size_function(T, iter)

    # do gradient descent with diminishing step size
    while (iter < optData.max_iterations) &&
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        iter += 1

        # take a step
        x .-= α .* store.grad
        grad!(progData, precomp, store, x)
        α = optData.step_size_scaling*optData.step_size_function(T, iter)

        # update histories
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad)
    end
    
    optData.stop_iteration = iter

    return x
end
