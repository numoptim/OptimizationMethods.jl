# OptimizationMethods.jl

"""
    FixedStepGD{T} <: AbstractOptimizerData{T}

A structure for storing data about fixed step-size gradient descent, and the
    progress of its application on an optimization problem.

# Fields

- `name::String`, name of the solver for reference
- `step_size::T`, the step-size selection for the optimization procedure
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int`, the maximum allowed iterations
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `gra_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    correpsonds to the gradient norm at iteration `k`

# Constructors

    FixedStepGD(::Type{T}; x0::Vector{T}, step_size::T, threshold::T, 
        max_iterations::Int) where {T}

Constructs the `struct` for the optimizer.

## Arguments

- `T::DataType`, specific data type for the calculations

## Keyword Arguments

- `x0::Vector{T}`, the initial iterate for the optimizers
- `step_size::T`, the step size of the optimizer 

- `max_iterations::Int`, the maximum number of iterations allowed  
"""
mutable struct FixedStepGD{T} <: AbstractOptimizerData{T}
    name::String
    step_size::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    gra_val_hist::Vector{T}
    stop_iteration::Int64
end
function FixedStepGD(
    ::Type{T};
    x0::Vector{T},
    step_size::T,
    threshold::T,
    max_iterations::Int,
)  where {T}

    d = length(x0)
    iter_hist = Vector{T}[ Vector{T}(undef, d) for i = 1:max_iterations+1]
    iter_hist[1] = x0

    gra_val_hist = Vector{T}(undef, max_iterations+1)
    stop_iteration = -1 #Not stopped

    return FixedStepGD("Gradient Descent with Fixed Step Size",
        step_size, threshold, max_iterations, iter_hist,
        gra_val_hist, stop_iteration)

end

"""
    fixed_step_gd(optData::FixedStepGD{T},progData<:AbstractNLPModel{T,S})
        where {T,S}

Implements fixed step-size gradient descent for the desired optimization problem
    specified by `progData`.

# Method 

The iterates are updated according to the procedure

```math
x_{k+1} = x_k - \\alpha ∇f(x_k),
```

where ``\\alpha`` is the step size, ``f`` is the objective function, and ``∇f`` is the 
    gradient function of ``f``. 

# Arguments 

- `optData::FixedStepGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function fixed_step_gd(
    optData::FixedStepGD{T},
    progData::P where P<:AbstractNLPModel{T,S}
) where {T,S}

    precomp, store = initialize(progData)

    iter = 0
    x = copy(optData.iter_hist[iter+1])
    grad!(progData, precomp, store, x)
    gra_norm = norm(store.grad)

    optData.gra_val_hist[iter+1] = gra_norm

    while (gra_norm > optData.threshold) && (iter < optData.max_iterations)
        iter += 1
        x .-= optData.step_size * store.grad
        grad!(progData, precomp, store, x)
        gra_norm = norm(store.grad)

        optData.iter_hist[iter+1] .= x
        optData.gra_val_hist[iter+1] = norm(store.grad)
    end

    optData.stop_iteration = iter

    return x

end