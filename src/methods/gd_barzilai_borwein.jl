# OptimizationMethods.jl

"""
    BarzilaiBorweinGD{T} <: AbstractOptimizerData{T}

A structure for storing data about gradient descent using the Barzilai-Borwein 
    step size, and the progress of its application on an optimization problem.

# Fields

- `name:String`, name of the solver for reference.
- `init_stepsize::T`, initial step size to start the method. 
- `long_stepsize::Bool`, flag for step size; if true, use the long version of 
    Barzilai-Borwein. If false, use the short version. 
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
- `iter_diff::Vector{T}`, a buffer for storing differences between subsequent
    iterate values that are used for computing the step size
- `grad_diff::Vector{T}`, a buffer for storing differences between gradient 
    values at adjacent iterates, which is used to compute the step size
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`. The `k+1` entry
    correpsonds to the gradient norm at iteration `k`.
- `stop_iteration::Int64`, the iteration number that the solver stopped on.
    The terminal iterate is saved at `iter_hist[stop_iteration+1]`.

# Constructors

    BarzilaiBorweinGD(::Type{T}; x0::Vector{T}, init_stepsize::T, 
        long_stepsize::Bool, threshold::T, max_iterations::Int) where {T}

Constructs the `struct` for the Barzilai-Borwein optimization method

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the solver at.
- `init_stepsize::T`, initial step size used for the first iteration. 
- `long_stepsize::Bool`, flag for step size; if true, use the long version of
    Barzilai-Borwein, if false, use the short version. 
- `threshold::T`, gradient threshold. If the norm gradient is below this, 
    then iteration is terminated. 
- `max_iterations::Int`, max number of iterations (gradient steps) taken by 
    the solver.

"""
mutable struct BarzilaiBorweinGD{T} <: AbstractOptimizerData{T}
    name::String
    init_stepsize::T 
    long_stepsize::Bool 
    threshold::T
    max_iterations::Int64
    iter_diff::Vector{T}
    grad_diff::Vector{T}
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function BarzilaiBorweinGD(
    ::Type{T};
    x0::Vector{T},
    init_stepsize::T, 
    long_stepsize::Bool, 
    threshold::T,
    max_iterations::Int,
) where {T}

    @assert init_stepsize > 0 "Initial step size must be a positive value."

    d = length(x0)

    iter_diff = zeros(T, d)
    grad_diff = zeros(T, d)

    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:max_iterations + 1]
    iter_hist[1] = x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    return BarzilaiBorweinGD("Gradient Descent with Barzilai-Borwein Step Size", 
        init_stepsize, long_stepsize, threshold, max_iterations, 
        iter_diff, grad_diff, iter_hist, grad_val_hist, stop_iteration)
end

# Utility Functions 
"""
    bb_long_step_size(Δx::S, Δg::S) where S <: AbstractVector 

Computes the long step size for the Borwein-Barzilai Method. See documentation
    for `barzilai_borwein_gd` for more details. 
"""
function bb_long_step_size(Δx::S, Δg::S) where S <: AbstractVector
    return (Δx' * Δx) / (Δx' * Δg)
end

"""
    bb_short_step_size(Δx::S, Δg::S) where S <: AbstractVector 

Computes the short step size for the Borwein-Barzilai Method. See documentation
    for `barzilai_borwein_gd` for more details. 
"""
function bb_short_step_size(Δx::S, Δg::S) where S <: AbstractVector
    return (Δx' * Δg) / (Δg' * Δg)
end


"""

    barzilai_borwein_gd(optData::BarzilaiBorweinGD{T}, progData::P 
        where P <: AbstractNLPModel{T, S}) where {T,S} 

Implements gradient descent with Barzilai-Borwein step size and applies the 
    method to the optimization problem specified by `progData`. 

# Reference(s)

Barzilai and Borwein. "Two-Point Step Size Gradient Methods". IMA Journal of 
    Numerical Analysis.

# Method

Given iterates ``\\lbrace x_0,\\ldots,x_k\\rbrace``, the iterate ``x_{k+1}``
    is equal to ``x_k - \\alpha_k \\nabla f(x_k)``, where ``\\alpha_k`` is
    one of two versions.

## Long Step Size Version (if `optData.long_stepsize==true`)

If ``k=0``, then ``\\alpha_0`` is set to `optData.init_stepsize`. For ``k>0``,

```math 
\\alpha_k = \\frac{ \\Vert x_k - x_{k-1} \\Vert_2^2}{(x_k - x_{k-1})^\\intercal 
    (\\nabla f(x_k) - \\nabla f(x_{k-1}))}.
```

## Short Step Size Version (if `optData.long_stepsize==false`)

If ``k=0``, then ``\\alpha_0`` is set to `optData.init_stepsize`. For ``k>0``,

```math
\\alpha_k = \\frac{(x_k - x_{k-1})^\\intercal (\\nabla f(x_k) - 
    \\nabla f(x_{k-1}))}{\\Vert \\nabla f(x_k) - \\nabla f(x_{k-1})\\Vert_2^2}.
```

# Arguments

- `optData::BarzilaiBorweinGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.
"""
function barzilai_borwein_gd(
    optData::BarzilaiBorweinGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T,S} 

    # initialize storage and pre-computed values
    precomp, store = OptimizationMethods.initialize(progData) 

    # select step size strategy 
    bb_step_size = optData.long_stepsize ? bb_long_step_size : bb_short_step_size
    
    # Iteration 0
    iter = 0

    # Update iteration 
    x = copy(optData.iter_hist[iter + 1]) 
    grad!(progData, precomp, store, x)

    # Compute step size to calculate x_{iter+1}
    step_size = optData.init_stepsize

    # Store Values
    optData.grad_val_hist[iter + 1] = norm(store.grad)

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        iter += 1

        # Needed for calculate step size for updating x_{iter} to  x_{iter+1}
        optData.iter_diff .= -x #Δx = -x_{iter-1}
        optData.grad_diff .= -store.grad #Δg = -∇f(x_{iter-1})
        
        # Update iteration 
        x .-= step_size * store.grad
        grad!(progData, precomp, store, x)

        # Compute step size to calculate x_{iter+1}
        optData.iter_diff .+= x #Δx = x_{iter} - x_{iter-1}
        optData.grad_diff .+= store.grad #Δg = ∇f(x_{iter}) - ∇f(x_{iter-1})
        step_size = bb_step_size(optData.iter_diff, optData.grad_diff)

        # Store values
        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = norm(store.grad) 
    end

    optData.stop_iteration = iter

    return x
end