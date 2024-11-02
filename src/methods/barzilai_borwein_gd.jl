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
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `gra_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
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
    iter_hist::Vector{Vector{T}}
    gra_val_hist::Vector{T}
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

    d = length(x0)

    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:max_iterations + 1]
    iter_hist[1] = x0

    gra_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    return BarzilaiBorweinGD("Gradient Descent with Barzilai-Borwein Step Size", 
        init_stepsize, long_stepsize, threshold, max_iterations, 
        iter_hist, gra_val_hist, stop_iteration)
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
\\alpha_k = \\frac{ \\Vert x_k - x_{k-1} \\Vert_2^2}{(x_k - x_{k-1}^\\intercal 
    (\\nabla f(x_k) - \\nabla f(x_{k-1})))}.
```

## Short Step Size Version (if `optData.long_stepsize==false`)

If ``k=0``, then ``\\alpha_0`` is set to `optData.init_stepsize`. For ``k>0``,

```math
\\alpha_k = \\frac{(x_k - x_{k-1}^\\intercal (\\nabla f(x_k) - 
    \\nabla f(x_{k-1})))}{\\Vert \\nabla f(x_k) - \\nabla f(x_{k-1})\\Vert_2^2}.
```

# Arguments

- `optData::FixedStepGD{T}`, the specification for the optimization method.
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

    # step size helper functions -- long variant of step size
    function _long_step_size(Δx::S, Δg::S)
        return (Δx' * Δx) / (Δx' * Δg)
    end

    # step size helper function -- short variant of step size
    function _short_step_size(Δx::S, Δg::S)
        return (Δx' * Δg) / (Δg' * Δg)
    end

    # initialization
    iter = 0
    precomp, store = OptimizationMethods.initialize(progData) 
    step_size = optData.long_stepsize ? _long_step_size : _short_step_size
    x = copy(optData.iter_hist[iter + 1]) 

    # buffer for previous gradient value
    gprev::S = zeros(T, size(x))

    # save initial values 
    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.gra_val_hist[iter + 1] = norm(store.grad)

    # first step
    iter += 1
    x .-= optData.init_stepsize .* store.grad
    gprev .= store.grad

    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.iter_hist[iter + 1] .= x
    optData.gra_val_hist[iter + 1] = norm(store.grad)

    # main iteration
    while (iter < optData.max_iterations) && (optData.gra_val_hist[iter + 1] > optData.threshold)
        iter += 1

        # compute Δx and Δg for the step size using memory already allocated
        optData.iter_hist[iter + 1] .= x - optData.iter_hist[iter - 1]
        gprev .*= -1
        gprev .+= store.grad # store.grad - gprev

        # update
        x .-= step_size(optData.iter_hist[iter + 1], gprev) .* store.grad 
        gprev .= store.grad

        # save values
        OptimizationMethods.grad!(progData, precomp, store, x)
        optData.iter_hist[iter + 1] .= x
        optData.gra_val_hist[iter + 1] = norm(store.grad) 
    end
    optData.stop_iteration = iter

    return x
end