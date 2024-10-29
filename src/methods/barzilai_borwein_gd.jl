# Date: 09/16/2024
# Author: Christian Varner
# Purpose: Implement barzilai-borwein.

"""
"""
mutable struct BarzilaiBorweinGD{T} <: AbstractOptimizerData{T}
    name::String
    alfa0::T
    long::Bool
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    gra_val_hist::Vector{T}
    stop_iteration::Int64
end
function BarzilaiBorweinGD(
    ::Type{T};
    x0::Vector{T},
    alfa0::T,
    long::Bool,
    threshold::T,
    max_iterations::Int,
) where {T}

    d = length(x0)

    iter_hist = Vector{T}[Vector{T}(undef, d) for i in 1:max_iterations + 1]
    iter_hist[1] = x0

    gra_val_hist = Vector{T}(undef, max_iterations + 1)
    stop_iteration = -1

    return BarzilaiBorweinGD("Gradient Descent with Barzilai-Borwein", alfa0, long, threshold, max_iterations, iter_hist, gra_val_hist,
    stop_iteration)
end

"""
    (x, stats) = barzilai_borwein_gd(progData, x, max_iter; alfa0, long)

Implementation of barzilai-borwein step size method using negative gradient
directions. To see more about the method, take a look at:

Barzilai and Borwein. "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis.

The method will take advantage of precomputed values and allocated space initialized 
by calling `initialize(progData)` (see documentation for problems).

## Arguments

- `progData::AbstractNLPModel{T, S}`, function to optimize
- `x::S`, initial starting value
- `max_iter::Int64`, max iteration limit
- `gradient_condition`, if positive, the algorithm stops once the gradient is less than or equal to `gradient_condition`. If negative the condition is not checked.
- `alfa0::T = 1e-4` (Optional), initial step size
- `long::Bool = true` (Optional), flag to indicate the use of the long version or the short version

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
    step_size = optData.long ? _long_step_size : _short_step_size
    x = copy(optData.iter_hist[iter + 1]) 

    # buffer for previous gradient value
    gprev :: S  = zeros(T, size(x))

    # save initial values 
    OptimizationMethods.grad!(progData, precomp, store, x)
    optData.gra_val_hist[iter + 1] = norm(store.grad)

    # first step
    iter += 1
    x .-= optData.alfa0 .* store.grad
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