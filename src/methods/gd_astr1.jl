# Date: 2026/11/07
# Purpose: First-order version (B_k = 0) of the ASTR1 algorithm of Gratton,
# Jerad and Toint, with the Adagrad-type and max-gradient scaling factors.

"""
    ASTR1GD{T} <: AbstractOptimizerData{T}

A structure for storing data about the first-order version (``B_k = 0``) of
    the ASTR1 algorithm of Gratton, Jerad and Toint, and the progress of its
    application on an optimization problem. See [`astr1_gd`](@ref).

# Fields

- `name::String`, name of the solver for reference
- `variant::Symbol`, either `:adagrad` (weights (3.1)) or `:maxg` (weights (4.1))
- `θ::T`, the step-size parameter ``\\theta > 0`` of the weights (the
    experiments use ``\\theta = 1/\\eta``)
- `ς::T`, the constant ``\\varsigma\\in(0,1]`` of the weights (the paper's
    typical value is ``1/100``)
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int`, the maximum allowed iterations
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`.
- `weights::Vector{T}`, the current weight vector ``w_k`` (state; holds the
    weights used for the last step when the solver returns)
- `accumulator::Vector{T}`, per-coordinate state: the running sum
    ``\\sum_{j\\leq k} g_{i,j}^2`` for `:adagrad`, the running maximum
    ``\\max_{j\\leq k}|g_{i,j}|`` for `:maxg`
- `stop_iteration::Int`, iteration number that the algorithm stopped at. 
   Iterate number `stop_iteration` is produced. 

# Constructors

    ASTR1GD(::Type{T}; x0::Vector{T}, variant::Symbol, θ::T, ς::T = T(1/100),
        threshold::T, max_iterations::Int) where {T}

Constructs the `struct` for the optimizer.

## Arguments

- `T::DataType`, specific data type for the calculations

## Keyword Arguments

- `x0::Vector{T}`, the initial iterate for the optimizers
- `variant::Symbol`, `:adagrad` or `:maxg`
- `θ::T`, the step-size parameter of the weights; must be positive
- `ς::T`, the constant ``\\varsigma\\in(0,1]``; default `1/100`
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int`, the maximum number of iterations allowed  
"""
mutable struct ASTR1GD{T} <: AbstractOptimizerData{T}
    name::String
    variant::Symbol
    θ::T
    ς::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    weights::Vector{T}
    accumulator::Vector{T}
    stop_iteration::Int64
end
function ASTR1GD(
    ::Type{T};
    x0::Vector{T},
    variant::Symbol,
    θ::T,
    ς::T = T(1 / 100),
    threshold::T,
    max_iterations::Int,
) where {T}
    @assert variant in (:adagrad, :maxg) "variant must be :adagrad or :maxg."
    @assert θ > 0 "θ must be positive."
    @assert 0 < ς <= 1 "ς must be in (0, 1]."

    d = length(x0)
    iter_hist = Vector{T}[Vector{T}(undef, d) for i = 1:max_iterations+1]
    iter_hist[1] = x0

    grad_val_hist = Vector{T}(undef, max_iterations + 1)
    weights = zeros(T, d)
    accumulator = zeros(T, d)
    stop_iteration = -1 # Not stopped

    return ASTR1GD("ASTR1 Gradient Descent (Gratton-Jerad-Toint, B_k = 0, \
        $(variant))", variant, θ, ς, threshold, max_iterations, iter_hist,
        grad_val_hist, weights, accumulator, stop_iteration)
end

"""
    astr1_adagrad_weights!(w::Vector{T}, accumulator::Vector{T}, g::Vector{T},
        θ::T, ς::T) where {T}

Pure update of the Adagrad-type scaling factors (3.1) of Gratton, Jerad and Toint
    with ``\\mu = 1/2``, ``\\vartheta = 1`` and the upper choice
    ``w_{i,k} = v_{i,k}``:

```math
w_{i,k} = \\theta\\Big(\\varsigma + \\sum_{j=0}^{k} g_{i,j}^2\\Big)^{1/2}.
```

`accumulator` holds ``\\sum_{j<k} g_{i,j}^2`` on entry and ``\\sum_{j\\leq k}
    g_{i,j}^2`` on exit; `g` is ``g_k``; `w` is overwritten with ``w_k``.
"""
function astr1_adagrad_weights!(
    w::Vector{T}, accumulator::Vector{T}, g::Vector{T}, θ::T, ς::T
) where {T}
    @. accumulator += g^2
    @. w = θ * sqrt(ς + accumulator)
    return w
end

"""
    astr1_maxg_weights!(w::Vector{T}, accumulator::Vector{T}, g::Vector{T},
        k::Int, θ::T, ς::T) where {T}

Pure update of the max-gradient scaling factors of Gratton, Jerad and Toint,
    Section 4, eq. (4.1) with ``\\mu = \\nu = 1/10`` and
    ``v_{i,k} = \\max_{j\\leq k}|g_{i,j}|`` (which satisfies (4.2)--(4.3) with
    ``h(k) = 1``), taking the upper end of the interval in (4.1):

```math
w_{i,k} = \\theta\\,(k+1)^{1/10}\\max\\Big\\{\\varsigma,\\ 
    \\max_{j\\leq k}|g_{i,j}|\\Big\\}.
```

`accumulator` holds ``\\max_{j<k}|g_{i,j}|`` on entry and
    ``\\max_{j\\leq k}|g_{i,j}|`` on exit; `g` is ``g_k``; `k` is the iteration
    index (from `0`); `w` is overwritten with ``w_k``.
"""
function astr1_maxg_weights!(
    w::Vector{T}, accumulator::Vector{T}, g::Vector{T}, k::Int, θ::T, ς::T
) where {T}
    @. accumulator = max(accumulator, abs(g))
    scale = θ * T(k + 1)^T(1 / 10)
    @. w = scale * max(ς, accumulator)
    return w
end

"""
    astr1_gd(optData::ASTR1GD{T}, progData::P where P<:AbstractNLPModel{T,S})
        where {T,S}

Implements the first-order version (``B_k = 0``) of Algorithm 2.1 (ASTR1) of
    Gratton, Jerad and Toint, *Complexity of a class of first-order
    objective-function-free optimization algorithms*, arXiv:2203.01647
    (v3, June 2023), for the optimization problem specified by `progData`.

# Method

Given ``g_k = \\nabla f(x_k)`` and weights ``w_k``, the trust region (2.2) is
    the box

```math
\\Delta_{i,k} = \\frac{|g_{i,k}|}{w_{i,k}}, \\qquad |s_{i,k}| \\leq \\Delta_{i,k}
\\quad (i = 1,\\ldots,n).
```

With ``B_k = 0`` the model is linear, ``\\gamma_k = 1`` in (2.8) and the
    generalized Cauchy point (2.6)--(2.7) is
    ``s_{i,k} = -\\mathrm{sign}(g_{i,k})\\Delta_{i,k} = -g_{i,k}/w_{i,k}``,
    which satisfies (2.5) with equality (any ``\\tau\\in(0,1]``). The next
    iterate is ``x_{k+1} = x_k + s_k`` (2.9). Two weight choices are used, with
    ``\\theta`` the paper's step-size parameter (footnote 4 of Section 3;
    in the experiments ``\\theta = 1/\\eta``) and ``\\varsigma = 1/100``:

- `:adagrad`, see [`astr1_adagrad_weights!`](@ref) (eq. (3.1), ``\\mu = 1/2``):
    ``w_{i,k} = \\theta(\\varsigma + \\sum_{j\\leq k} g_{i,j}^2)^{1/2}``,
- `:maxg`, see [`astr1_maxg_weights!`](@ref) (eq. (4.1), ``\\mu=\\nu=1/10``):
    ``w_{i,k} = \\theta(k+1)^{1/10}\\max\\{\\varsigma, \\max_{j\\leq k}|g_{i,j}|\\}``.

The algorithm as stated has no termination rule; following comment 4 after
    Algorithm 2.1 we stop when ``\\Vert g_k\\Vert \\leq \\epsilon``
    (`optData.threshold`).

!!! note
    Each iteration uses exactly one gradient evaluation and no objective
    evaluation.

# Arguments

- `optData::ASTR1GD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 

# Returns

- `x::S`, the final iterate.
"""
function astr1_gd(
    optData::ASTR1GD{T},
    progData::P where P<:AbstractNLPModel{T,S}
) where {T,S}

    # initialize storage and pre-computed values
    precomp, store = initialize(progData)

    iter = 0
    x = copy(optData.iter_hist[iter + 1])
    grad!(progData, precomp, store, x)
    gra_norm = norm(store.grad)
    optData.grad_val_hist[iter + 1] = gra_norm
    fill!(optData.accumulator, T(0))
    record_iterate!(progData, iter, x, gra_norm)

    while (gra_norm > optData.threshold) && (iter < optData.max_iterations)

        # weights w_k from g_0, ..., g_k
        if optData.variant == :adagrad
            astr1_adagrad_weights!(optData.weights, optData.accumulator,
                store.grad, optData.θ, optData.ς)
        else
            astr1_maxg_weights!(optData.weights, optData.accumulator,
                store.grad, iter, optData.θ, optData.ς)
        end

        # s_{i,k} = -g_{i,k} / w_{i,k}, |s_{i,k}| = Δ_{i,k}
        x .-= store.grad ./ optData.weights
        iter += 1

        grad!(progData, precomp, store, x)
        gra_norm = norm(store.grad)

        optData.iter_hist[iter + 1] .= x
        optData.grad_val_hist[iter + 1] = gra_norm
        record_iterate!(progData, iter, x, gra_norm)
    end

    optData.stop_iteration = iter

    return x
end
