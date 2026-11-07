# Date: 2026/11/07
# Purpose: First-order version (B_k = 0) of the adaptive trust-region method
# without function evaluations of Grapiglia and Stella (2022).

"""
    AdaptiveTrustRegionGD{T} <: AbstractOptimizerData{T}

A structure for storing data about the first-order adaptive trust-region method
    of Grapiglia and Stella (Algorithm 1 with ``B_k = 0``), and the progress of
    its application on an optimization problem. See [`adaptive_trust_region_gd`](@ref).

# Fields

- `name::String`, name of the solver for reference
- `init_scaling::T`, the initial scaling ``b_0 \\geq b_{\\min}`` (``\\delta_0 = 1/b_0``)
- `α::T`, the parameter ``\\alpha\\in[0,1)`` of update (2.4); `α = 0` is the
    conservative variant (AdaTrust1), `α = 0.9` the flexible one (AdaTrust2)
- `b_min::T`, the lower bound ``b_{\\min} > 0`` in (2.5)
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int`, the maximum allowed iterations
- `iter_hist::Vector{Vector{T}}`, a history of the iterates. The first entry
    corresponds to the initial iterate (i.e., at iteration `0`). The `k+1` entry
    corresponds to the iterate at iteration `k`.
- `grad_val_hist::Vector{T}`, a vector for storing `max_iterations+1` gradient
    norm values. The first entry corresponds to iteration `0`.
- `scaling_hist::Vector{T}`, a vector for storing `max_iterations+1` values of
    ``b_k`` (the scaling used to compute the step from iterate `k`; the entry
    for the stopping iterate is the scaling that would have been used next).
- `stop_iteration::Int`, iteration number that the algorithm stopped at. 
   Iterate number `stop_iteration` is produced. 

# Constructors

    AdaptiveTrustRegionGD(::Type{T}; x0::Vector{T}, init_scaling::T, α::T,
        b_min::T, threshold::T, max_iterations::Int) where {T}

Constructs the `struct` for the optimizer. Requires `init_scaling >= b_min > 0`
    and `0 <= α < 1`.

## Arguments

- `T::DataType`, specific data type for the calculations

## Keyword Arguments

- `x0::Vector{T}`, the initial iterate for the optimizers
- `init_scaling::T`, the initial ``b_0``. In the experiments ``b_0 = 1/\\eta``
    for the initial-step grid ``\\eta``.
- `α::T`, the parameter ``\\alpha`` of (2.4)
- `b_min::T`, the lower bound ``b_{\\min}``. The paper uses ``10^{-4}``.
- `threshold::T`, the threshold on the norm of the gradient to induce stopping
- `max_iterations::Int`, the maximum number of iterations allowed  
"""
mutable struct AdaptiveTrustRegionGD{T} <: AbstractOptimizerData{T}
    name::String
    init_scaling::T
    α::T
    b_min::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    scaling_hist::Vector{T}
    stop_iteration::Int64
end
function AdaptiveTrustRegionGD(
    ::Type{T};
    x0::Vector{T},
    init_scaling::T,
    α::T,
    b_min::T,
    threshold::T,
    max_iterations::Int,
) where {T}
    error("not implemented")
end

"""
    adaptive_trust_region_update(b::T, ω::T, gnorm_next::T, α::T, b_min::T,
        b_hat_max::T) where {T}

Pure update rule (2.4)--(2.5) of Grapiglia and Stella for the pair
    ``(b_k, \\omega_k)``, specialised to the first-order step
    ``d_k = -\\nabla f(x_k)/b_k`` (for which ``\\Vert d_k\\Vert = \\Delta_k``, so the
    "otherwise" branch of the flexible rule applies). Given `b` ``= b_k``,
    `ω` ``= \\omega_k`` and `gnorm_next` ``= \\Vert\\nabla f(x_{k+1})\\Vert``, it
    returns `(b_next, ω_next)`.

```math
(b_{k+1}, \\omega_{k+1}) = \\begin{cases}
    (\\hat b_k, \\Vert\\nabla f(x_{k+1})\\Vert) & \\text{if } 
        \\Vert\\nabla f(x_{k+1})\\Vert \\leq \\alpha\\omega_k, \\\\
    (b_k + \\Vert\\nabla f(x_{k+1})\\Vert^2/b_k, \\omega_k) & \\text{otherwise,}
\\end{cases}
```

where ``\\hat b_k\\in[b_{\\min}, b_k]`` is, for the flexible variant (Section 3,
    AdaTrust2) with ``\\Vert d_k\\Vert > \\Delta_k/2``,

```math
\\hat b_k = \\min\\{\\hat b_{\\max}, \\max\\{b_{\\min}, b_k/2\\}\\},
```

and `b_hat_max` ``= \\hat b_{\\max}``. To guarantee (2.5) when
    ``\\hat b_{\\max} < b_{\\min}`` (possible only if the initial gradient is
    tiny), the result is additionally clamped to ``[b_{\\min}, b_k]``.
"""
function adaptive_trust_region_update(
    b::T, ω::T, gnorm_next::T, α::T, b_min::T, b_hat_max::T
) where {T}
    error("not implemented")
end

"""
    adaptive_trust_region_gd(optData::AdaptiveTrustRegionGD{T},
        progData::P where P<:AbstractNLPModel{T,S}) where {T,S}

Implements the first-order version of the adaptive trust-region method of
    Grapiglia and Stella, *An adaptive trust-region method without function
    evaluations*, Comput. Optim. Appl. 82:31--60, 2022
    (doi:10.1007/s10589-022-00356-0), Algorithm 1 with ``B_k = 0``, for the
    optimization problem specified by `progData`.

# Method

At iteration ``k`` the trust-region subproblem (2.1)--(2.2),

```math
\\min_{d}\\ m_k(d) = \\nabla f(x_k)^\\intercal d + \\tfrac12 d^\\intercal B_k d
\\quad\\text{s.t.}\\quad \\Vert d\\Vert\\leq\\Delta_k = \\delta_k\\Vert\\nabla f(x_k)\\Vert,
\\qquad \\delta_k = 1/b_k,
```

is solved with ``B_k = 0``, so that ``m_k`` is linear and the solution is on the
boundary: 

```math
d_k = -\\delta_k\\nabla f(x_k), \\qquad \\Vert d_k\\Vert = \\Delta_k, \\qquad
x_{k+1} = x_k + d_k.
```

(Condition (2.3) holds with equality in the first term.) Set
    ``\\omega_0 = \\Vert\\nabla f(x_0)\\Vert``. The pair ``(b_k,\\omega_k)`` is
    updated by (2.4)--(2.5), see [`adaptive_trust_region_update`](@ref):

```math
(b_{k+1}, \\omega_{k+1}) = \\begin{cases}
    (\\hat b_k, \\Vert\\nabla f(x_{k+1})\\Vert) & \\text{if } 
        \\Vert\\nabla f(x_{k+1})\\Vert \\leq \\alpha\\omega_k, \\\\
    (b_k + \\Vert\\nabla f(x_{k+1})\\Vert^2/b_k, \\omega_k) & \\text{otherwise,}
\\end{cases}
\\qquad \\hat b_k\\in[b_{\\min}, b_k].
```

We use the choice of Section 3 (AdaTrust2) with
    ``\\hat b_k = \\min\\{\\hat b_{\\max}, \\max\\{b_{\\min}, b_k/2\\}\\}``
    (the case ``\\Vert d_k\\Vert > \\Delta_k/2`` always holds), 
    ``\\hat b_{\\max} = \\Vert\\nabla f(x_0)\\Vert`` and ``b_{\\min} = 10^{-4}``.
With ``\\alpha = 0`` the first branch never fires (unless the gradient vanishes),
    ``\\delta_k`` is non-increasing, and the update is the (batch) WNGrad rule
    (AdaTrust1).

!!! note
    The paper's own experiments use BFGS matrices ``B_k`` with Steihaug--Toint
    subproblem solves. Here we use ``B_k = 0`` so that every compared method
    is first-order.

!!! note
    Each iteration uses exactly one gradient evaluation and no objective
    evaluation.

# Arguments

- `optData::AdaptiveTrustRegionGD{T}`, the specification for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem. 

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument. 

# Returns

- `x::S`, the final iterate.
"""
function adaptive_trust_region_gd(
    optData::AdaptiveTrustRegionGD{T},
    progData::P where P<:AbstractNLPModel{T,S}
) where {T,S}
    error("not implemented")
end
