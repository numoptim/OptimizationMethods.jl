# Date: 01/23/2025
# Author: Christian Varner
# Purpose: Implementation of a simple backtracking algorithm

"""
    backtracking!(θk::S, θkm1::S, gkm1::S, step_direction::S,
        reference_value::T, α::T, δ::T, ρ::T; 
        max_iteration::Int64 = 100)

Implementation of backtracking which modifies `θk` in place.

# Reference(s)

# Method

Let ``\\theta_{k-1}`` be the current iterate, and let 
``\\alpha \\in \\mathbb{R}_{>0}``, ``\\delta \\in (0, 1)``, and
``\\rho \\in (0, 1)``. Let ``d_k`` be the step direction, then
``\\theta_k = \\theta_{k-1} - \\delta^t\\alpha d_k`` where 
``t + 1 \\in \\mathbb{N}`` is the smallest such number satisfying

```math
    F(\\theta_k) \\leq O_{k-1} - \\rho\\delta^t\\alpha
    \\dot F(\\theta_{k-1})^\\intercal d_k,
```

where ``O_{k-1}`` is some reference value. 

# Arguments

- `θk::S`, buffer array for the next iterate.
- `θkm1::S`, current iterate the optimization algorithm.
- `gkm1::S`, gradient value at `θkm1`.
- `step_direction::S`, direction to move `θkm1` to form `θk`.
- `reference_value::T`, value to check the objective value
    at `θk` against.
- `α::T`, initial step size.
- `δ::T`, backtracking decrease factor.
- `ρ::T`, factor involved in the acceptance criterion in `θk`.
    Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions on `θk`.

## Optional Keyword Arguments

- `max_iteration::Int64 = 100`, the maximum allowable iterations
    for the line search procedure. In the language above, when
    ``t`` is equal to `max_iteration` the algorithm will terminate.

"""
function backtracking!(
    θk::S,
    θkm1::S,
    gkm1::S,
    step_direction::S,
    reference_value::T,
    α::T,
    δ::T,
    ρ::T;
    max_iteration::Int64 = 100
) where {T, S}

    # initial step
    t = 0
    inner_prod_grad_by_direction = dot(gkm1, step_direction)
    θk .= θkm1 - (δ^t * α) .* step_direction

    # backtracking
    while (t < max_iteration) &&
        (F(θk) > reference_value - ρ * (δ^t * α) * inner_prod_grad_by_direction)

        t += 1
        θk .= θkm1 - (δ^t * α) .* step_direction
    end
end