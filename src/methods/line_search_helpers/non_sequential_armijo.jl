# Date: 02/13/2025
# Author: Christian Varner
# Purpose: Implement the non-sequential armijo descent check

"""
    non_sequential_armijo_condition(F_ψjk::T, reference_value::T,
        norm_grad_θk::T, ρ::T, δk::T, α0k::T) where {T}

Check if `F_ψjk` satisfies the non-sequential armijo condition with respect
to `reference_value` and the remaining parameters. Returns a boolean value
indicating if the descent condition is satisfied or not.

# Method

Let ``F(\\theta) : \\mathbb{R}^n \\to \\mathbb{R}`` be the objective function.
Suppose that ``\\theta_k \\in \\mathbb{R}^n`` is an iterate of an optimization
routine. Let ``\\psi_0^k = \\theta_k`` and for ``j \\in \\lbrace 1,...,T \\rbrace``
for some ``T \\in \\mathbb{N}``, recursively define

```math
    \\psi_j^k = \\psi_0^k - \\sum_{t = 0}^{j-1} \\delta_k \\alpha_t^k \\dot F(\\psi_t^k).
```

Then, the (monotone) non-sequential armijo condition requires that

```math
    F(\\psi_j^k) < F(\\psi_0^k) - \\rho \\delta_k \\alpha_0^k ||\\dot F(\\psi_0^k)||_2^2,
```
where ``||\\cdot||_2`` is the L2-norm and ``\\rho \\in (0, 1)``.

This function implements checking the inequality, where `F_ψjk` corresponds to
    ``F(\\psi_j^k)``, `reference_value` corresponds to ``F(\\psi_0^k)``,
    `norm_grad_θk` to ``||\\dot F(\\psi_0^k)||_2``, `ρ` to ``\\rho``, 
    `δk` to ``\\delta_k``, and `α0k` to ``\\alpha_0^k``. To see more 
    about how this method is used read the documentation for 
    [gradient descent with non-sequential armijo](@ref NonsequentialArmijoGD)

# Arguments

- `F_ψjk::T`, numeric value on the LHS of the inequality. In optimization context,
    this is the objective value of a trial iterate to check if sufficient descent
    is achieved.
- `reference_value::T`, numeric value on the RHS of the inequality. In the
    optimization context, the value of the current iterate `F_ψjk` must be
    smaller than this to guarantee a sufficient descent criterion.
- `norm_grad_θk::T`, numeric value forming the amount of descent that needs
    to be achieved. This value is usually the norm of the gradient of a 
    previous iterate.
- `ρ::T`, parameter in the line search criterion dictating how much descent
    should be required. Should be positive. Larger values indicate stricter
    conditions, and lower value indicate looser conditions.
- `δk::T`, numeric value that corresponds to a scaling factor for the step size.
- `α0k::T`, numeric value. In the context of 
    [non-sequential armijo gradient descent](@ref NonsequentialArmijoGD)
    this is the first step size used in an inner loop.

# Return

- `flag::Bool`, `true` if the descent condition is satisfied, and `false`
    if the descent condition is not satisfied.
"""
function non_sequential_armijo_condition(F_ψjk::T, reference_value::T, 
    norm_grad_θk::T, ρ::T, δk::T, α0k::T) where {T}

    return (F_ψjk < reference_value - ρ * δk * α0k * (norm_grad_θk ^ 2))
end