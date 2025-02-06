# Date: 02/05/2025
# Author: Christian Varner
# Purpose: Helper functions that will implement wolfe line search
#
# Methodology:
#   The helper function implements the Extrapolation-Bisection Line Search
#   routine (EBLS) as described in "Optimization for Data Analysis" by 
#   Stephen J. Wright and Benjamin Recht


"""
    EBLS!(θk::Vector{T}, θkm1::Vector{T}, optData::AbstractOptimizerData{T, S},
        precomp::AbstractPrecompute{T}, store::AbstractProblemAllocate{T},
        gkm1::Vector{T}, step_direction::Vector{T}, reference_value::T,
        α::T, δ::T, c1::T, c2::T; max_iterations::Int64 = 100) where {T, S}

Implementation of an algorithm that finds a step size satisfying the weak
    wolfe condition. The function modifies `θk` in place, and updates
    the values in `store` related to the gradient. If `gkm1` is the step
    direction, use the other `EBLS!(...)` method to save on a dot product
    computation. This method will return a boolean flag which indicates
    whether the method was successful or not.

# Reference(s)

[Wright, Stephen J., and Benjamin Recht. "Optimization for Data Analysis." 
    Cambridge: Cambridge University Press, 2022. Print.](@cite wright2020Optimization)

# Method

Suppose an optimization method is trying to solve
```math
    \\min_{\\theta \\in \\mathbb{R}^n} F(\\theta),
```
through generating an iterate sequence defined by
```math
    \\theta_{k} = \\theta_{k-1} - \\alpha_{k-1} d_{k-1},~k \\in \\mathbb{N} 
```
where ``d_k`` is gradient related. **TODO: define this.**

To select the step size, one can perform the Extrapolation-Bisection Line Search
Routine (EBLS) which finds an ``\\alpha_{k-1}`` that satisfies two conditions which
are collectively known as the weak Wolfe conditions. The first condition is
the sufficient descent condition which enforces that

```math
    F(\\theta_k) \\leq F(\\theta_{k-1}) -
        c1 * \\alpha_{k-1} * \\dot F(\\theta_{k-1})^\\intercal d_{k-1},
```
and the second condition is the curvature condition which enforces that

```math
    \\dot F(\\theta_k)^\\intercal (-d_k) \\geq 
        c_2 \\dot F(\\theta_{k-1})^\\intercal (-d_{k-1}).
```

This is done by maintaining a guess ``\\alpha`` and a range
``\\alpha \\in [L, U]`` of possible values. The upper and lower bounds
are updated similar to a binary search algorithm. Initially, ``L = 0``,
``U = \\infty``, and (commonly) ``\\alpha = 1``. When the first
condition is not satisfied, ``U = \\alpha`` and the algorithms next guess
is ``\\alpha = (U + L)/2``. If the first condition is satisfied, but
the second is not then ``L = \\alpha``, and the algorithms next guess
for the step size is ``\\delta \\alpha`` when ``U = \\infty`` and 
``(U + L)/2`` when ``U < \\infty``. Here ``\\delta > 1``.

# Arguments

- `θk::Vector{T}`, buffer array for the next iterate. The point for which the
    weak Wolfe conditions are satisfied.
- `θkm1::Vector{T}`, the current iterate of an optimization algorithm. 
    The point at which a step is taken.
- `progData::AbstractNLPModel{T, S}`, data specifying an optimization problem.
    This is used to compute the function and gradient value at `θk`.
- `precomp::AbstractPrecompute{T}`, precomputed values for the optimization 
    problem.
- `store::AbstractProblemAllocate{T}`, storage data type for the optimization
    problem.
- `gkm1::Vector{T}`, gradient value at `θkm1`.
- `step_direction::Vector{T}`, direction to move `θkm1`.
- `reference_value::T`, reference value to compare the objective value
    at `θk` against. This will commonly be the objective value at `θk`.
- `α::T`, initial step size for the method.
- `δ::T`, inflation factor used to increase the step size when the
    curvature condition is not satisfied.
- `c1::T`, factor involved in the descent condition,
- `c2::T`, factor involved in the curvature condition.

## Optional Keyword Arguments

- `max_iterations::Int64 = 100`, maximum number of trial step sizes that
    are computed.

# Returns

- `wolfe_condition_satisfied::Bool`, boolean flag indicated whether the 
    vector `θk` satisfies the weak Wolfe conditions.
"""
function EBLS!(
    θk::Vector{T},
    θkm1::Vector{T},
    progData::AbstractNLPModel{T, S},
    precomp::AbstractPrecompute{T},
    store::AbstractProblemAllocate{T},
    gkm1::Vector{T},
    step_direction::Vector{T},
    reference_value::T,
    α::T,
    δ::T,
    c1::T,
    c2::T;
    max_iterations::Int64 = 100
) where {T, S}

    # values necessary for wolfe
    L, U = 0, Inf
    gkm1_dot_step = dot(gkm1, -step_direction)

    # try to satisfy wolfe conditions
    iter = 0
    wolfe_condition_satisfied = false
    while (iter <= max_iterations) && (!wolfe_condition_satisfied)

        # update iteration counter
        iter += 1

        # check the wolfe conditions
        θk .= θkm1 - α .* step_direction
        Fk = OptimizationMethods.obj(progData, precomp, store, θk)

        ## sufficient descent condition
        if Fk <= reference_value + c1 * α * gkm1_dot_step

            ## curvature condition
            OptimizationMethods.grad!(progData, precomp, store, θk)
            if dot(store.grad, -step_direction) >= c2 * gkm1_dot_step
                wolfe_condition_satisfied = true
            else
                L = α
                if U == Inf
                    α = δ * L
                else
                    α = (L + U)/2
                end
            end
        else
            U = α
            α = (U + L)/2
        end 
    end

    return wolfe_condition_satisfied
end

"""
    EBLS!(θk::Vector{T}, θkm1::Vector{T}, progData::AbstractNLPModel{T, S},
        precomp::AbstractPrecompute{T}, store::AbstractProblemAllocate{T},
        gkm1::Vector{T}, norm_gkm1_squared::T, reference_value::T, α::T,
        δ::T, c1::T, c2::T; max_iterations::Int64 = 100) where {T, S}

Implementation of an algorithm that finds a step size satisfying the weak
    wolfe condition. The function modifies `θk` in place, and updates
    the values in `store` related to the gradient. A boolean flag which 
    indicates whether the method was successful or not is returned. 
    
!!! note
    This implementation assumes `-gkm1` is the step direction.

# Reference(s)

[Wright, Stephen J., and Benjamin Recht. "Optimization for Data Analysis." 
    Cambridge: Cambridge University Press, 2022. Print.](@cite wright2020Optimization)

# Method

Suppose an optimization method is trying to solve
```math
    \\min_{\\theta \\in \\mathbb{R}^n} F(\\theta),
```
through generating an iterate sequence defined by
```math
    \\theta_{k} = \\theta_{k-1} - \\alpha_{k-1} \\dot F(\\theta_{k-1}),
    ~k \\in \\mathbb{N}. 
```

To select the step size, one can perform the Extrapolation-Bisection Line Search
Routine (EBLS) which finds an ``\\alpha_{k-1}`` that satisfies two conditions which
are collectively known as the weak Wolfe conditions. The first condition is
the sufficient descent condition which enforces that

```math
    F(\\theta_k) \\leq F(\\theta_{k-1}) -
        c1 * \\alpha_{k-1} * ||\\dot F(\\theta_{k-1})||_2^2,
```
and the second condition is the curvature condition which enforces that

```math
    \\dot F(\\theta_k)^\\intercal (-\\dot F(\\theta_{k-1})) \\geq 
        -c_2 ||\\dot F(\\theta_{k-1})||_2^2.
```

This is done by maintaining a guess ``\\alpha`` and a range
``\\alpha \\in [L, U]`` of possible values. The upper and lower bounds
are updated similar to a binary search algorithm. Initially, ``L = 0``,
``U = \\infty``, and (commonly) ``\\alpha = 1``. When the first
condition is not satisfied, ``U = \\alpha`` and the algorithms next guess
is ``\\alpha = (U + L)/2``. If the first condition is satisfied, but
the second is not then ``L = \\alpha``, and the algorithms next guess
for the step size is ``\\delta \\alpha`` when ``U = \\infty`` and 
``(U + L)/2`` when ``U < \\infty``. Here ``\\delta > 1``.

# Arguments

- `θk::Vector{T}`, buffer array for the next iterate. The point for which the
    weak Wolfe conditions are satisfied.
- `θkm1::Vector{T}`, the current iterate of an optimization algorithm. 
    The point at which a step is taken.
- `progData::AbstractNLPModel{T, S}`, data specifying an optimization problem.
    This is used to compute the function and gradient value at `θk`.
- `precomp::AbstractPrecompute{T}`, precomputed values for the optimization 
    problem.
- `store::AbstractProblemAllocate{T}`, storage data type for the optimization
    problem.
- `gkm1::Vector{T}`, gradient value at `θkm1`.
- `norm_gkm1_squared::T`, norm squared of `gkm1`.
- `reference_value::T`, reference value to compare the objective value
    at `θk` against. This will commonly be the objective value at `θk`.
- `α::T`, initial step size for the method.
- `δ::T`, inflation factor used to increase the step size when the
    curvature condition is not satisfied.
- `c1::T`, factor involved in the descent condition,
- `c2::T`, factor involved in the curvature condition.

## Optional Keyword Arguments

- `max_iterations::Int64 = 100`, maximum number of trial step sizes that
    are computed.

# Returns

- `wolfe_condition_satisfied::Bool`, boolean flag indicated whether the 
    vector `θk` satisfies the weak Wolfe conditions.
"""
function EBLS!(
    θk::Vector{T},
    θkm1::Vector{T},
    progData::AbstractNLPModel{T, S},
    precomp::AbstractPrecompute{T},
    store::AbstractProblemAllocate{T},
    gkm1::Vector{T},
    norm_gkm1_squared::T,
    reference_value::T,
    α::T,
    δ::T,
    c1::T,
    c2::T;
    max_iterations::Int64 = 100
) where {T, S}

    # values necessary for wolfe
    L, U = 0, Inf

    # try to satisfy wolfe conditions
    iter = 0
    wolfe_condition_satisfied = false
    while (iter <= max_iterations) && (!wolfe_condition_satisfied)

        # update iteration counter
        iter += 1

        # check the wolfe conditions
        θk .= θkm1 - α .* gkm1
        Fk = OptimizationMethods.obj(progData, precomp, store, θk)

        ## sufficient descent condition
        if Fk <= reference_value - c1 * α * norm_gkm1_squared

            ## curvature condition
            OptimizationMethods.grad!(progData, precomp, store, θk)
            if dot(store.grad, -gkm1) >= c2 * (-norm_gkm1_squared)
                wolfe_condition_satisfied = true
            else
                ## if curvature incorrect then increase the step size
                L = α
                if U == Inf
                    α = δ * L
                else
                    α = (L + U)/2
                end
            end
        else
            ## If objective too large decrease the step size
            U = α
            α = (U + L)/2
        end 
    end

    return wolfe_condition_satisfied
end