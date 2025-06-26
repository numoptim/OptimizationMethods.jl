# Date: 2025/04/23
# Author: Christian Varner
# Purpose: Implement the Non-sequential Armijo Line search with
# BFGS steps and fixed step sizes

"""
    NonsequentialArmijoFixedDampedBFGSGD{T} <: AbstractOptimizerData{T}

Mutable struct that represents and parameterizes the non-sequential armijo
    globalization framework with fixed step size and damped BFGS steps. This
    struct also stores values during the progress of the optimization routine.
        
# Fields

- `name::String`, name of the solver for reference.
- `∇F_θk::Vector{T}`, buffer vector for the starting gradient of the inner loop
- `B_θk::Matrix{T}`, buffer matrix for the starting damped BFGS approximation
    before an inner loop.
- `Bjk::Matrix{T}`, buffer matrix for the damped BFGS approximation in the
    inner loop.
- `δBjk::Matrix{T}`, buffer matrix for the update term added to the BFGS 
    approximation.
- `rjk::Vector{T}`, buffer vector for the update term in the damped BFGS
    approximation.
- `sjk::Vector{T}`, buffer vector for a term used in the damped BFGS approximation.
    Should correspond to the difference of consecutive iterates in the 
    inner loop.
- `yjk::Vector{T}`, buffer vector for a term used in the damped BFGS approximation.
    Should correspond to the difference of gradient values between 
    consecutive iterates in the inner loop.
- `djk::Vector{T}`, buffer vector used to store the step used in the inner
    loop.
- `c::T`, initial factor used in the approximation of the Hessian.
- `β::T`, shift applied to the approximation of the hessian to ensure it is
    bounded away from zero.
- `norm_∇F_ψ::T`, the norm of the gradient of the current inner loop iterate.
- `α::T`, step size used in the inner loop.
- `δk::T`, scaling factor for the step size.
- `δ_upper::T`, upper bound on the scaling factor.
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `objective_hist::CircularVector{T, Vector{T}}`, buffer array of size     
    `window_size` that stores `window_size` previous objective values.
- `reference_value::T`, maximum value of `objective_hist`. This is the reference 
    objective value used in the line search procedure.
- `reference_value_index::Int64`, index of the maximum value that corresponds to the 
    reference objective value.
- `acceptance_cnt::Int64`, number of accepted iterates. Used to properly update
    `objective_hist`.
- `τ_lower::T `, lower bound on the gradient interval triggering event.
- `τ_upper::T`, upper bound on the gradient interval triggering event.
- `inner_loop_radius::T`, radius for the bounding ball constraint in the inner
    loop. 
- `inner_loop_max_iterations::Int64`, maximum number of iterations in an 
    inner loop.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    is at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
- `iter_hist::Vector{Vector{T}}`, store the iterate sequence as the algorithm 
    progresses. The initial iterate is stored in the first position.
- `grad_val_hist::Vector{T}`, stores the norm gradient values at each iterate. 
    The norm of the gradient evaluated at the initial iterate is stored in the 
    first position.
- `stop_iteration::Int64`, the iteration number the algorithm stopped on. The 
    iterate that induced stopping is saved at `iter_hist[stop_iteration + 1]`.

# Constructors

    NonsequentialArmijoFixedDampedBFGSGD(::Type{T}; x0::Vector{T}, c::T, β::T,
        α::T, δ0::T, δ_upper::T, ρ::T, M::Int64, inner_loop_radius::T,
        inner_loop_max_iterations::Int64, threshold::T, 
        max_iterations::Int64) where {T}

## Arguments

- `T::DataType`, specific data type used for calculations.

## Keyword Arguments

- `x0::Vector{T}`, initial starting point for the optimization algorithm.
- `c::T`, initial factor used in the approximation of the Hessian.
- `β::T`, shift applied to the approximation of the hessian to ensure it is
    bounded away from zero.
- `α::T`, step size used in the inner loop.
- `δ0::T`, initial scaling factor for the method
- `δ_upper::T`, upper bound on the scaling factor
- `ρ::T`, factor involved in the acceptance criterion in the line search
    procedure. Larger values correspond to stricter descent conditions, and
    smaller values correspond to looser descent conditions.
- `M::Int64`, number of previous objective values that are used
    to construct the reference value for the line search criterion.
- `inner_loop_radius::T`, radius for the bounding ball constraint in the inner
    loop.
- `inner_loop_max_iterations::Int64`, maximum number of iterations in an 
    inner loop.
- `threshold::T`, gradient threshold. If the norm gradient is below this, then 
    iteration stops.
- `max_iterations::Int64`, max number of iterations (gradient steps) taken by 
    the solver.
"""
mutable struct NonsequentialArmijoFixedDampedBFGSGD{T} <: AbstractOptimizerData{T}
    name::String
    ∇F_θk::Vector{T}
    B_θk::Matrix{T}
    Bjk::Matrix{T}
    δBjk::Matrix{T}
    rjk::Vector{T}
    sjk::Vector{T}
    yjk::Vector{T}
    djk::Vector{T}
    c::T
    β::T
    norm_∇F_ψ::T
    α::T
    δk::T
    δ_upper::T
    ρ::T
    objective_hist::CircularVector{T, Vector{T}}
    reference_value::T
    reference_value_index::Int64
    acceptance_cnt::Int64
    τ_lower::T 
    τ_upper::T
    inner_loop_radius::T
    inner_loop_max_iterations::Int64
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonsequentialArmijoFixedDampedBFGSGD(::Type{T};
    x0::Vector{T},
    c::T,
    β::T,
    α::T,
    δ0::T,
    δ_upper::T,
    ρ::T,
    M::Int64,
    inner_loop_radius::T,
    inner_loop_max_iterations::Int64,
    threshold::T,
    max_iterations::Int64
    ) where {T}

    # error checking

    # name for recording purposed
    name = "Gradient Descent with Triggering Events and Nonsequential"*
    " Armijo with fixed step size and Damped BFGS steps."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    # return the struct
    return NonsequentialArmijoFixedDampedBFGSGD{T}(
        name,
        zeros(T, d),
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        zeros(T, d),
        c,
        β,
        T(0),
        α,
        δ0,
        δ_upper,
        ρ,
        CircularVector(zeros(T, M)),
        T(0),
        -1,
        0,
        T(-1),
        T(-1),
        inner_loop_radius,
        inner_loop_max_iterations,
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    )
end

"""
    inner_loop!(ψjk::S, θk::S, optData::NonsequentialArmijoFixedDampedBFGSGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64; radius = 10, max_iteration = 100) where {T, S}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and
`store` in place. `ψjk` gets updated to be the terminal iterate of the inner loop.
This inner loop function uses a constant step size with quasi-newton steps
using the damped BFGS approximation the Hessian.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. Let 
``\\alpha \\in \\mathbb{R},\\alpha > 0`` be 
a user defined step size that remains fixed.

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\delta_k \\alpha d_i^k,
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``||\\psi_{j_k}^k - \\theta_k||_2 > 10``, 
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, 
3. ``j_k == 100``.

The vector ``d_i^k \\in \\mathbb{R}^n`` is defined as the Quasi-Newton step using
the [damped BFGS method](@ref OptimizationMethods.update_bfgs!). In particular,
let ``B_j^k \\in \\mathbb{R}^{n \\times n}`` be the damped BFGS matrix, then

```math
    d_k = B_k^{-1} \\nabla F(\\theta_k).
```

!!! note
    When we say an accepted iterate at time ``k+1 \\in \\mathbb{N}``, 
    ``\\psi_{j_k}^k`` must satisfy our non-sequential armijo condition.

If an inner loop iterate is rejected, we restart the quasi-newton approximation
from the last accepted iterate. In particular, let ``\\ell_1`` be the first
time such that ``\\theta_{\\ell_1} \\not= \\theta_0``, 
then ``B_0^k = B_0^0`` for ``k \\in [0, \\ell_1)`` (i.e., the starting
approximation). Let ``\\ell_j`` be the jth time this occurred for ``j \\in \\mathbb{N}``,
then ``B_0^k = B_{0}^{\\ell_{j-1}} = B_{j_{\\ell_{j-1}}}^{\\ell_{j-1}}`` 
for ``k \\in [\\ell_{j-1}, \\ell_{j})`` (i.e., we set it equal to the
last inner loop approximation for which the last iterate of the inner loop was
accepted).

We update the and solve this system at each iteration of the method.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::NonsequentialArmijoFixedDampedBFGSGD{T}`, `struct` that specifies 
    the optimization algorithm. Fields are modified during the inner loop.
- `progData::P1 where P1 <: AbstractNLPModel{T, S}`, `struct` that specifies the
    optimization problem. Fields are modified during the inner loop.
- `precomp::P2 where P2 <: AbstractPrecompute{T}`, `struct` that has precomputed
    values. Required to take advantage of this during the gradient computation.
- `store::P3 where P3 <: AbstractProblemAllocate{T}`, `struct` that contains
    buffer arrays for computation.
- `k::Int64`, outer loop iteration for computation of the local Lipschitz
    approximation scheme.

## Optional Keyword Arguments

- `radius = 10`, the radius of the bounding ball event. Should be kept at `10`,
    however we allow it to be a parameter for future purposes.
- `max_iteration = 100`, maximum number of allowable iteration of the inner loop.
    Should be kept at `100` as that is what is specified in the paper, but
    is useful to change for testing.

# Returns

- `j::Int64`, the iteration for which a triggering event evaluated to true.
"""
function inner_loop!(
    ψjk::S,
    θk::S,
    optData::NonsequentialArmijoFixedDampedBFGSGD{T},
    progData::P1 where P1 <: AbstractNLPModel{T, S},
    precomp::P2 where P2 <: AbstractPrecompute{T},
    store::P3 where P3 <: AbstractProblemAllocate{T},
    k::Int64;
    radius = 10,
    max_iteration = 100) where {T, S}

    # inner loop
    j::Int64 = 0
    optData.norm_∇F_ψ = optData.grad_val_hist[k]
    while ((norm(ψjk - θk) <= radius) &&
        (optData.τ_lower < optData.norm_∇F_ψ && optData.norm_∇F_ψ < optData.τ_upper) &&
        (j < max_iteration))

        # Increment the inner loop counter
        j += 1

        # store values for update
        optData.sjk .= -ψjk
        optData.yjk .= -store.grad

        # compute step
        optData.djk .= optData.Bjk \ store.grad

        # take step
        ψjk .-= (optData.δk * optData.α) .* optData.djk

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)

        # update approximation
        optData.sjk .+= ψjk
        optData.yjk .+= store.grad
        update_success = OptimizationMethods.update_bfgs!(
            optData.Bjk, optData.rjk, optData.δBjk,
            optData.sjk, optData.yjk; damped_update = true)

        optData.norm_∇F_ψ = norm(store.grad)
    end

    return j
end

"""
    nonsequential_armijo_fixed_damped_bfgs(
        optData::NonsequentialArmijoFixedDampedBFGSGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of the (non-monotone) non-sequential Armijo globalization
    framework using triggering events. The inner loop is composed of a
    constant step size Quasi-Newton step using the damped BFGS update. 
    The optimization algorithm is specified through `optData`, and
    applied to the problem `progData`.

# Reference(s)

[Nocedal and Wright, "Numerical Optimization". Springer. 2nd Edition. 
    Chapter 6 and 18.](@cite nocedal2006Numerical)

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. Let ``\\alpha`` be
a user selected constant step size.
The ``k+1^{th}`` iterate and parameters are produced by the following procedure. 

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\delta_k \\alpha d_i^k,
```
where the vector ``d_i^k \\in \\mathbb{R}^n`` is defined as the Quasi-Newton step using
the [damped BFGS method](@ref OptimizationMethods.update_bfgs!). In particular,
let ``B_j^k \\in \\mathbb{R}^{n \\times n}`` be the damped BFGS matrix, then

```math
    d_k = B_k^{-1} \\nabla F(\\theta_k).
```
For more information on how the damped BFGS method is updated in this method,
see the `inner_loop!` function documentation for 
`NonsequentialArmijoFixedDampedBFGSGD`.

Let ``j_k \\in \\mathbb{N}`` be the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``||\\psi_{j_k}^k - \\theta_k||_2 > 10``, 
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, 
3. ``j_k == 100``.

The next iterate and algorithmic parameters in `optData` are updated based on 
the result of the non-sequential Armijo condition
```math
    F(\\psi_{j_k}^k) \\leq 
    F(\\theta_k) - \\rho\\delta_k\\alpha_0^k||\\dot F(\\theta_k)||_2.
```

When this condition is not satisfied, the following quantities are updated.

1. The iterate ``\\psi_{j_k}^k`` is rejected, and ``\\theta_{k+1} = \\theta_k``
2. The scaling factor ``\\delta_{k+1} = .5\\delta_k``

When this condition is satisfied, the following quantities are updated.

1. The iterate ``\\psi_{j_k}^k`` is accepted, and ``\\theta_{k+1} = \\psi_{j_k}^k``.
2. The scaling factor is updated as ``\\delta_{k+1} = \\min(1.5*\\delta_k, \\bar\\delta)``
    when ``||\\dot F(\\psi_{j_k}^k)||_2 > \\tau_{\\mathrm{grad},\\mathrm{lower}}^k``,
    otherwise ``\\delta_{k+1} = \\delta_k``.
3. If ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, then 
    ``\\tau_{\\mathrm{grad},\\mathrm{lower}}^{k+1} = 
    ||\\dot F(\\psi_{j_k}^k)||_2/\\sqrt{2}`` and 
    ``\\tau_{\\mathrm{grad},\\mathrm{upper}}^{k+1} = 
    \\sqrt{10}||\\dot F(\\psi_{j_k}^k)||_2``. Otherwise, this parameters are held
    constant.

# Arguments

- `optData::NonsequentialArmijoFixedDampedBFGSGD{T}`, the specification 
    for the optimization method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.

# Return

- `x::S`, final iterate of the optimization algorithm.
"""
function nonsequential_armijo_fixed_damped_bfgs(
    optData::NonsequentialArmijoFixedDampedBFGSGD{T},
    progData::P where P <: AbstractNLPModel{T, S}
) where {T, S}

    # initialize the problem data nd save initial values
    precomp, store = OptimizationMethods.initialize(progData)
    F(θ) = OptimizationMethods.obj(progData, precomp, store, θ)
    
    # initial iteration
    iter = 0
    x = copy(optData.iter_hist[1])
    OptimizationMethods.grad!(progData, precomp, store, optData.iter_hist[1])
    optData.grad_val_hist[1] = norm(store.grad)

    # Initialize hessian approximation
    fill!(optData.Bjk, 0)
    OptimizationMethods.add_identity(optData.Bjk,
        optData.c * optData.grad_val_hist[iter + 1])

    # update constants needed for triggering events
    optData.τ_lower = optData.grad_val_hist[1] / sqrt(2)  
    optData.τ_upper = sqrt(10) * optData.grad_val_hist[1]   

    # Initialize the objective history
    M = length(optData.objective_hist)
    optData.acceptance_cnt += 1
    optData.objective_hist[1] = F(optData.iter_hist[1]) 
    optData.reference_value, optData.reference_value_index = 
        optData.objective_hist[1], 1

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration counter
        iter += 1

        # inner loop
        optData.∇F_θk .= store.grad
        optData.B_θk .= optData.Bjk
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter; radius = optData.inner_loop_radius,
            max_iteration = optData.inner_loop_max_iterations)

        # check non-sequential armijo condition
        Fx = F(x)
        achieved_descent = 
        OptimizationMethods.non_sequential_armijo_condition(Fx, optData.reference_value, 
            optData.grad_val_hist[iter], optData.ρ, optData.δk, optData.α)
        
        # update the algorithm parameters and current iterate
        update_algorithm_parameters!(x, optData, achieved_descent, iter)
        
        # update histories
        optData.iter_hist[iter + 1] .= x
        if achieved_descent
            # accepted case
            optData.acceptance_cnt += 1
            optData.objective_hist[optData.acceptance_cnt] = Fx
            if ((optData.acceptance_cnt - 1) % M) + 1 == optData.reference_value_index
                optData.reference_value, optData.reference_value_index =
                findmax(optData.objective_hist)
            end

            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
        else
            # rejected case
            store.grad .= optData.∇F_θk
            optData.Bjk .= optData.B_θk
            optData.grad_val_hist[iter + 1] = optData.grad_val_hist[iter]
        end
    end

    optData.stop_iteration = iter

    return x

end