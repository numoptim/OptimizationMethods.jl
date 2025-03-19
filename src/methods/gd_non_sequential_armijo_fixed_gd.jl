# Date: 2025/03/18
# Purpose: Implemented our method with
# Non sequential armijo condition

"""
    NonsequentialArmijoFixedGD{T} <: AbstractOptimizerData{T}

A mutable struct that represents gradient descent with non-sequential armijo
    line search and triggering events. The inner loop is composed of a fixed
    step size and negative gradient directions.

# Fields

- `name::String`, name of the optimizer for reference.
- `∇F_θk::Vector{T}`, buffer array for the gradient of the initial inner
    loop iterate.
- `norm_∇F_ψ::T`, norm of the gradient of the current inner loop iterate.
- `α::T`, fixed step size value. 
- `δk::T`, scaling factor used to condition the step size.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `objective_hist::Vector{T}`, 
- `reference_value::T`,
- `reference_value_index::T`
- `τ_lower::T`, lower bound on the gradient interval triggering event.
- `τ_upper::T`, upper bound on the gradient interval triggering event.
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

    NonsequentialArmijoFixedGD(::Type{T}; x0::Vector{T}, δ0::T, δ_upper::T, ρ::T,
        threshold::T, max_iterations::Int64) where {T}

Constructs an instance of type `NonsequentialArmijoFixedGD{T}`.

## Arguments

- `T::DataType`, type for data and computation.

## Keyword Arguments

- `x0::Vector{T}`, initial point to start the optimization routine. Saved in
    `iter_hist[1]`.
- `α::T`, fixed step size for the inner loop.
- `δ0::T`, starting scaling factor.
- `δ_upper::T`, upper limit imposed on the scaling factor when updating.
- `ρ::T`, parameter used in the non-sequential Armijo condition. Larger
    numbers indicate stricter descent conditions. Smaller numbers indicate
    less strict descent conditions.
- `threshold::T`, norm gradient tolerance condition. Induces stopping when norm 
    at most `threshold`.
- `max_iterations::Int64`, max number of iterates that are produced, not 
    including the initial iterate.
"""
mutable struct NonsequentialArmijoFixedGD{T} <: AbstractOptimizerData{T}
    name::String
    ∇F_θk::Vector{T}
    norm_∇F_ψ::T
    α::T
    δk::T
    δ_upper::T
    ρ::T
    objective_hist::Vector{T}
    reference_value::T
    reference_value_index::T
    τ_lower::T
    τ_upper::T
    threshold::T
    max_iterations::Int64
    iter_hist::Vector{Vector{T}}
    grad_val_hist::Vector{T}
    stop_iteration::Int64
end
function NonsequentialArmijoFixedGD(
    ::Type{T};
    x0::Vector{T},
    α::T,
    δ0::T,
    δ_upper::T,
    ρ::T,
    M::T,
    threshold::T,
    max_iterations::Int64
) where {T}

    # error checking
    @assert 0 < δ0 "Initial scaling factor $(δ0) needs to be positive."

    @assert δ0 <= δ_upper "Initial scaling factor $(δ0) needs to be smaller"*
    " than its upper bound $(δ_upper)."

    @assert α > 0 "The fixed step size $(α) needs to be positive."

    # name for recording purposes
    name::String = "Gradient Descent with Triggering Events and Nonsequential"*
        "Armijo with fixed step size and gradient directions."

    # initialize buffer for history keeping
    d = length(x0)
    iter_hist::Vector{Vector{T}} = Vector{Vector{T}}([
        Vector{T}(undef, d) for i in 1:(max_iterations + 1)
    ])
    iter_hist[1] = x0

    grad_val_hist::Vector{T} = Vector{T}(undef, max_iterations + 1)
    stop_iteration::Int64 = -1 # dummy value

    return NonsequentialArmijoFixedGD(
        name,                       # name
        zeros(T, d),                # ∇F_θk
        T(0),                       # norm_∇F_ψ  
        α,                          # αk
        δ0,                         # δk
        δ_upper,                    # δ_upper
        ρ,                          # ρ
        zeros(T, M),                # objective_hist
        T(-1),                      # τ_lower
        T(-1),                      # τ_upper
        threshold,
        max_iterations,
        iter_hist,
        grad_val_hist,
        stop_iteration
    ) 
end

################################################################################
# Utility
################################################################################

"""
    inner_loop!(ψjk::S, θk::S, optData::NonsequentialArmijoFixedGD{T}, 
        progData::P1 where P1 <: AbstractNLPModel{T, S}, 
        precomp::P2 where P2 <: AbstractPrecompute{T}, 
        store::P3 where P3 <: AbstractProblemAllocate{T}, 
        k::Int64; max_iteration = 100) where {T, S}

Conduct the inner loop iteration, modifying `ψjk`, `optData`, and
`store` in place. `ψjk` gets updated to be the terminal iterate of the inner loop.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. Let ``\\alpha`` be 
a user defined step size that remains fixed.
The ``k+1^{th}`` iterate and parameters are produced by the following procedure. 

Let ``\\psi_0^k = \\theta_k``, then this method returns
```math
    \\psi_{j_k}^k = \\psi_0^k - \\sum_{i = 0}^{j_k-1} \\delta_k \\alpha \\dot F(\\psi_i^k),
```
where ``j_k \\in \\mathbb{N}`` is the smallest iteration for which at least one of the
conditions are satisfied: 

1. ``||\\psi_{j_k}^k - \\theta_k||_2 > 10``, 
2. ``||\\dot F(\\psi_{j_k}^k)||_2 \\not\\in (\\tau_{\\mathrm{grad},\\mathrm{lower}}^k,
    \\tau_{\\mathrm{grad},\\mathrm{upper}}^k)``, 
3. ``j_k == 100``.

# Arguments

- `ψjk::S`, buffer array for the inner loop iterates.
- `θk::S`, starting iterate.
- `optData::NonsequentialArmijoFixedGD{T}`, `struct` that specifies the optimization
    algorithm. Fields are modified during the inner loop.
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
    optData::NonsequentialArmijoFixedGD{T}, 
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

        # take step
        ψjk .-= (optData.δk * optData.α) .* store.grad

        ## store values for next iteration
        OptimizationMethods.grad!(progData, precomp, store, ψjk)
        optData.norm_∇F_ψ = norm(store.grad)
    end

    return j
end

"""
    nonsequential_armijo_adaptive_gd(optData::NonsequentialArmijoFixedGD{T},
        progData::P where P <: AbstractNLPModel{T, S}) where {T, S}

Implementation of gradient descent with non-sequential armijo and triggering
    events. The optimization algorithm is specified through `optData`, and
    applied to the problem `progData`.

# Method

In what follows, we let ``||\\cdot||_2`` denote the L2-norm. 
Let ``\\theta_{k}`` for ``k + 1 \\in\\mathbb{N}`` be the ``k^{th}`` iterate
of the optimization algorithm. Let ``\\delta_{k}, 
\\tau_{\\mathrm{grad},\\mathrm{lower}}^k, \\tau_{\\mathrm{grad},\\mathrm{upper}}^k``
be the ``k^{th}`` parameters for the optimization method. 
The ``k+1^{th}`` iterate and parameters are produced by the following procedure. 

Let ``\\psi_0^k = \\theta_k``, and recursively define
```math
    \\psi_{j}^k = \\psi_0^k - \\sum_{i = 0}^{j-1} \\delta_k \\alpha \\dot F(\\psi_i^k).
```
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

- `optData::NesterovAcceleratedGD{T}`, the specification for the optimization 
    method.
- `progData<:AbstractNLPModel{T,S}`, the specification for the optimization
    problem.

!!! warning
    `progData` must have an `initialize` function that returns subtypes of
    `AbstractPrecompute` and `AbstractProblemAllocate`, where the latter has
    a `grad` argument.

# Return

- `x::S`, final iterate of the optimization algorithm.
"""
function nonsequential_armijo_fixed_gd(
    optData::NonsequentialArmijoFixedGD{T},
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

    # update constants needed for triggering events
    optData.τ_lower = optData.grad_val_hist[1] / sqrt(2)  
    optData.τ_upper = sqrt(10) * optData.grad_val_hist[1]   

    # Initialize the objective history
    M = length(optData.objective_hist)
    optData.objective_hist[M] = F(optData.iter_hist[1]) 
    optData.reference_value, optData.reference_value_index = 
        optData.objective_hist[M], M 

    while (iter < optData.max_iterations) && 
        (optData.grad_val_hist[iter + 1] > optData.threshold)

        # Increment iteration counter
        iter += 1

        # inner loop
        optData.∇F_θk .= store.grad
        inner_loop!(x, optData.iter_hist[iter], optData, progData,
            precomp, store, iter)

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
            shift_left!(optData.objective_hist, M)
            optData.objective_hist[M] = Fx
            optData.reference_value, optData.reference_value_index = 
                update_maximum(optData.objective_hist, optData.reference_value_index-1, M)

            optData.grad_val_hist[iter + 1] = optData.norm_∇F_ψ
        else
            # rejected case
            store.grad .= optData.∇F_θk
            optData.grad_val_hist[iter + 1] = optData.grad_val_hist[iter]
        end
    end

    optData.stop_iteration = iter

    return x
end

