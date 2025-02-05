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

# Method

# Arguments

# Returns
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
    the values in `store` related to the gradient. This implementation assumes
    `-gkm1` is the step direction. A boolean flag which indicates whether the
    method was successful or not is returned.

# Reference(s)

# Method

# Arguments

# Returns
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