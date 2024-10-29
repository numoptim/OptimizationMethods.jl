module OptimizationMethods

# Dependencies
using LinearAlgebra
using NLPModels

# Optimization Problems 

## Data Structures
"""
    AbstractPrecompute{T}

Parametric type for storing precomputed values of an optimization problem. 
"""
abstract type AbstractPrecompute{T} end

"""
    AbstractProblemAllocate{T}

Parametric type for pre-allocating data structures for an optimization problem.
"""
abstract type AbstractProblemAllocate{T} end

## Specific Problems 
include("problems/gaussian_least_squares.jl")


abstract type AbstractOptimizerData{T} end

## Exports - Optimizers
export FixedStepGD, fixed_step_gd


## Specific Optimizers
include("methods/gd_fixed.jl")

end
