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

## Util
include("simple_stats.jl")

## Specific Problems 
include("problems/gaussian_least_squares.jl")

## Optimization Routines
include("optimization_routines/barzilai_borwein_gd.jl")

end
