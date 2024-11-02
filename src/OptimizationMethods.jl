module OptimizationMethods

# Dependencies
using LinearAlgebra
using NLPModels

################################################################################
# Optimization Problems 
################################################################################

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

## Source Code
include("problems/gaussian_least_squares.jl")


################################################################################
# Optimization Methods 
################################################################################

## Data Structures
"""
  AbstractOptimizerData{T}

Parametric abstract type for storing parameters and progress of an optimizer.
"""
abstract type AbstractOptimizerData{T} end

## Methods
export BarzilaiBorweinGD, barzilai_borwein_gd
export FixedStepGD, fixed_step_gd

## Source Code 
include("methods/barzilai_borwein_gd.jl")
include("methods/gd_fixed.jl")


end
