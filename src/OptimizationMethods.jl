module OptimizationMethods

# Dependencies
using LinearAlgebra
using NLPModels
using Distributions

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
include("problems/logistic_regression.jl")
include("problems/poisson_regression.jl")


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
export LipschitzApproxGD, lipschitz_approximation_gd

## Source Code 
include("methods/gd_barzilai_borwein.jl")
include("methods/gd_fixed.jl")
include("methods/gd_lipschitz_approximation.jl")

end
