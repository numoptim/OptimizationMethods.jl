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

## Helper functions
include("problems/regression_helpers/link_functions.jl")
include("problems/regression_helpers/variance_functions.jl")

## Source Code
include("problems/least_squares.jl")
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
export NesterovAcceleratedGD, nesterov_accelerated_gd
export DiminishingStepGD, diminishing_step_gd
export WeightedNormDampingGD, weighted_norm_damping_gd
export BacktrackingGD, backtracking_gd

## Helper functions for optimization methods
include("methods/stepsize_helpers/diminishing_stepsizes.jl")
include("methods/line_search_helpers/backtracking.jl")

## Source Code 
include("methods/gd_barzilai_borwein.jl")
include("methods/gd_fixed.jl")
include("methods/gd_lipschitz_approximation.jl")
include("methods/gd_nesterov_accelerated.jl")
include("methods/gd_diminishing.jl")
include("methods/gd_weighted_norm_damping.jl")
include("methods/gd_backtracking.jl")

end
