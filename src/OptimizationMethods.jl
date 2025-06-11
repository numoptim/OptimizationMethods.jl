module OptimizationMethods

# Dependencies
using CircularArrays
using LinearAlgebra
using NLPModels
using Distributions
using QuadGK: quadgk

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

"""
    AbstractDefaultQL{T, S} <: AbstractNLPModel{T, S}

Parametric type for default implementations of data structures for 
  quasi-likelihood problems.
"""
abstract type AbstractDefaultQL{T, S} <: AbstractNLPModel{T, S} end

"""
    AbstractDefaultQLPrecompute{T} <: AbstractPrecompute{T}
  
Parametric type for default implementations of data structures for
  precomputed values for quasi-likelihood optimization problems.
"""
abstract type AbstractDefaultQLPrecompute{T} <: AbstractPrecompute{T} end

"""
    AbstractDefaultQLAllocate{T} <: AbstractProblemAllocate{T}

Parametric type for default implementations of data structures that
  pre-allocate space for quasi-likelihood optimization problems.
"""
abstract type AbstractDefaultQLAllocate{T} <: AbstractProblemAllocate{T} end

## Helper functions
include("problems/regression_helpers/link_functions.jl")
include("problems/regression_helpers/link_function_derivatives.jl")
include("problems/regression_helpers/variance_functions.jl")
include("problems/regression_helpers/variance_functions_derivatives.jl")
include("problems/regression_helpers/quasi_likelihood_functionality.jl")

## Source Code
include("problems/least_squares.jl")
include("problems/logistic_regression.jl")
include("problems/poisson_regression.jl")
include("problems/ql_logistic_sin.jl")
include("problems/ql_logistic_centered_exp.jl")
include("problems/ql_logistic_centered_log.jl")
include("problems/ql_logistic_monomial.jl")

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
export FixedStepNonmonLSMaxValGD, fixed_step_nls_maxval_gd
export SafeBarzilaiBorweinNLSMaxValGD, safe_barzilai_borwein_nls_maxval_gd
export NonsequentialArmijoAdaptiveGD, nonsequential_armijo_adaptive_gd
export NonsequentialArmijoFixedGD, nonsequential_armijo_fixed_gd
<<<<<<< HEAD
export NonsequentialArmijoFixedMNewtonGD, nonsequential_armijo_mnewton_fixed_gd
=======
export NonsequentialArmijoSafeBBGD, nonsequential_armijo_safe_bb_gd
>>>>>>> main-v0.0.1

## Helper functions for optimization methods
include("methods/stepsize_helpers/diminishing_stepsizes.jl")
include("methods/line_search_helpers/backtracking.jl")
include("methods/line_search_helpers/non_sequential_armijo.jl")

### Helper functions for second order optimization methods
include("methods/second_order_helpers/modified_newton.jl")
include("methods/second_order_helpers/triangle_solve.jl")

## Source Code 
include("methods/gd_barzilai_borwein.jl")
include("methods/gd_fixed.jl")
include("methods/gd_lipschitz_approximation.jl")
include("methods/gd_nesterov_accelerated.jl")
include("methods/gd_diminishing.jl")
include("methods/gd_weighted_norm_damping.jl")
include("methods/gd_backtracking.jl")
include("methods/gd_fixed_nonmonotone_ls.jl")
include("methods/gd_safe_bb_nls_max_val.jl")
include("methods/gd_non_sequential_armijo_adaptive.jl")
include("methods/gd_non_sequential_armijo_fixed.jl")
<<<<<<< HEAD
include("methods/gd_non_sequential_armijo_fixed_mnewton.jl")
=======
include("methods/gd_non_sequential_armijo_safe_bb.jl")
>>>>>>> main-v0.0.1

end
