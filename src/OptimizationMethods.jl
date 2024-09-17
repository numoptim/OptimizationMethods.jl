module OptimizationMethods

############################
# Packages Includes
############################
using NLPModels, LinearAlgebra

############################
# Exports 
############################

# For objectives
export obj, grad, grad!, hess, hess!

# For optimization methods
export barzilai_borwein_gd

############################
# Algorithm Includes
############################

# Objective function free methods
include("optimization_routines/barzilai_borwein_gd.jl")

# Methods that ensure descent 

############################
# Objective Includes
############################

# simple functions -- for testing code
include("objective_functions/simple_least_squares.jl")

# quasi-likelihood 

end
