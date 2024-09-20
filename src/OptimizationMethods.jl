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

############################
# Algorithm Includes
############################

# Objective function free methods

# Methods that ensure descent 

############################
# Objective Includes
############################

# simple functions -- for testing code
include("objective_functions/simple_least_squares.jl")

# quasi-likelihood 

end
