module OptimizationMethods

############################
# Packages Includes
############################
using NLPModels, LinearAlgebra

############################
# Exports 
############################

# For objectives
export initialize

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
include("objective_functions/gaussian_least_squares.jl")

# quasi-likelihood 

end
