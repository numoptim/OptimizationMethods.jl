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
export diminishing_step_size_gd

############################
# Algorithm Includes
############################

# step size util 
include("optimization_routines/step_size_util/inverse_k.jl")
include("optimization_routines/step_size_util/root_k.jl")

# Objective function free methods
include("optimization_routines/barzilai_borwein_gd.jl")

include("optimization_routines/diminishing_step_size_gd.jl")
# Methods that ensure descent 

############################
# Objective Includes
############################

# simple functions -- for testing code
include("objective_functions/simple_least_squares.jl")

# quasi-likelihood 

end
