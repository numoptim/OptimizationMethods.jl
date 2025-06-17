# Date: 2025/06/17
# Author: Christian Varner
# Purpose: Example of solving a Poisson Regression problem using the 
# Line search modified Newton method.

using OptimizationMethods

# Initialize the Poisson Regression problem
progData = OptimizationMethods.PoissonRegression(Float64)
x0 = randn(50)

# Set up the optimizer data for the Line Search Modified Newton method (Monotone)
optData = FixedModifiedNewtonNLSMaxValGD(Float64;
    x0 = x0,
    α = 1.0,
    δ = 0.5,
    ρ = 1e-4,
    line_search_max_iteration = 100,
    window_size = 1,
    β = 1e-3,
    λ = 0.0,
    hessian_modification_max_iteration = 10,
    threshold = 1e-10,
    max_iterations = 100
)

# Solve the optimization problem
x = fixed_modified_newton_nls_maxval_gd(
    optData,
    progData
)

# Compute objective and residual evaluations during optimization
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])

# Print the results
println(
"""
    Problem: $(progData.meta.name)
    Solver: $(optData.name)
    Monotone: Yes
    Parameter Dimension: $(progData.meta.nvar)
    
    Max Iterations Allowed: $(optData.max_iterations)
    Gradient Stopping Threshold: $(optData.threshold)

    Initial Objective: $obj_init
    Initial Grad Norm: $(optData.grad_val_hist[1])

    Terminal Iteration: $(optData.stop_iteration)
    Terminal Objective: $obj_term
    Terminal Grad Norm: $(optData.grad_val_hist[optData.stop_iteration + 1])

    Objective Window: $(optData.objective_hist)

    Objective Evaluations: $obj_evals
    Gradient Evaluations: $(progData.counters.neval_grad)
    Hessian Evaluations: $(progData.counters.neval_hess)
"""
)

# reset for non-monotone version
progData.counters.neval_obj = 0
progData.counters.neval_grad = 0

# Set up the optimizer data for the Line Search Modified Newton method (Non-Monotone)
optData = FixedModifiedNewtonNLSMaxValGD(Float64;
    x0 = x0,
    α = 1.0,
    δ = 0.5,
    ρ = 1e-4,
    line_search_max_iteration = 100,
    window_size = 3,
    β = 1e-3,
    λ = 0.0,
    hessian_modification_max_iteration = 10,
    threshold = 1e-10,
    max_iterations = 100
)

# Solve the optimization problem
x = fixed_modified_newton_nls_maxval_gd(
    optData,
    progData
)

# Compute objective and residual evaluations during optimization
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])

# Print the results
println(
"""
    Problem: $(progData.meta.name)
    Solver: $(optData.name)
    Monotone: No
    Parameter Dimension: $(progData.meta.nvar)
    
    Max Iterations Allowed: $(optData.max_iterations)
    Gradient Stopping Threshold: $(optData.threshold)

    Initial Objective: $obj_init
    Initial Grad Norm: $(optData.grad_val_hist[1])

    Terminal Iteration: $(optData.stop_iteration)
    Terminal Objective: $obj_term
    Terminal Grad Norm: $(optData.grad_val_hist[optData.stop_iteration + 1])

    Objective Window: $(optData.objective_hist)

    Objective Evaluations: $obj_evals
    Gradient Evaluations: $(progData.counters.neval_grad)
    Hessian Evaluations: $(progData.counters.neval_hess)
"""
)