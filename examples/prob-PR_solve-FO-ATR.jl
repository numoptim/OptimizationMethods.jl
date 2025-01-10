# Date: 01/10/2025
# Author: Christian Varner
# Purpose: Example of using first order adaptive trust region
# method for poisson regression

using OptimizationMethods

progData = OptimizationMethods.PoissonRegression(Float64)
optData = OptimizationMethods.FirstOrderAdaptiveTrustRegionGD(
    Float64;
    x0 = randn(50),
    τ = 1.0,
    μ = 1/2, 
    ζ = 1.0,
    threshold = 1e-10,
    max_iterations = 1000
)

x = first_order_adaptive_trust_region_gd(optData, progData)

# Compute objective and residual evals during optimization 
obj_evals = progData.counters.neval_obj

# Compute objective values of different iterates for reporting purposes
obj_init = OptimizationMethods.obj(progData, optData.iter_hist[1])
obj_term = OptimizationMethods.obj(progData, 
    optData.iter_hist[optData.stop_iteration + 1])


println(
"""
    Problem: $(progData.meta.name)
    Solver: $(optData.name)
    Parameter Dimension: $(progData.meta.nvar)
    
    Max Iterations Allowed: $(optData.max_iterations)
    Gradient Stopping Threshold: $(optData.threshold)

    Initial Objective: $obj_init
    Initial Grad Norm: $(optData.grad_val_hist[1])

    Terminal Iteration: $(optData.stop_iteration)
    Terminal Objective: $obj_term
    Terminal Grad Norm: $(optData.grad_val_hist[optData.stop_iteration + 1])

    Objective Evaluations: $obj_evals
    Gradient Evaluations: $(progData.counters.neval_grad)
    Hessian Evaluations: $(progData.counters.neval_hess)
"""
)