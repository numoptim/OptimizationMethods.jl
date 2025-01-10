# Date: 1/10/2025
# Author: Christian Varner
# Purpose: Example of running first order adaptive trust region on
# a least squares problem

using OptimizationMethods

progData = OptimizationMethods.LeastSquares(Float64);
optData = OptimizationMethods.FirstOrderAdaptiveTrustRegionGD(
    Float64;
    x0 = randn(50),
    τ = 1.0,
    μ = 1/2, 
    ζ = 1.0,
    threshold = 1e-10,
    max_iterations = 1000
)

x = OptimizationMethods.first_order_adaptive_trust_region_gd(optData, progData);

# Compute objective and residual evals during optimization 
obj_evals = progData.counters.neval_obj
res_evals = progData.counters.neval_residual 

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
    Residual Evaluations: $res_evals
    Jacobian Evaluations: $(progData.counters.neval_jac_residual)
"""
)