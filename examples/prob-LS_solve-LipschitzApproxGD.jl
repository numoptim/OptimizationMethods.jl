# Date: 10/04/2024
# Author: Christian Varner
# Purpose: Implement an example of running lipschitz_approximation_gd() on 
# an instance of GaussianLeastSquares.

using OptimizationMethods

progData = OptimizationMethods.LeastSquares(Float64);
optData = OptimizationMethods.LipschitzApproxGD(
    Float64, 
    x0=randn(50), 
    init_stepsize=0.0005, 
    threshold=1e-10, 
    max_iterations=100
)

x = OptimizationMethods.lipschitz_approximation_gd(optData, progData);

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