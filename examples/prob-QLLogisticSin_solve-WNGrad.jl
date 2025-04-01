# Part of OptimizationMethods.jl

using OptimizationMethods

progData = OptimizationMethods.QLLogisticSin(Float64)
optData = OptimizationMethods.WeightedNormDampingGD(
    Float64, 
    x0=randn(50), 
    init_norm_damping_factor = 10000.0,
    threshold=1e-10, 
    max_iterations=5000
)

x = OptimizationMethods.weighted_norm_damping_gd(optData, progData);

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
    Terminal Grad Norm: $(optData.grad_val_hist[optData.stop_iteration+1])

    Objective Evaluations: $obj_evals
    Gradient Evaluations: $(progData.counters.neval_grad)
"""
)