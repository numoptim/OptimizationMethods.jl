using OptimizationMethods

progData = OptimizationMethods.GaussianLeastSquares(Float64);
optData = OptimizationMethods.FixedStepGD(
    Float64, 
    x0=randn(50), 
    step_size=0.0005, 
    threshold=1e-10, 
    max_iterations=100
)

x = OptimizationMethods.fixed_step_gd(optData, progData);
obj = optData.obj_val_hist[optData.stop_iteration]
gra = optData.gra_val_hist[optData.stop_iteration]

println(
"""
    Problem: $(progData.meta.name)
    Solver: $(optData.name)
    Parameter Dimension: $(progData.meta.nvar)
    
    Max Iterations Allowed: $(optData.max_iterations)
    Gradient Stopping Threshold: $(optData.threshold)

    Initial Objective: $(optData.obj_val_hist[1])
    Initial Grad Norm: $(optData.gra_val_hist[1])

    Terminal Iteration: $(optData.stop_iteration)
    Terminal Objective: $(optData.obj_val_hist[optData.stop_iteration])
    Terminal Grad Norm: $(optData.gra_val_hist[optData.stop_iteration])

    Objective Evaluations: $(progData.counters.neval_obj)
    Gradient Evaluations: $(progData.counters.neval_grad)
    Residual Evaluations: $(progData.counters.neval_residual)
    Jacobian Evaluations: $(progData.counters.neval_jac_residual)
"""
)